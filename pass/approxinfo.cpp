#include "accept.h"
#include "llvm/Pass.h"
#include "llvm/Metadata.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/DebugInfo.h"
#include "llvm/Module.h"

#include <fstream>

using namespace llvm;


/**** DESCRIBE SOURCE INFORMATION ****/

// Format a source position.
std::string llvm::srcPosDesc(const Module &mod, const DebugLoc &dl) {
  LLVMContext &ctx = mod.getContext();
  std::string out;
  raw_string_ostream ss(out);

  // Try to get filename from debug location.
  DIScope scope(dl.getScope(ctx));
  if (scope.Verify()) {
    ss << scope.getFilename().data();
  } else {
    // Fall back to the compilation unit name.
    ss << "(" << mod.getModuleIdentifier() << ")";
  }
  ss << ":";

  ss << dl.getLine();
  if (dl.getCol())
    ss << "," << dl.getCol();
  return out;
}

// Describe an instruction.
std::string llvm::instDesc(const Module &mod, Instruction *inst) {
  std::string out;
  raw_string_ostream ss(out);
  ss << srcPosDesc(mod, inst->getDebugLoc()) << ": ";

  // call and invoke instructions
  Function *calledFunc = NULL;
  bool isCall = false;
  if (CallInst *call = dyn_cast<CallInst>(inst)) {
    calledFunc = call->getCalledFunction();
    isCall = true;
  } else if (InvokeInst *invoke = dyn_cast<InvokeInst>(inst)) {
    calledFunc = invoke->getCalledFunction();
    isCall = true;
  }

  if (isCall) {
    if (calledFunc) {
      StringRef name = calledFunc->getName();
      if (!name.empty() && name.front() == '_') {
        // C++ name. An extra leading underscore makes the name legible by
        // c++filt.
        ss << "call to _" << name;
      } else {
        ss << "call to " << name << "()";
      }
    }
    else
      ss << "indirect function call";
  } else if (StoreInst *store = dyn_cast<StoreInst>(inst)) {
    Value *ptr = store->getPointerOperand();
    StringRef name = ptr->getName();
    if (!name.empty() && name.front() == '_') {
      ss << "store to _" << ptr->getName().data();
    } else {
      ss << "store to " << ptr->getName().data();
    }
  } else {
    inst->print(ss);
  }

  return out;
}


/**** ANALYSIS HELPERS ****/

// Look at the qualifier metadata on the instruction and determine whether it
// has approximate semantics.
bool isApprox(Instruction *instr) {
  MDNode *md = instr->getMetadata("quals");
  if (!md)
    return false;

  Value *val = md->getOperand(0);
  ConstantInt *ci = cast<ConstantInt>(val);
  if (ci) {
    APInt intval = ci->getValue();
    return intval == ECQ_APPROX;
  } else {
    llvm::errs() << "INVALID METADATA";
    return false;
  }
}

// An internal whitelist for functions considered to be pure.
char const* _funcWhitelistArray[] = {
  // math.h
  "cos",
  "sin",
  "tan",
  "acos",
  "asin",
  "atan",
  "atan2",
  "cosh",
  "sinh",
  "tanh",
  "exp",
  "frexp",
  "ldexp",
  "log",
  "log10",
  "modf",
  "pow",
  "sqrt",
  "ceil",
  "fabs",
  "floor",
  "fmod",
};
const std::set<std::string> funcWhitelist(
  _funcWhitelistArray,
  _funcWhitelistArray + sizeof(_funcWhitelistArray) /
                        sizeof(_funcWhitelistArray[0])
);



/**** ANALYSIS PASS WORKFLOW ****/

ApproxInfo::ApproxInfo() : FunctionPass(ID) {
  std::string error;
  log = new raw_fd_ostream("accept_log.txt", error,
                           raw_fd_ostream::F_Append);
}

ApproxInfo::~ApproxInfo() {
  log->close();
  delete log;
}

bool ApproxInfo::runOnFunction(Function &F) {
  errs() << "running on function " << F.getName() << "\n";
  return false;
}

bool ApproxInfo::doInitialization(Module &M) {
  // Analyze the purity of each function in the module up-front.
  for (Module::iterator i = M.begin(); i != M.end(); ++i) {
    isPrecisePure(&*i);
  }
  return false;
}

bool ApproxInfo::doFinalization(Module &M) {
  return false;
}


/**** ANALYSIS CORE ****/

void ApproxInfo::successorsOfHelper(BasicBlock *block,
                        std::set<BasicBlock*> &succ) {
  TerminatorInst *term = block->getTerminator();
  if (!term)
    return;
  for (int i = 0; i < term->getNumSuccessors(); ++i) {
    BasicBlock *sb = term->getSuccessor(i);
    if (!succ.count(sb)) {
      succ.insert(sb);
      successorsOfHelper(sb, succ);
    }
  }
}
std::set<BasicBlock*> ApproxInfo::successorsOf(BasicBlock *block) {
  // TODO memoize
  std::set<BasicBlock*> successors;
  successorsOfHelper(block, successors);
  return successors;
}

// Conservatively check whether a store instruction can be observed by any
// load instructions *other* than those in the specified set of instructions.
bool ApproxInfo::storeEscapes(StoreInst *store, std::set<Instruction*> insts) {
  Value *ptr = store->getPointerOperand();

  // Make sure the pointer was created locally. That is, conservatively assume
  // that pointers coming from arguments or returned from other functions are
  // aliased somewhere else.
  if (!isa<AllocaInst>(ptr))
    return true;

  // Give up if the pointer is copied and leaves the function. This could be
  // smarter if it only looked *after* the store (flow-wise).
  if (PointerMayBeCaptured(ptr, true, true))
    return true;

  // Look for loads to the pointer not present in our exclusion set. We
  // only look for loads in successors to this block. This could be made
  // more precise by detecting anti-dependencies (i.e., stores that shadow
  // this store).
  std::set<BasicBlock*> successors = successorsOf(store->getParent());
  for (std::set<BasicBlock*>::iterator bi = successors.begin();
        bi != successors.end(); ++bi) {
    for (BasicBlock::iterator ii = (*bi)->begin(); ii != (*bi)->end(); ++ii) {
      if (LoadInst *load = dyn_cast<LoadInst>(ii)) {
        if (load->getPointerOperand() == ptr && !insts.count(load)) {
          return true;
        }
      }
    }
  }

  return false;
}

int ApproxInfo::preciseEscapeCheckHelper(std::map<Instruction*, bool> &flags,
                              const std::set<Instruction*> &insts) {
  int changes = 0;
  for (std::map<Instruction*, bool>::iterator i = flags.begin();
      i != flags.end(); ++i) {
    // Only consider currently-untainted instructions.
    if (i->second) {
      continue;
    }

    // Precise store: check whether it escapes.
    if (StoreInst *store = dyn_cast<StoreInst>(i->first)) {
      if (!storeEscapes(store, insts)) {
        i->second = true;
        ++changes;
      }
      continue;
    }

    // Calls must be to precise-pure functions.
    Function *calledFunc = NULL;
    if (CallInst *call = dyn_cast<CallInst>(i->first)) {
      if (!isa<DbgInfoIntrinsic>(call)) {
        calledFunc = call->getCalledFunction();
        if (!calledFunc)
          continue;
      }
    } else if (InvokeInst *invoke = dyn_cast<InvokeInst>(i->first)) {
      calledFunc = invoke->getCalledFunction();
      if (!calledFunc)
        continue;
    }
    if (calledFunc && !isPrecisePure(calledFunc))
        continue;
    // Otherwise, the call itself is precise-pure, but now we need to make
    // sure that the uses are also tainted. Fall through to usage check.

    bool allUsesTainted = true;
    for (Value::use_iterator ui = i->first->use_begin();
          ui != i->first->use_end(); ++ui) {
      Instruction *user = dyn_cast<Instruction>(*ui);
      if (user && !flags[user]) {
        allUsesTainted = false;
        break;
      }
    }

    if (allUsesTainted) {
      ++changes;
      i->second = true;
    }
  }
  return changes;
}

bool ApproxInfo::hasPermit(Instruction *inst) {
  DebugLoc dl = inst->getDebugLoc();
  DIScope scope(dl.getScope(inst->getContext()));
  if (!scope.Verify())
    return false;

  // Read line N of the file.
  std::ifstream srcFile(scope.getFilename().data());
  int lineno = 1;
  std::string theLine;
  while (srcFile.good()) {
    std::string curLine;
    getline(srcFile, curLine);
    if (lineno == dl.getLine()) {
      theLine = curLine;
      break;
    } else {
      ++lineno;
    }
  }
  srcFile.close();

  return theLine.find(PERMIT) != std::string::npos;
}

bool ApproxInfo::approxOrLocal(std::set<Instruction*> &insts,
                               Instruction *inst) {
  Function *calledFunc = NULL;

  if (hasPermit(inst)) {
    return true;
  } else if (CallInst *call = dyn_cast<CallInst>(inst)) {

    if (isa<DbgInfoIntrinsic>(call)) {
      return true;
    }

    calledFunc = call->getCalledFunction();
    if (!calledFunc)  // Indirect call.
      return false;

  } else if (InvokeInst *invoke = dyn_cast<InvokeInst>(inst)) {
    calledFunc = invoke->getCalledFunction();
    if (!calledFunc)
      return false;
  } else if (isApprox(inst)) {
    return true;
  } else if (isa<StoreInst>(inst) ||
              isa<ReturnInst>(inst) ||
              isa<BranchInst>(inst)) {
    return false;  // Never approximate.
  }

  // For call and invoke instructions, ensure the function is precise-pure.
  if (calledFunc) {
    if (!isPrecisePure(calledFunc)) {
      return false;
    }
    if (isApprox(inst)) {
      return true;
    }
    // Otherwise, fall through and check usages for escape.
  }

  for (Value::use_iterator ui = inst->use_begin();
        ui != inst->use_end(); ++ui) {
    Instruction *user = dyn_cast<Instruction>(*ui);
    if (user && insts.count(user) == 0) {
      return false;  // Escapes.
    }
  }
  return true;  // Does not escape.
}

std::set<Instruction*> ApproxInfo::preciseEscapeCheck(
    std::set<Instruction*> insts) {
  std::map<Instruction*, bool> flags;

  // Mark all approx and non-escaping instructions.
  for (std::set<Instruction*>::iterator i = insts.begin();
        i != insts.end(); ++i) {
    flags[*i] = approxOrLocal(insts, *i);
  }

  // Iterate to a fixed point.
  while (preciseEscapeCheckHelper(flags, insts)) {}

  // Construct a set of untainted instructions.
  std::set<Instruction*> untainted;
  for (std::map<Instruction*, bool>::iterator i = flags.begin();
      i != flags.end(); ++i) {
    if (!i->second)
      untainted.insert(i->first);
  }
  return untainted;
}

std::set<Instruction*> ApproxInfo::preciseEscapeCheck(
    std::set<BasicBlock*> blocks) {
  std::set<Instruction*> insts;
  for (std::set<BasicBlock*>::iterator bi = blocks.begin();
        bi != blocks.end(); ++bi) {
    for (BasicBlock::iterator ii = (*bi)->begin();
          ii != (*bi)->end(); ++ii) {
      insts.insert(ii);
    }
  }
  return preciseEscapeCheck(insts);
}

// Determine whether a function can only affect approximate memory (i.e., no
// precise stores escape).
bool ApproxInfo::isPrecisePure(Function *func) {
  // Check for cached result.
  if (functionPurity.count(func)) {
    return functionPurity[func];
  }

  *log << "checking function _" << func->getName() << "\n";

  // LLVM's own nominal purity analysis.
  if (func->onlyReadsMemory()) {
    *log << " - only reads memory\n";
    functionPurity[func] = true;
    return true;
  }

  // Whitelisted pure functions from standard libraries.
  if (func->empty() && funcWhitelist.count(func->getName())) {
      *log << " - whitelisted\n";
      functionPurity[func] = true;
      return true;
  }

  // Empty functions (those for which we don't have a definition) are
  // conservatively marked non-pure.
  if (func->empty()) {
    *log << " - definition not available\n";
    functionPurity[func] = false;
    return false;
  }

  // Begin by marking the function as non-pure. This avoids an infinite loop
  // for recursive function calls (but is, of course, conservative).
  functionPurity[func] = false;

  std::set<BasicBlock*> blocks;
  for (Function::iterator bi = func->begin(); bi != func->end(); ++bi) {
    blocks.insert(bi);
  }
  std::set<Instruction*> blockers = preciseEscapeCheck(blocks);

  *log << " - blockers: " << blockers.size() << "\n";
  for (std::set<Instruction*>::iterator i = blockers.begin();
        i != blockers.end(); ++i) {
    *log << " - * " << instDesc(*(func->getParent()), *i) << "\n";
  }
  if (blockers.empty()) {
    *log << " - precise-pure function: _" << func->getName() << "\n";
  }

  functionPurity[func] = blockers.empty();
  return blockers.empty();
}

char ApproxInfo::ID = 0;