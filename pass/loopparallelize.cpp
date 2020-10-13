#include "accept.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <sstream>

using namespace llvm;

// Shamelessly stolen from LLVMs source code
template <typename T, size_t N> T *begin(T (&arr)[N]) { return arr; }
template <typename T, size_t N> T *end(T (&arr)[N]) { return arr + N; }

LoopParallelize::LoopParallelize() : LoopPass(ID) {}

void printLoop(Loop *L) {
  errs() << "printing loop\n";
  errs() << "@@@@@@@@@@@@@@@@@@@\n";
  for (Loop::block_iterator bit = L->block_begin(); bit != L->block_end();
       ++bit) {
    errs() << **bit << '\n';
  }
  errs() << "@@@@@@@@@@@@@@@@@@@\n";
}

bool LoopParallelize::isOnLoop(Instruction *inst, Loop *L) {
  for (Loop::block_iterator it = L->block_begin(); it != L->block_end(); ++it) {
    if (inst->getParent() == *it)
      return true;
  }
  return false;
}

bool LoopParallelize::isOnLoop(BasicBlock *bb, Loop *L) {
  for (Loop::block_iterator it = L->block_begin(); it != L->block_end(); ++it) {
    if (bb == *it)
      return true;
  }
  return false;
}

// Evaluate if instruction is in any of the body blocks
bool LoopParallelize::isOnLoopBody(Instruction *inst,
                                   std::vector<BasicBlock *> bodyBlocks) {
  for (std::vector<BasicBlock *>::iterator it = bodyBlocks.begin();
       it != bodyBlocks.end(); ++it) {
    if (inst->getParent() == *it)
      return true;
  }
  return false;
}

void LoopParallelize::recurseRemovefromLPM(Loop *L, LPPassManager &LPM) {
  std::vector<Loop *> subLoops = L->getSubLoops();
  for (std::vector<Loop *>::iterator sl = subLoops.begin();
       sl != subLoops.end(); ++sl) {
    recurseRemovefromLPM(*sl, LPM);
  }
  LPM.deleteLoopFromQueue(L);
}

// Adapted from LoopDeletion pass
void LoopParallelize::deleteLoop(Loop *L, LPPassManager &LPM) {
  // Remove subloops from queue, since they have been moved to cloned function
  std::vector<Loop *> subLoops = L->getSubLoops();
  for (std::vector<Loop *>::iterator sl = subLoops.begin();
       sl != subLoops.end(); ++sl) {
    recurseRemovefromLPM(*sl, LPM);
  }

  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();
  // Now that we know the removal is safe, remove the loop by changing the
  // branch from the preheader to go to the single exit block.
  BasicBlock *exitBlock = L->getExitBlock();
  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.  Don't
  // mess with this unless you have good reason and know what you're doing.

  // Tell ScalarEvolution that the loop is deleted. Do this before
  // deleting the loop so that ScalarEvolution can look at the loop
  // to determine what it needs to clean up.
  SE.forgetLoop(L);

  BasicBlock *preheader = L->getLoopPreheader();
  // Connect the preheader directly to the exit block.
  TerminatorInst *TI = preheader->getTerminator();
  TI->replaceUsesOfWith(L->getHeader(), exitBlock);

  // Rewrite phis in the exit block to get their inputs from
  // the preheader instead of the exiting block.
  SmallVector<BasicBlock *, 4> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);

  BasicBlock *exitingBlock = exitingBlocks[0];
  BasicBlock::iterator BI = exitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    int j = P->getBasicBlockIndex(exitingBlock);
    assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
    P->setIncomingBlock(j, preheader);
    for (unsigned i = 1; i < exitingBlocks.size(); ++i)
      P->removeIncomingValue(exitingBlocks[i]);
    ++BI;
  }

  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    // Remove the block from the reference counting scheme, so that we can
    // delete it freely later.
    (*LI)->dropAllReferences();
  }

  // Erase the instructions and the blocks without having to worry
  // about ordering because we already dropped the references.
  // NOTE: This iteration is safe because erasing the block does not remove its
  // entry from the loop's block list.  We do that in the next section.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI)
    (*LI)->eraseFromParent();

  // Finally, the blocks from loopinfo.  This has to happen late because
  // otherwise our loop iterators won't work.
  LoopInfo &loopInfo = getAnalysis<LoopInfo>();
  SmallPtrSet<BasicBlock *, 8> blocks;
  blocks.insert(L->block_begin(), L->block_end());
  for (SmallPtrSet<BasicBlock *, 8>::iterator I = blocks.begin(),
                                              E = blocks.end();
       I != E; ++I)
    loopInfo.removeBlock(*I);

  LPM.deleteLoopFromQueue(L);
}

// Search the loop and update values of the lower and upper bounds of the loop
// to be passed to .omp_outlined.
bool LoopParallelize::getLowerAndUpperBounds(Loop *L, Value *&lower,
                                             Value *&upper, bool &willInvert) {
  BasicBlock *header = L->getHeader();
  CmpInst *compInst = dyn_cast<CmpInst>(header->getTerminator()->getOperand(0));
  willInvert = false;
  if (compInst && compInst->isIntPredicate()) {
    unsigned int predicate = compInst->getPredicate();
    switch (predicate) {
    default: {
      return false;
    }
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGE:
    case CmpInst::ICMP_SGT: {
      lower = compInst->getOperand(1);
      upper = compInst->getOperand(0);
      // A > comparison indicates the loop counter must be altered in case it
      // is needed inside the loop on the acc_cloned function
      willInvert = true;
      return true;
    }
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE: {
      lower = compInst->getOperand(0);
      upper = compInst->getOperand(1);
      return true;
    }
    }
  }
  return false;
}

// If possibleLoad is a load or getelementptr instruction, return the
// pointer it loads from. If not, returns NULL.
Value *LoopParallelize::getPointerValue(Value *possibleLoad) {
  if (LoadInst *load = dyn_cast<LoadInst>(possibleLoad)) {
    return load->getPointerOperand();
  }
  if (GetElementPtrInst *getel = dyn_cast<GetElementPtrInst>(possibleLoad)) {
    return getel->getPointerOperand();
  }
  return NULL;
}

// Find the instructions that increment the loop
bool LoopParallelize::getIncrement(
    Loop *L, bool isForLike,
    SmallVector<Instruction *, 3> &incrementInstructions) {
  BasicBlock *header = L->getHeader();
  Instruction *incInst;
  // Assume the first instruction of the header loads the counter
  Value *counterPtr = cast<LoadInst>(header->front()).getPointerOperand();
  // Find a use of it OUTSIDE the loop header (but inside the loop). This should
  // lead to the increment instruction
  for (Value::use_iterator it = counterPtr->use_begin();
       it != counterPtr->use_end(); ++it) {
    Value *use = *it;
    Instruction *useOfInst = dyn_cast<StoreInst>(use);
    // Not a store, or does not store the counter
    if (!useOfInst || useOfInst->getOperand(1) != counterPtr)
      continue;
    // The instruction is not inside the latch of a for loop
    if (isForLike && useOfInst->getParent() != L->getLoopLatch())
      continue;
    // The instruction is on the header, or not on the loop at all
    if (!isForLike &&
        (useOfInst->getParent() == header || !isOnLoop(useOfInst, L)))
      continue;
    // It is a valid store, now lets check if it stores an add or sub
    // instruction
    Instruction *possibleIncOp =
        dyn_cast<Instruction>(useOfInst->getOperand(0));
    if (possibleIncOp) {
      unsigned int opcode = possibleIncOp->getOpcode();
      if (opcode == Instruction::Add || opcode == Instruction::Sub) {
        incInst = possibleIncOp;
        for (User::op_iterator oit = incInst->op_begin();
             oit != incInst->op_end(); ++oit) {
          Value *alloca = getPointerValue(*oit);
          if (alloca == counterPtr) {
            incrementInstructions.push_back(cast<Instruction>(*oit));
            break;
          }
        }
        incrementInstructions.push_back(incInst);
        incrementInstructions.push_back(useOfInst);
        // Add the load pointer instruction
        return true;
        break;
      }
    }
  }
  return false;
}

// Search every load and getelementptr inside the loop block. If the pointer
// operand is not inside the loop and is also not the counterPtr, store it.
void LoopParallelize::searchBodyPointers(BasicBlock *bodyBlock,
                                         std::set<Value *> &bodyPointers,
                                         std::vector<BasicBlock *> bodyBlocks,
                                         Loop *L) {
  BasicBlock *header = L->getHeader();
  Value *counterPtr = cast<LoadInst>(header->front()).getPointerOperand();
  for (BasicBlock::iterator bit = bodyBlock->begin(); bit != bodyBlock->end();
       ++bit) {
    Instruction *inst = &(*bit);
    // Test load or getelpointers instructions
    if (inst->getOpcode() == Instruction::GetElementPtr ||
        inst->getOpcode() == Instruction::Load) {
      Value *pValue = getPointerValue(inst);
      if (!pValue)
        continue;
      // Ignore the loop counter
      if (pValue == counterPtr)
        continue;
      Instruction *pointerInst = dyn_cast<Instruction>(pValue);
      // If it is an instruction defined inside the loop, skip it
      if (pointerInst && isOnLoop(pointerInst, L))
        continue;
      // Otherwise, store it to use as an argument on omp_outlined clone
      bodyPointers.insert(pValue);
    } else if (inst->getOpcode() == Instruction::Call) {
      // Iterate through call arguments in search of a pointer
      for (User::op_iterator opit = inst->op_begin(); opit != inst->op_end();
           ++opit) {
        // Stores allocas to be passed as argument
        if (isa<AllocaInst>(*opit))
          bodyPointers.insert(*opit);
      }
    }
  }
}

// if *pointer is really a IR pointer, create a load. Else, it is a constant.
// Just return it
Value *LoopParallelize::ensureLoad(Value *pointer, IRBuilder<> &builder) {
  if (dyn_cast<AllocaInst>(pointer))
    return builder.CreateLoad(pointer);
  return pointer;
}

// Create a function in which our omp.outlined will be cloned into
Function *LoopParallelize::createFunction(Function *ompFunc,
                                          const std::set<Value *> &allocaToArgs,
                                          llvm::ValueToValueMapTy &valueM) {
  Function *newFunc;
  Type *returnTy = Type::getVoidTy(module->getContext());
  std::vector<Type *> argTypeV;
  for (Function::arg_iterator it = ompFunc->arg_begin();
       it != ompFunc->arg_end(); ++it) {
    Type *argT = (*it).getType();
    argTypeV.push_back(argT);
  }

  for (std::set<Value *>::iterator it = allocaToArgs.begin();
       it != allocaToArgs.end(); ++it) {
    Value *inst = *it;
    argTypeV.push_back(inst->getType());
  }
  // Create the type array vector
  ArrayRef<Type *> typeArray(argTypeV);
  // The function type is ready
  FunctionType *fcType = FunctionType::get(returnTy, typeArray, false);
  newFunc =
      Function::Create(fcType, ompFunc->getLinkage(), "accept_cloned", module);
  Function::arg_iterator argNew = newFunc->arg_begin();
  // Set variables of the new function to carry the names of the old ones
  for (Function::arg_iterator it = ompFunc->arg_begin();
       it != ompFunc->arg_end(); ++it) {
    argNew->setName(it->getName());
    valueM[it] = argNew;
    argNew++;
  }
  // The rest carries the name of the alloca instructions to pass trough
  for (std::set<Value *>::iterator it = allocaToArgs.begin();
       it != allocaToArgs.end(); ++it) {
    argNew->setName((*it)->getName());
    argNew++;
  }
  return newFunc;
}

// Replace counterPtr with:
// If loop decrements: upperv - (plower*incr)
// If loop increments: plower*incr
Value *LoopParallelize::replaceCounter(Value *plower, Value *upperv, Value *incr,
                      bool isUnsigned, bool willInvert, IRBuilder<> &builder) {
  Value *toRet;
  Value *loadplower = builder.CreateLoad(plower);
  if (isUnsigned)
    toRet = builder.CreateMul(loadplower, incr);
  else
    toRet = builder.CreateNSWMul(loadplower, incr);
  if (willInvert) {
    if (isUnsigned)
      toRet = builder.CreateSub(upperv, toRet);
    else
      toRet = builder.CreateNSWSub(upperv, toRet);
  }
  return toRet;
}

bool LoopParallelize::runOnLoop(Loop *L, LPPassManager &LPM) {
  // Don't search skippable functions
  if (transformPass->shouldSkipFunc(*(L->getHeader()->getParent())))
    return false;
  Instruction *loopStart = L->getHeader()->begin();
  std::stringstream ss;
  ss << "ploop to paralellize at "
     << srcPosDesc(*module, loopStart->getDebugLoc());
  std::string loopName = ss.str();
  LogDescription *desc = AI->logAdd("loop paralellization", loopStart);
  ACCEPT_LOG << loopName << "\n";

  Instruction *inst = L->getHeader()->begin();
  Function *func = inst->getParent()->getParent();
  std::string funcName = func->getName().str();

  ACCEPT_LOG << "within function " << funcName << "\n";

  // Look for ACCEPT_FORBID marker.
  if (AI->instMarker(L->getHeader()->begin()) == markerForbid) {
    ACCEPT_LOG << "optimization forbidden\n";
    return false;
  }

  // We are looking for a loop with ONE header, who has a preheader and ONE exit
  // block
  if (!L->getHeader() || !L->getLoopLatch() || !L->getLoopPreheader() ||
      !L->getExitBlock()) {
    ACCEPT_LOG << "loop not in perforatable form\n";
    return false;
  }

  // Skip array constructor loops manufactured by Clang.
  if (L->getHeader()->getName().startswith("arrayctor.loop")) {
    ACCEPT_LOG << "array constructor\n";
    return false;
  }

  // Skip empty-body loops
  if (L->getNumBlocks() == 2 && L->getHeader() != L->getLoopLatch()) {
    BasicBlock *latch = L->getLoopLatch();
    if (&(latch->front()) == &(latch->back())) {
      ACCEPT_LOG << "empty body\n";
      return false;
    }
  }

  BasicBlock *loopHeader = L->getHeader();

  // The header must branch to two basic blocks. One is the start of the body,
  // the other is the exitBlock
  BranchInst *headerCond = dyn_cast<BranchInst>(loopHeader->getTerminator());
  if (!headerCond || headerCond->getNumSuccessors() != 2) {
    ACCEPT_LOG << "loop does not present valid branch condition\n";
    return false;
  }
  if (headerCond->getSuccessor(0) != L->getExitBlock() &&
      headerCond->getSuccessor(1) != L->getExitBlock()) {
    ACCEPT_LOG << "Header does not direct to exit block. Malformed loop\n";
    return false;
  }

  // Lets find the upper and lower bounds of the loop
  Value *lower, *upper;
  bool willInvert;

  if (!getLowerAndUpperBounds(L, lower, upper, willInvert)) {
    ACCEPT_LOG << "Loop does not present a valid increment comparison!\n";
    return false;
  }

  // Revert the lower and upper values to their respective alloca instructions
  // (if any)
  Value *tryLoad;
  tryLoad = getPointerValue(lower);
  if (tryLoad)
    lower = tryLoad;
  tryLoad = getPointerValue(upper);
  if (tryLoad)
    upper = tryLoad;

  // Check if loop originated from for or while
  bool isForLike;
  StringRef headerName = loopHeader->getName();
  if (headerName.startswith("for.cond")) {
    ACCEPT_LOG << "for-like loop\n";
    isForLike = true;
  } else if (headerName.startswith("while.cond")) {
    ACCEPT_LOG << "while-like loop\n";
    isForLike = false;
  } else {
    ACCEPT_LOG << " is not a while-like or for-like loop. Ignoring it\n";
    return false;
  }

  // Now, lets find the increment of the loop
  SmallVector<Instruction *, 3> incrementInstructions;
  bool foundIncrement = getIncrement(L, isForLike, incrementInstructions);
  Instruction *incInst = *(incrementInstructions.begin() + 1);
  if (!foundIncrement) {
    ACCEPT_LOG << "No valid increment operation (sum or subtraction) found "
                  "inside the loop. Cannot optimize\n";
    return false;
  }

  Value *increment = incInst->getOperand(1);
  // Check if fixed value (Constant)
  if (Instruction *possibleInst = dyn_cast<Instruction>(increment)) {
    increment = getPointerValue(possibleInst);
    if (isOnLoop(cast<Instruction>(increment), L)) {
      ACCEPT_LOG
          << "the increment is defined inside the loop, cannot parallelize\n";
      return false;
    }
  }

  // Get the body blocks of the loop
  std::vector<BasicBlock *> bodyBlocks;
  for (Loop::block_iterator bit = L->block_begin(); bit != L->block_end();
       ++bit) {
    BasicBlock *currentB = *bit;
    if (currentB == loopHeader)
      continue;
    if (isForLike && currentB == L->getLoopLatch())
      continue;
    bodyBlocks.push_back(currentB);
  }

  // Check whether the body of this loop is elidable (precise-pure).
  std::set<BasicBlock *> bodyBlocksSet(bodyBlocks.begin(), bodyBlocks.end());
  std::set<Instruction *> blockers = AI->preciseEscapeCheck(bodyBlocksSet);

  // Search all body blocks for pointers created outside the loop. They must
  // be passed as arguments on __kmpc_fork_call
  std::set<Value *> bodyPointers;
  for (std::vector<BasicBlock *>::iterator it = bodyBlocks.begin();
       it != bodyBlocks.end(); ++it) {
    searchBodyPointers(*it, bodyPointers, bodyBlocks, L);
  }

  // Parallelize loop
  if (transformPass->relax) {
    int param = transformPass->relaxConfig[loopName];
    if (param) {
      ACCEPT_LOG << "paralellizing with factor 2^" << param << "\n";
      viableLoop VL(L, lower, upper, increment, willInvert, isForLike,
                    incrementInstructions, bodyBlocks, bodyPointers);

      return paralellizeLoop(VL, LPM, param);
    } else {
      ACCEPT_LOG << "not paralellizing\n";
      return false;
    }
  }

  // Print the blockers to the log.
  for (std::set<Instruction *>::iterator i = blockers.begin();
       i != blockers.end(); ++i) {
    ACCEPT_LOG << *i;
  }

  if (!blockers.size()) {
    ACCEPT_LOG << "can paralellize loop\n";
    transformPass->relaxConfig[loopName] = 0;
  } else {
    ACCEPT_LOG << "cannot paralellize loop\n";
  }
  return false;
}

// Parallelization begins here. Return true if modified code (even if
// parallelization was not sucessfull)
bool LoopParallelize::paralellizeLoop(viableLoop VL, LPPassManager &LPM,
                                      int logthreads) {
  Loop *L = VL.L;
  BasicBlock *loopHeader = L->getHeader();
  BasicBlock *preHeader = L->getLoopPreheader();
  LoopInfo *LI = &getAnalysis<LoopInfo>();
  // Get the global variable, function prototype to clone and  kmpc_fork_call
  // function
  GlobalVariable *accStr = module->getGlobalVariable("accept_struct", true);
  Function *kmpcFC = module->getFunction("__kmpc_fork_call");
  Function *setNumThreads = module->getFunction("omp_set_num_threads");
  if (!accStr) {
    errs() << "cannot find the global variable accept_struct! Skipping "
              "paralellization\n";
    return false;
  }
  if (!kmpcFC) {
    errs() << "cannot find __kmpc_fork_call() prototype! Skipping "
              "paralellization\n";
    return false;
  }

  Value *lower = VL.lower;
  Value *upper = VL.upper;
  Value *increment = VL.increment;
  // Use CmpInst to evaluate the type of the increment operations
  TerminatorInst *headerTerm = loopHeader->getTerminator();
  CmpInst *compInst = dyn_cast<CmpInst>(headerTerm->getOperand(0));
  Type *lowerTy = lower->getType();
  if (lowerTy->isPointerTy())
    lowerTy = lowerTy->getPointerElementType();
  unsigned long bitmask = cast<IntegerType>(lowerTy)->getBitMask();
  std::string cloneName = "accept_omp_outlined";
  // Set string name of function
  // int32
  if (bitmask == 0xFFFFFFFF) {
    if (compInst->isUnsigned()) {
      cloneName += "u32";
    } else {
      cloneName += "u32";
    }
    // int64
  } else if (bitmask == 0xFFFFFFFFFFFFFFFF) {
    if (compInst->isUnsigned()) {
      cloneName += "u64";
    } else {
      cloneName += "s64";
    }
  } else {
    errs() << "Invalid bit size\n";
    return false;
  }

  Function *toClone = module->getFunction(cloneName);
  if (!toClone) {
    errs() << "accept_omp_outlined. function not found! Skipping "
              "paralellization\n";
    return false;
  }

  // If this is a while loop, the increment is in he body. Remove it before
  // inserting the BBs on the cloned function
  bool isForLike = VL.isForLike;
  if (!isForLike) {
    for (SmallVector<Instruction *, 3>::iterator it =
             VL.incrementInstructions.end() - 1;
         it >= VL.incrementInstructions.begin(); --it) {
      Instruction *toDel = *it;
      toDel->dropAllReferences();
      toDel->eraseFromParent();
    }
  }

  std::set<Value *> bodyPointers = VL.bodyPointers;
  // Clone the omp.untlined function
  llvm::ValueToValueMapTy valueM;
  SmallVector<ReturnInst *, 1> returns;
  // Create a function blank in which toClone will be inserted into
  Function *clonedF = createFunction(toClone, bodyPointers, valueM);
  llvm::CloneFunctionInto(clonedF, toClone, valueM, false, returns);
  Function::iterator beforeAddition = clonedF->begin();
  // This is the lower_or_eq BB
  std::advance(beforeAddition, 5);
  Function::iterator afterAddition = beforeAddition;
  afterAddition++;

  std::vector<BasicBlock *> bodyBlocks = VL.bodyBlocks;
  // Find the first and last basicblocks of the body

  // beginBody is the basic block that comes after the header
  // Ideally, this is the first BasicBlock when iterating on bodyBlocks, but we
  // must be safe if this is not the case
  BasicBlock *beginBody;
  if (isOnLoop(cast<BasicBlock>(headerTerm->getOperand(1)), L)) {
    beginBody = cast<BasicBlock>(headerTerm->getOperand(1));
  } else {
    beginBody = cast<BasicBlock>(headerTerm->getOperand(2));
  }
  // If this is a while-like loop, endBody is the latch. Otherwise, it is the
  // predecessor from the latch the loop
  BasicBlock *endBody;
  if (!isForLike) {
    endBody = L->getLoopLatch();
  } else {
    BasicBlock *exitB = L->getLoopLatch();
    // Iterate trhought latch predecessors to find the first one inside the loop
    for (pred_iterator PI = pred_begin(exitB); PI != pred_end(exitB); ++PI) {
      if (isOnLoop(*PI, L)) {
        endBody = *PI;
        break;
      }
    }
  }

  // Deal with debug calls and metadata
  std::vector<Instruction *> dbgToDel;
  for (std::vector<BasicBlock *>::iterator bit = bodyBlocks.begin();
       bit != bodyBlocks.end(); ++bit) {
    BasicBlock *b = *bit;
    for (BasicBlock::iterator ii = b->begin(); ii != b->end(); ++ii) {
      // Remove llvm.dbg.* calls to avoid metadata errors on cloned function
      CallInst *callFunc = dyn_cast<CallInst>(ii);
      if (callFunc &&
          callFunc->getCalledFunction()->getName().startswith("llvm.dbg")) {
        dbgToDel.push_back(callFunc);
      }
      // Remove debug data from moved instructions to avoid module check
      // errors (Should not happen anyway)
      ii->setDebugLoc(llvm::DebugLoc());
    }
  }

  for (std::vector<Instruction *>::iterator it = dbgToDel.begin();
       it != dbgToDel.end(); ++it) {
    (*it)->eraseFromParent();
  }

  // Move the blocks from the original loop to the new function. Remove all
  // blocks from the original loop
  beginBody->moveAfter(beforeAddition);
  // While iterating through bodyBlocks, remove them from all nested loops
  LI->removeBlock(beginBody);
  BasicBlock *lastMoved = beginBody;
  for (std::vector<BasicBlock *>::iterator it = bodyBlocks.begin();
       it != bodyBlocks.end(); ++it) {
    BasicBlock *bb = *it;
    if (bb != beginBody && bb != endBody) {
      LI->removeBlock(bb);
      bb->moveAfter(lastMoved);
      lastMoved = bb;
    }
  }
  LI->removeBlock(endBody);
  endBody->moveAfter(lastMoved);

  BranchInst *newJump;
  // Replace the branch from the BB before the replaceable to the first
  // bodyBlock
  newJump = BranchInst::Create(beginBody);
  llvm::ReplaceInstWithInst(beforeAddition->getTerminator(), newJump);
  // Replace the branch from the final bodyBlock to the cloned function
  newJump = BranchInst::Create(afterAddition);
  llvm::ReplaceInstWithInst(endBody->getTerminator(), newJump);

  // Avoid the header branching to the loop body, that is now on the cloned
  // function
  BasicBlock *toJump;
  if (isForLike)
    toJump = L->getLoopLatch();
  else
    toJump = L->getExitBlock();
  headerTerm->replaceUsesOfWith(beginBody, toJump);

  IRBuilder<> builder(module->getContext());

  // Find plower in clonedF
  BasicBlock *clonedEntry = &clonedF->getEntryBlock();
  Instruction *plower;
  for (BasicBlock::iterator it = clonedEntry->begin(); it != clonedEntry->end();
       ++it) {
    if (it->getName() == "plower") {
      plower = &*it;
      break;
    }
  }
  // Get upperv and incr
  Function::arg_iterator argit = clonedF->arg_begin();
  std::advance(argit, 3);
  Value *upperv = argit;
  std::advance(argit, 1);
  Value *incr = argit;

  Value *counterPtr = cast<LoadInst>(loopHeader->front()).getPointerOperand();
  Instruction *lastLoad = NULL;
  Value *lastOtherUse = NULL;
  std::set<Instruction *> replaceUsesLoad;
  std::set<Instruction *> replaceUsesOtherV;
  // Iterate throught the uses of counterPtr to find any that are now in
  // clonedF. Replace them with the proper value
  for (Value::use_iterator uit = counterPtr->use_begin();
       uit != counterPtr->use_end(); ++uit) {
    Instruction *useOfCtr = dyn_cast<Instruction>(*uit);
    if (useOfCtr && useOfCtr->getParent()->getParent() == clonedF) {
      // If the use is a load, replace it with plower
      if (isa<LoadInst>(useOfCtr)) {
        lastLoad = useOfCtr;
        // Add to a set that will be iterated after the loop
        replaceUsesLoad.insert(useOfCtr);
      }
      // Else, create a new pointer to replace it
      else {
        lastOtherUse = useOfCtr;
        replaceUsesOtherV.insert(useOfCtr);
      }
    }
  }

  if (lastLoad) {
    Value *replaceValue;
    // Create the replacement value for counterPtr on its earliest use
    builder.SetInsertPoint(lastLoad);
    Value *loadplower = builder.CreateLoad(plower);
    replaceValue = replaceCounter(plower, upperv, incr, compInst->isUnsigned(),
                                  VL.willInvert, builder);
    for (std::set<Instruction *>::iterator it = replaceUsesLoad.begin();
         it != replaceUsesLoad.end(); ++it) {
      Instruction *willRep = *it;
      // Replace the load with the current increment and remove it
      willRep->replaceAllUsesWith(replaceValue);
      willRep->eraseFromParent();
    }
  }
  if (lastOtherUse) {
    // Allocate a new pointer
    builder.SetInsertPoint(clonedF->getEntryBlock().begin());
    Value *replacePtr = builder.CreateAlloca(
        counterPtr->getType()->getPointerElementType(), NULL, "replacePtr");
    // Replace each use of the new pointer with the same combo from loads, but
    // in pointer form
    for (std::set<Instruction *>::iterator it = replaceUsesOtherV.begin();
         it != replaceUsesOtherV.end(); ++it) {
      Instruction *willRep = *it;
      builder.SetInsertPoint(willRep);
      Value *replaceValue = replaceCounter(
          plower, upperv, incr, compInst->isUnsigned(), VL.willInvert, builder);
      // Store operations on new pointer
      builder.CreateStore(replaceValue, replacePtr);
      // Replace use of CounterPtr with our modified pointer
      willRep->replaceUsesOfWith(counterPtr, replacePtr);
    }
  }

  // For each use of bodyPointers[i] that is now on the acc_cloned function,
  // replace its uses with the corresponding function argument
  argit = clonedF->arg_begin();
  std::advance(argit, toClone->arg_size());
  std::set<Instruction *> replaceUses;
  for (std::set<Value *>::iterator it = bodyPointers.begin();
       it != bodyPointers.end(); ++it, ++argit) {
    Value *bpAlloca = *it;
    replaceUses.clear();
    for (Value::use_iterator uit = bpAlloca->use_begin();
         uit != bpAlloca->use_end(); ++uit) {
      Instruction *usesBP = dyn_cast<Instruction>(*uit);
      if (usesBP && usesBP->getParent()->getParent() == clonedF) {
        // Store required replacements on a set as to not replace uses while
        // iterating over the use structure
        replaceUses.insert(usesBP);
      }
    }
    for (std::set<Instruction *>::iterator rit = replaceUses.begin();
         rit != replaceUses.end(); ++rit) {
      (*rit)->replaceUsesOfWith(bpAlloca, argit);
    }
  }

  std::vector<Instruction *> toMove;
  // Move instructions on the header to the preheader, since they may possibly
  // manipulate plower and pupper
  for (BasicBlock::iterator I = loopHeader->begin(); I != loopHeader->end();
       ++I) {
    // Stop at the comparison
    if (isa<CmpInst>(I))
      break;
    toMove.push_back(I);
  }
  for (std::vector<Instruction *>::iterator it = toMove.begin();
       it != toMove.end(); ++it) {
    (*it)->moveBefore(preHeader->getTerminator());
  }

  // We must allocate a new pointer to store the increment, which will be
  // manipulated
  builder.SetInsertPoint(loopHeader->getParent()->getEntryBlock().begin());
  Type *incrTy = incr->getType();
  if (incrTy->isPointerTy())
    incrTy = incrTy->getPointerElementType();
  Value *pincr = builder.CreateAlloca(incrTy, 0, "pincr");

  // Load the values for upper, lower and incr value
  builder.SetInsertPoint(preHeader->getTerminator());
  Value *lowerV = ensureLoad(lower, builder);
  Value *upperV = ensureLoad(upper, builder);
  Value *incrV = ensureLoad(increment, builder);
  // Store
  builder.CreateStore(incrV, pincr);
  Value *zero = ConstantInt::get(incrV->getType(), 0, true);
  if (compInst->isSigned()) {
    // Check if increment is negative at execution time
    CmpInst *negCmp =
        CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SLE, incrV, zero,
                        "negCmp", preHeader->getTerminator());
    // Create a new block where the value of the increment is inverted
    TerminatorInst *newBlockTerm =
        llvm::SplitBlockAndInsertIfThen(negCmp, false);
    // If the preheader is inside a loop, ensure the new blocks are inserted
    // into it
    Loop *parentLoop = L->getParentLoop();
    if (parentLoop) {
      parentLoop->addBasicBlockToLoop(newBlockTerm->getParent(), LI->getBase());
      parentLoop->addBasicBlockToLoop(
          cast<BasicBlock>(newBlockTerm->getOperand(0)), LI->getBase());
    }

    builder.SetInsertPoint(newBlockTerm);
    incrV = builder.CreateNSWSub(zero, incrV);
    builder.CreateStore(incrV, pincr);
    // Set builder to insert on final block (before loop header)
    preHeader = cast<BasicBlock>(newBlockTerm->getOperand(0));
    builder.SetInsertPoint(preHeader->getTerminator());
  }

  // Load pincr
  incrV = builder.CreateLoad(pincr);

  // Ensure incr type matches lower and upper (comparison type)
  unsigned long incrBitmask = cast<IntegerType>(incrTy)->getBitMask();
  if (incrBitmask != bitmask) {
    if (compInst->isUnsigned()) {
      incrV = builder.CreateZExtOrTrunc(incrV, cast<IntegerType>(lowerTy));
    } else {
      incrV = builder.CreateSExtOrTrunc(incrV, cast<IntegerType>(lowerTy));
    }
  }

  // Set number of threads to 2^(logthreads)
  int nthreads = 1 << logthreads;
  Value *nthrC = builder.getInt32(nthreads);
  builder.CreateCall(setNumThreads, nthrC);

  // Build the argument array
  Type *microParams[] = {PointerType::getUnqual(builder.getInt32Ty()),
                         PointerType::getUnqual(builder.getInt32Ty())};
  FunctionType *kmpc_MicroTy =
      FunctionType::get(builder.getVoidTy(), microParams, true);
  Type *kmpcRes = PointerType::getUnqual(kmpc_MicroTy);
  Value *bitcastV = builder.CreateBitCast(clonedF, kmpcRes);
  Value *args[] = {
      accStr,                                    // OpenMP struct
      builder.getInt32(3 + bodyPointers.size()), // number of total args
      bitcastV // cast cloned_acc to void (i32*, i32*, ...)*
  };
  Value *initArgs[] = {lowerV, upperV, incrV};
  // Final argument array
  SmallVector<Value *, 16> realArgs;
  // First batch of arguments
  realArgs.append(begin(args), end(args));
  // Second batch of arguments
  realArgs.append(begin(initArgs), end(initArgs));
  realArgs.append(bodyPointers.begin(), bodyPointers.end());
  // Create the call to __kmpc_fork_call
  builder.CreateCall(kmpcFC, realArgs);
  // Remove what remains of the original loop
  deleteLoop(L, LPM);
  return true;
}

bool LoopParallelize::doInitialization(Loop *L, LPPassManager &) {
  transformPass = (ACCEPTPass *)sharedAcceptTransformPass;
  AI = transformPass->AI;
  module = L->getHeader()->getParent()->getParent();
  return false;
}

bool LoopParallelize::doFinalization() { return false; }

void LoopParallelize::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();

  AU.addPreserved<ScalarEvolution>();
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<LoopInfo>();
}

const char *LoopParallelize::getPassName() const {
  return "Loop Parallelization";
}

LoopPass *llvm::createLoopParallelizePass() { return new LoopParallelize(); }

char LoopParallelize::ID = 0;
