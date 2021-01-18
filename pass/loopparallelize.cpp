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

bool LoopParallelize::isOnLoop(Instruction *inst) {
  for (Loop::block_iterator it = L->block_begin(); it != L->block_end(); ++it) {
    if (inst->getParent() == *it)
      return true;
  }
  return false;
}

bool LoopParallelize::isOnLoop(BasicBlock *bb) {
  for (Loop::block_iterator it = L->block_begin(); it != L->block_end(); ++it) {
    if (bb == *it)
      return true;
  }
  return false;
}

// Evaluate if instruction is in any of the body blocks
bool LoopParallelize::isOnLoopBody(Instruction *inst) {
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
void LoopParallelize::deleteLoop(LPPassManager &LPM) {
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

// Search the loop and update values of the lower and upper bounds to be passed
// to .omp_outlined.
bool LoopParallelize::getLowerAndUpperBounds() {
  BasicBlock *header = L->getHeader();
  CmpInst *compInst = dyn_cast<CmpInst>(header->getTerminator()->getOperand(0));
  if (compInst && compInst->isIntPredicate()) {
    this->willInvert = false;
    this->isUnsigned = compInst->isUnsigned();
    unsigned int predicate = compInst->getPredicate();
    switch (predicate) {
    default: {
      return false;
    }
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGE:
    case CmpInst::ICMP_SGT: {
      this->lower = compInst->getOperand(1);
      this->upper = compInst->getOperand(0);
      // A > comparison indicates the loop counter must be altered in case it
      // is needed inside the loop on the acc_cloned function
      this->willInvert = true;
      return true;
    }
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE: {
      this->lower = compInst->getOperand(0);
      this->upper = compInst->getOperand(1);
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
bool LoopParallelize::getIncrement() {
  BasicBlock *header = L->getHeader();
  Instruction *incInst;
  // If the loop does not attain the counter from a pointer, the pass will not
  // work
  if (!isa<LoadInst>(header->front())) {
    return false;
  } else {
    this->counterPtr = cast<LoadInst>(header->front()).getPointerOperand();
  }
  // Find a use of it OUTSIDE the loop header (but inside the loop). This should
  // lead to the increment instruction
  for (Value::use_iterator it = counterPtr->use_begin();
       it != counterPtr->use_end(); ++it) {
    Value *use = *it;
    StoreInst *useOfInst = dyn_cast<StoreInst>(use);
    // Not a store, or does not store the counter
    if (!useOfInst || useOfInst->getOperand(1) != counterPtr)
      continue;
    // The instruction is not inside the latch of a for loop
    if (isForLike && useOfInst->getParent() != L->getLoopLatch())
      continue;
    // The instruction is on the header, or not on the loop at all
    if (!isForLike &&
        (useOfInst->getParent() == header || !isOnLoop(useOfInst)))
      continue;
    // It is a valid store, now lets check if it stores an add or sub
    // instruction
    Instruction *possibleIncOp =
        dyn_cast<Instruction>(useOfInst->getValueOperand());
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
        return true;
      }
    }
  }
  return false;
}

// Search every load and getelementptr inside the loop block in search of
// out-of-loop value uses. If the operand is not inside the loop and is
// also not the counterPtr, store it.
void LoopParallelize::searchBodyPointers(BasicBlock *bodyBlock) {
  BasicBlock *header = L->getHeader();
  for (BasicBlock::iterator bit = bodyBlock->begin(); bit != bodyBlock->end();
       ++bit) {
    Instruction *inst = &(*bit);
    // Load and getelementptr instructions are treated differently
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
      if (pointerInst && isOnLoop(pointerInst))
        continue;
      // Otherwise, store it to use as an argument on omp_outlined clone
      bodyPointers.insert(pValue);
    } else {
      // Iterate through instruction operands in search of a pointer
      for (User::op_iterator opit = inst->op_begin(); opit != inst->op_end();
           ++opit) {
        Instruction *fromOp = dyn_cast<Instruction>(*opit);
        // If pointer was created inside the loop, leave it
        if (!fromOp || isOnLoop(fromOp))
          continue;
        // Stores pointer instructions to be passed as argument
        if (fromOp->getType()->isPointerTy())
          bodyPointers.insert(fromOp);
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

// Obtain plower, upperv and incr from accept_cloned function
void getFunctionPointers(Function *accept_cloned, Value *&plower,
                         Value *&upperv, Value *&incr) {
  // Find plower in clonedF
  BasicBlock *clonedEntry = &accept_cloned->getEntryBlock();
  for (BasicBlock::iterator it = clonedEntry->begin(); it != clonedEntry->end();
       ++it) {
    if (it->getName() == "plower") {
      plower = &*it;
      break;
    }
  }
  // Get upperv and incr
  Function::arg_iterator argit = accept_cloned->arg_begin();
  std::advance(argit, 3);
  upperv = argit;
  std::advance(argit, 1);
  incr = argit;
}

// Replace counterPtr with:
// If loop decrements: upperv - (plower*incr)
// If loop increments: plower*incr
Value *replaceCounter(Value *plower, Value *upperv, Value *incr,
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

// Iterate through accept_cloned blocks and replace uses of counterPtr
void replaceCounterPtr(Value *counterPtr, Function *accept_cloned,
                       std::vector<BasicBlock *> bodyBlocks, bool isUnsigned,
                       bool willInvert) {
  Value *allocatedNewPointer = NULL;
  std::set<Instruction *> replaceUsesLoad;
  std::set<Instruction *> replaceUsesOther;
  Value *plower, *upperv, *incr;

  IRBuilder<> builder(accept_cloned->getContext());

  getFunctionPointers(accept_cloned, plower, upperv, incr);
  for (std::vector<BasicBlock *>::iterator bit = bodyBlocks.begin();
       bit != bodyBlocks.end(); ++bit) {
    Instruction *firstLoad = NULL;
    Instruction *firstNoLoad = NULL;
    replaceUsesLoad.clear();
    replaceUsesOther.clear();

    // Run through the block in search of uses of counterPtr
    for (BasicBlock::iterator I = (*bit)->begin(); I != (*bit)->end(); ++I) {
      Instruction::op_iterator opit =
          std::find(I->op_begin(), I->op_end(), counterPtr);
      // Skip instructions that don't use counterPtr
      if (opit == I->op_end())
        continue;
      Value *iOper = *opit;
      // Load instructions are special, treat them differently
      if (isa<LoadInst>(I)) {
        if (!firstLoad)
          firstLoad = I;
        replaceUsesLoad.insert(I);
      } else {
        if (!firstNoLoad)
          firstNoLoad = I;
        replaceUsesOther.insert(I);
      }
    }
    // Replace load(counterPtr) with load(plower)
    if (firstLoad) {
      Value *replaceValue;
      builder.SetInsertPoint(firstLoad);
      // Create the arithmetic conversion from plower to the desired value
      replaceValue =
          replaceCounter(plower, upperv, incr, isUnsigned, willInvert, builder);
      // Replace uses of oldLoad with newLoad. Erase oldLoad from the BB
      for (std::set<Instruction *>::iterator it = replaceUsesLoad.begin();
           it != replaceUsesLoad.end(); ++it) {
        Instruction *willRep = *it;
        // Special case (original is of a different type from the comparison)
        // Thus, we replace its extended (or truncated ) value
        if (willRep->getType() != replaceValue->getType()) {
          Value *firstUse = *willRep->use_begin();
          // Check if the load is casted
          if (isa<CastInst>(firstUse)) {
            willRep = cast<Instruction>(firstUse);
          }
          // We must cast replaceValue manually to fit willReps type
          else {
            builder.SetInsertPoint(cast<Instruction>(firstUse));
            IntegerType *toAttain = cast<IntegerType>(willRep->getType());
            if (isUnsigned)
              replaceValue = builder.CreateSExtOrTrunc(replaceValue, toAttain);
            else
              replaceValue = builder.CreateZExtOrTrunc(replaceValue, toAttain);
          }
        }
        willRep->replaceAllUsesWith(replaceValue);
        willRep->eraseFromParent();
      }
    }
    if (firstNoLoad) {
      // For other uses, we must replace the pointer itself. Allocate a new
      // pointer if this was not done yet
      if (!allocatedNewPointer) {
        builder.SetInsertPoint(accept_cloned->getEntryBlock().begin());
        allocatedNewPointer = builder.CreateAlloca(
            counterPtr->getType()->getPointerElementType(), NULL, "replacePtr");
      }
      for (std::set<Instruction *>::iterator it = replaceUsesOther.begin();
           it != replaceUsesOther.end(); ++it) {
        Instruction *willRep = *it;
        builder.SetInsertPoint(willRep);
        Value *replaceValue = replaceCounter(plower, upperv, incr, isUnsigned,
                                             willInvert, builder);
        // Store operations on new pointer
        builder.CreateStore(replaceValue, allocatedNewPointer);
        willRep->replaceUsesOfWith(counterPtr, allocatedNewPointer);
      }
    }
  }
}

bool LoopParallelize::runOnLoop(Loop *L, LPPassManager &LPM) {
  // Don't search skippable functions
  if (transformPass->shouldSkipFunc(*(L->getHeader()->getParent())))
    return false;
  this->L = L;
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
  if (!getLowerAndUpperBounds()) {
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
  incrementInstructions.clear();
  bool foundIncrement = getIncrement();
  Instruction *incInst = *(incrementInstructions.begin() + 1);
  if (!foundIncrement) {
    ACCEPT_LOG << "No valid increment operation (sum or subtraction) found "
                  "inside the loop. Cannot optimize\n";
    return false;
  }

  increment = incInst->getOperand(1);
  // Check if fixed value (Constant)
  if (Instruction *possibleInst = dyn_cast<Instruction>(increment)) {

    if (isOnLoop(cast<Instruction>(increment))) {
      ACCEPT_LOG
          << "the increment is defined inside the loop, cannot parallelize\n";
      return false;
    }
  }

  // Get the body blocks of the loop
  bodyBlocks.clear();
  for (Loop::block_iterator bit = L->block_begin(); bit != L->block_end();
       ++bit) {
    BasicBlock *currentB = *bit;
    if (currentB == loopHeader)
      continue;
    if (isForLike && currentB == L->getLoopLatch())
      continue;
    bodyBlocks.push_back(currentB);
  }

  if (bodyBlocks.empty()) {
    ACCEPT_LOG << "empty body\n";
    return false;
  }

  // Check for control flow in the loop body. We don't perforate anything
  // with a break, continue, return, etc.
  for (std::vector<BasicBlock *>::iterator i = bodyBlocks.begin();
       i != bodyBlocks.end(); ++i) {
    if (L->isLoopExiting(*i)) {
      ACCEPT_LOG << "contains loop exit\n";
      ACCEPT_LOG << "cannot perforate loop\n";
      return false;
    }
  }

  // Check whether the body of this loop is elidable (precise-pure).
  std::set<BasicBlock *> bodyBlocksSet(bodyBlocks.begin(), bodyBlocks.end());
  std::set<Instruction *> blockers = AI->preciseEscapeCheck(bodyBlocksSet);

  // Search all body blocks for pointers created outside the loop. They must
  // be passed as arguments on __kmpc_fork_call
  bodyPointers.clear();
  for (std::vector<BasicBlock *>::iterator it = bodyBlocks.begin();
       it != bodyBlocks.end(); ++it) {
    searchBodyPointers(*it);
  }

  // Parallelize loop
  if (transformPass->relax) {
    int param = transformPass->relaxConfig[loopName];
    if (param) {
      ACCEPT_LOG << "paralellizing with factor 2^" << param << "\n";
      return paralellizeLoop(LPM, param);
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
bool LoopParallelize::paralellizeLoop(LPPassManager &LPM, int logthreads) {
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
      cloneName += "s32";
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

  // If this is a while loop, the increment is in the body. Remove it before
  // inserting the BBs on the cloned function
  if (!isForLike) {
    for (SmallVector<Instruction *, 3>::iterator it =
             incrementInstructions.end() - 1;
         it >= incrementInstructions.begin(); --it) {
      Instruction *toDel = *it;
      toDel->dropAllReferences();
      toDel->eraseFromParent();
    }
  }

  // Clone the omp.untlined function
  ValueToValueMapTy valueM;
  SmallVector<ReturnInst *, 1> returns;
  // Create a function blank in which toClone will be inserted into
  Function *clonedF = createFunction(toClone, bodyPointers, valueM);
  CloneFunctionInto(clonedF, toClone, valueM, false, returns);
  Function::iterator beforeAddition = clonedF->begin();
  // This is the lowerexec BB
  std::advance(beforeAddition, 4);
  // This is the after_lower BB
  Function::iterator afterAddition = beforeAddition;
  afterAddition++;

  // Deal with debug calls and metadata
  std::vector<Instruction *> dbgToDel;
  for (std::vector<BasicBlock *>::iterator bit = bodyBlocks.begin();
       bit != bodyBlocks.end(); ++bit) {
    BasicBlock *b = *bit;
    for (BasicBlock::iterator ii = b->begin(); ii != b->end(); ++ii) {
      // Remove llvm.dbg.* calls to avoid metadata errors on cloned function
      CallInst *callFunc = dyn_cast<CallInst>(ii);
      if (callFunc) {
        if (!callFunc->getCalledFunction() ||
            !callFunc->getCalledFunction()->getName().startswith("llvm.dbg"))
          continue;
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

  // Find the first and last basicblocks of the body

  // beginBody is the basic block that comes after the header
  // Ideally, this is the first BasicBlock when iterating on bodyBlocks, but we
  // must be safe if this is not the case
  BasicBlock *beginBody;
  if (isOnLoop(cast<BasicBlock>(headerTerm->getOperand(1)))) {
    beginBody = cast<BasicBlock>(headerTerm->getOperand(1));
  } else {
    beginBody = cast<BasicBlock>(headerTerm->getOperand(2));
  }

  // Avoid the header branching to the loop body, that will be sent to the
  // cloned function
  BasicBlock *afterBody;
  if (isForLike)
    afterBody = L->getLoopLatch();
  else
    afterBody = L->getExitBlock();

  // Create a new jump instruction if toJump is already a sucessor of the header
  if (std::find(headerTerm->op_begin(), headerTerm->op_end(), afterBody) !=
      headerTerm->op_end()) {
    BranchInst *toExitB = BranchInst::Create(afterBody);
    ReplaceInstWithInst(headerTerm, toExitB);
  } else {
    headerTerm->replaceUsesOfWith(beginBody, afterBody);
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
    if (bb != beginBody) {
      LI->removeBlock(bb);
      bb->moveAfter(lastMoved);
      lastMoved = bb;
    }
  }

  BranchInst *newJump;
  // Adjust the label to jump to from the block before beginBody
  beforeAddition->getTerminator()->replaceUsesOfWith(afterAddition, beginBody);

  // Search all bodyBlocks for jumps not in clonedF
  for (std::vector<BasicBlock *>::iterator bit = bodyBlocks.begin();
       bit != bodyBlocks.end(); ++bit) {
    TerminatorInst *term = (*bit)->getTerminator();
    // Replace possible jumps to original function into equivalents on new
    // function
    term->replaceUsesOfWith(loopHeader, beforeAddition);
    term->replaceUsesOfWith(afterBody, afterAddition);
  }

  IRBuilder<> builder(module->getContext());

  // Replace uses of counterPtr inside clonedF with some modification of plower
  this->counterPtr = cast<LoadInst>(loopHeader->front()).getPointerOperand();
  replaceCounterPtr(counterPtr, clonedF, bodyBlocks, compInst->isUnsigned(),
                    willInvert);

  // For each use of bodyPointers[i] that is now on the acc_cloned function,
  // replace its uses with the corresponding function argument
  Function::arg_iterator argit = clonedF->arg_begin();
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
  Type *incrTy = increment->getType();
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
    TerminatorInst *newBlockTerm = SplitBlockAndInsertIfThen(negCmp, false);
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
    if (isUnsigned) {
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
  deleteLoop(LPM);
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
