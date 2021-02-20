#include "llvm/Module.h"
#include "llvm/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"

#include <sstream>

#include "accept.h"

using namespace llvm;

bool acceptAllFAP;

char const* functionarray [] = {
  "exp", 
  "exp2",
  "pow", 
  "sin", 
  "cos", 
  "tan", 
  "log", 
  "log2", 
  "erfc", 
  "erf", 
  "lgamma", 
  "digamma", 
  "sinh",
  "cosh",
  "tanh"
};

const std::set<std::string> functionReplacementList(
    functionarray,
    functionarray+
    sizeof(functionarray)/
    sizeof(functionarray[0])
);

std::string approxVersion(int param) {
  std::string opString;
  switch (param) {
    case 1: {
      opString = "fast";
      break;
    }
    case 2: {
      opString = "faster";
      break;
    }
  }
  return opString;
}

// We want to replace both float and double implementations
// thus, remove the final f
StringRef formatString(StringRef functionName) {
  if (functionName.endswith("f") && functionName != "erf")
    return functionName.drop_back();
  return functionName;
}

enum functionOptions {
  POW,
  SINorCOS,
  TAN,
  OTHER
};

functionOptions resolveOptions(StringRef name) {
  name = formatString(name);
  if (name == "pow")
    return POW;
  if (name == "sin" || name == "cos")
    return SINorCOS;
  if (name == "tan")
    return TAN;
  return OTHER;
}

namespace {
  cl::opt<bool, true> optProf("all-fap",
      cl::desc("ACCEPT: test all function approximation options"),
      cl::location(acceptAllFAP));

  struct FunctionApprox : public FunctionPass {
    static char ID;
    ACCEPTPass *transformPass;
    ApproxInfo *AI;
    Module *module;

    FunctionApprox();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual const char *getPassName() const;
    virtual bool doInitialization(llvm::Module &M);
    virtual bool doFinalization(llvm::Module &M);
    virtual bool runOnFunction(llvm::Function &F);

    bool tryToOptimizeCall(CallInst *Call);
    void copyCallMetadata(CallInst *C1, CallInst *C2);
    std::vector<Value *> createArgumentsArray(CallInst *Call,
                                              IRBuilder<> &builder);
    bool populateReplacementRequirements(
        CallInst *&Call, FunctionType *&fcType, FunctionType *&otherFCType,
        ArrayRef<Value *> &argsArray, ArrayRef<Value *> &otherArgsArray,
        std::string &opString, std::string &otherOpString, Value *&Comparison,
        IRBuilder<> &builder);
    void replaceFunction(CallInst *Call, unsigned int option);
  };
}

FunctionApprox::FunctionApprox() : FunctionPass(ID) {
  module = 0;
  AI = NULL;
}

bool FunctionApprox::doInitialization(Module &M) {
  module = &M;
  transformPass = (ACCEPTPass *)sharedAcceptTransformPass;
  return false;
}

const char *FunctionApprox::getPassName() const {
  return "ACCEPT Function Approximation";
}

bool FunctionApprox::runOnFunction(Function &F) {
  if (!AI)
    AI = transformPass->AI;
  if (transformPass->shouldSkipFunc(F))
    return false;
  std::vector<CallInst *> toReplace;
  bool modified = false;
  // Search each function for instruction calls
  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        Function *calledF = CI->getCalledFunction();
        if(!calledF)
          continue;
        StringRef functionName = formatString(calledF->getName());
        if (functionReplacementList.count(functionName)) {
          toReplace.push_back(CI);
        }
      }
    }
  }

  for (int i = 0; i < toReplace.size(); ++i) {
    modified |= tryToOptimizeCall(toReplace[i]);
  }
  return modified;
}

void FunctionApprox::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
}

bool FunctionApprox::doFinalization(Module &M) { return false; }

bool FunctionApprox::tryToOptimizeCall(CallInst *Call) {
  StringRef calledFunctionName = Call->getCalledFunction()->getName();
  std::stringstream ss;
  ss << "call to " << calledFunctionName.str() << " at "
     << srcPosDesc(*module, Call->getDebugLoc());
  std::string callName = ss.str();
  LogDescription *desc = AI->logAdd("Call", Call);
  ACCEPT_LOG << callName << "\n";
  Function *func = Call->getParent()->getParent();
  std::string funcName = func->getName().str();

  ACCEPT_LOG << "within function " << funcName << "\n";

  // Look for ACCEPT_FORBID marker.
  if (AI->instMarker(Call) == markerForbid)
    return false;
  Function *calledFunc = module->getFunction(calledFunctionName);

  // Check for function purity
  if (!AI->isPrecisePure(calledFunc)) {
    return false;
  }
  // Ensure the function call is approximate
  if (!isApprox(Call) && !acceptAllFAP) {
    ACCEPT_LOG << "cannot replace function call\n";
    for (int i = 0; i < Call->getNumArgOperands(); ++i) {
      Instruction *arg = dyn_cast<Instruction>(Call->getArgOperand(i));
      // Register argument blockers
      if (arg && !isApprox(arg)) {
        ACCEPT_LOG << "Arg " << i << " not approx\n";
      }
    }
    return false;
  }

  if (transformPass->relax) {
    int param = transformPass->relaxConfig[callName];
    if (param) {
      ACCEPT_LOG << "replacing fuction with version " << approxVersion(param)
                 << '\n';
      replaceFunction(Call, param);
      return true;
    } else {
      ACCEPT_LOG << "not replacing function\n";
      return false;
    }
  }

  ACCEPT_LOG << "can replace function\n";
  transformPass->relaxConfig[callName] = 0;

  return false;
}

void FunctionApprox::copyCallMetadata(CallInst *C1, CallInst *C2) {
  SmallVector<std::pair<unsigned, MDNode *>, 0> MD;
  C1->getAllMetadata(MD);
  for (SmallVector<std::pair<unsigned, MDNode *>, 0>::iterator mdit =
           MD.begin();
       mdit != MD.end(); ++mdit) {
    C2->setMetadata(mdit->first, mdit->second);
  }
  C2->setCallingConv(C1->getCallingConv());
  C2->setAttributes(C1->getAttributes());
}

std::vector<Value *>
FunctionApprox::createArgumentsArray(CallInst *Call, IRBuilder<> &builder) {
  builder.SetInsertPoint(Call);
  // Create arguments array
  std::vector<Value *> argumentsVector;
  for (int i = 0; i < Call->getNumArgOperands(); i++) {
    // The arguments must be converted to float (in case they came from a
    // double standard)
    Value *oldArgument, *newArgument;
    oldArgument = Call->getArgOperand(i);
    // If they are single precision type, we are ok
    if (oldArgument->getType()->getTypeID() == Type::FloatTyID) {
      newArgument = oldArgument;
    }
    // else, use the IRBuilder to create a truncate instruction and use it
    // as
    // the argument
    else {
      newArgument = builder.CreateFPTrunc(
          oldArgument, Type::getFloatTy(module->getContext()));
    }
    argumentsVector.push_back(newArgument);
  }

  return argumentsVector;
}

bool FunctionApprox::populateReplacementRequirements(
    CallInst *&Call, FunctionType *&fcType, FunctionType *&otherFCType,
    ArrayRef<Value *> &argsArray, ArrayRef<Value *> &otherArgsArray,
    std::string &opString, std::string &otherOpString, Value *&Comparison,
    IRBuilder<> &builder) {

  LLVMContext &ctx = module->getContext();
  Function *currentFunction = Call->getCalledFunction();
  // Create prototype Types
  std::vector<Type *> tp;
  // some functions require 2 args, some only one
  // whenever we need only one, just slice the Array
  for (int i = 0; i < currentFunction->getFunctionType()->getNumParams(); i++)
    tp.push_back(Type::getFloatTy(ctx));
  ArrayRef<Type *> typeArray(tp);
  Type *returnType = Type::getFloatTy(ctx);


  bool requiresTwoCalls = false;
  StringRef functionName = formatString(currentFunction->getName());
  // the exp2() function is called pow2() in fastapprox
  if (functionName == "exp2")
    opString += "pow2";
  else
    opString += functionName;

  builder.SetInsertPoint(Call);
  switch (resolveOptions(functionName)) {
    case POW: {
      ConstantFP *arg0 = dyn_cast<ConstantFP>(Call->getArgOperand(0));
      // use pow2 if first argument is a constant of value exactly 2
      if (arg0) {
        if (arg0->isExactlyValue(2.0)) {
          opString += '2';
          // Construct the FunctionType
          fcType = FunctionType::get(returnType, typeArray.slice(1), false);
          // Construct the argument array (slicing the first argument since
          // we wont use it)
          argsArray = argsArray.slice(1);
        }
        // use pow (or pow2 in case it came from exp2)
        else
          fcType = FunctionType::get(returnType, typeArray, false);
      } else {
        // Compare the first argument with two and branch
        Constant *valueTwo = ConstantFP::get(Type::getFloatTy(ctx), 2.0);
        Comparison = builder.CreateFCmpOEQ(argsArray[0], valueTwo);

        otherOpString = opString;
        opString += '2';
        fcType = FunctionType::get(returnType, typeArray.slice(1), false);
        otherFCType = FunctionType::get(returnType, typeArray, false);
        otherArgsArray = argsArray;
        argsArray = argsArray.slice(1);
        requiresTwoCalls = true;
      }
      break;
    }
    // for sin, cos and tan, the fast/faster versions
    // should be used only on specific angle intervals
    case SINorCOS: { //[-\pi,pi]
      ConstantFP *arg0 = dyn_cast<ConstantFP>(Call->getOperand(0));
      // in case arg is not a constant, or the angle is not within the
      // intervall use the full version
      if (arg0) {
        if (fabs(arg0->getValueAPF().convertToFloat()) > M_PI)
          opString += "full";
        fcType = FunctionType::get(returnType, typeArray, false);
      } else {
        Constant *valuePI = ConstantFP::get(Type::getFloatTy(ctx), M_PI);
        Constant *valueMinusPI = ConstantFP::get(Type::getFloatTy(ctx), -M_PI);
        Value *firstComparison = builder.CreateFCmpOLE(argsArray[0], valuePI);
        Value *secondComparison =
            builder.CreateFCmpOGE(argsArray[0], valueMinusPI);
        firstComparison->setName("ltPI");
        secondComparison->setName("gtMinusPI");
        Comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

        otherOpString = opString + "full";
        fcType = FunctionType::get(returnType, typeArray, false);
        otherFCType = FunctionType::get(returnType, typeArray, false);
        otherArgsArray = argsArray;
        requiresTwoCalls = true;
      }
      break;
    }
    case TAN: { //[-\pi / 2, pi / 2]
      ConstantFP *arg0 = dyn_cast<ConstantFP>(Call->getOperand(0));
      if (arg0) {
        if (2 * fabs(arg0->getValueAPF().convertToFloat()) > M_PI)
          opString += "full";
        fcType = FunctionType::get(returnType, typeArray, false);
      } else {
        Constant *valuePI = ConstantFP::get(Type::getFloatTy(ctx), M_PI / 2.0);
        Constant *valueMinusPI =
            ConstantFP::get(Type::getFloatTy(ctx), -M_PI / 2.0);
        Value *firstComparison = builder.CreateFCmpOLE(argsArray[0], valuePI);
        Value *secondComparison =
            builder.CreateFCmpOGE(argsArray[0], valueMinusPI);
        firstComparison->setName("lePI/2");
        firstComparison->setName("geMinusPI/2");
        Comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

        otherOpString = opString + "full";
        fcType = FunctionType::get(returnType, typeArray, false);
        otherFCType = FunctionType::get(returnType, typeArray, false);
        otherArgsArray = argsArray;
        requiresTwoCalls = true;
      }
      break;
    }
    case OTHER: {
      fcType = FunctionType::get(returnType, typeArray, false);
      break;
    }
  }
  return requiresTwoCalls;
}

void FunctionApprox::replaceFunction(CallInst *Call, unsigned int option) {
  Function *newFunction, *currentFunction;
  FunctionType *fcType;
  CallInst *newCall;
  std::string opString, functionName;
  // Needed when using two functions
  Function *otherNewFunction;
  FunctionType *otherFcType;
  CallInst *otherNewCall;
  std::string otherOpString;
  // Value of the comparison used for creating branch instruction
  Value *comparison;

  // Global context for creation of types
  currentFunction = Call->getCalledFunction();
  Module *parent = currentFunction->getParent();
  LLVMContext &ctx = parent->getContext();
  functionName = formatString(currentFunction->getName());

  // Default string for new prototype
  opString = approxVersion(option);

  // Start the IR builder
  IRBuilder<> builder(ctx);

  std::vector<Value *> argumentsVector = createArgumentsArray(Call, builder);
  ArrayRef<Value *> argsArray(argumentsVector);
  // Needed when using two functions
  ArrayRef<Value *> otherArgsArray;

  // Populates the required variables to ensure the creation of
  // new calls
  bool requiresTwoCalls = populateReplacementRequirements(
      Call, fcType, otherFcType, argsArray, otherArgsArray, opString,
      otherOpString, comparison, builder);

  // check that denotes we are creating two functions
  if (requiresTwoCalls) {
    BasicBlock::iterator bit = ++BasicBlock::iterator(Call);
    // Create a basic block for after adding the two calls after Call
    BasicBlock *afterIfElse =
        Call->getParent()->splitBasicBlock(bit, "afterIfElse");

    // Create basic blocks required for ifelse comparison
    BasicBlock *ifBB = BasicBlock::Create(
        ctx, "ifComp.then", afterIfElse->getParent(), afterIfElse);
    BasicBlock *elseBB = BasicBlock::Create(
        ctx, "ifComp.else", afterIfElse->getParent(), afterIfElse);
    // Create the branch function and replace
    // the one created automatically by splitBasicBlock()
    BranchInst *condJump = BranchInst::Create(ifBB, elseBB, comparison);
    Instruction *toReplace = Call->getParent()->getTerminator();
    ReplaceInstWithInst(toReplace, condJump);
    // Create both functions
    newFunction = cast<Function>(parent->getOrInsertFunction(opString, fcType));
    otherNewFunction =
        cast<Function>(parent->getOrInsertFunction(otherOpString, otherFcType));

    // Insert the call and jump into
    // both basic blocks
    builder.SetInsertPoint(ifBB);
    newCall = builder.CreateCall(newFunction, argsArray);
    builder.CreateBr(afterIfElse);
    builder.SetInsertPoint(elseBB);
    otherNewCall = builder.CreateCall(otherNewFunction, otherArgsArray);
    builder.CreateBr(afterIfElse);

    // Set calling metadata IF any
    copyCallMetadata(Call, newCall);
    copyCallMetadata(Call, otherNewCall);

    // Allocate a new pointer to receive result from both newCalls
    BasicBlock::iterator pointToInsert =
        Call->getParent()->getFirstInsertionPt();
    builder.SetInsertPoint(pointToInsert);
    Value *resultPtr = builder.CreateAlloca(Type::getFloatTy(ctx));
    resultPtr->setName("resultFromApproxCall");

    // Store the result from calls into resultPtr
    pointToInsert = ++BasicBlock::iterator(newCall);
    builder.SetInsertPoint(pointToInsert);
    builder.CreateStore(newCall, resultPtr);
    pointToInsert = ++BasicBlock::iterator(otherNewCall);
    builder.SetInsertPoint(pointToInsert);
    builder.CreateStore(otherNewCall, resultPtr);

    // Load pointer into variable result
    builder.SetInsertPoint(afterIfElse->getFirstInsertionPt());
    Value *result = builder.CreateLoad(resultPtr, "result");
    // If Call is a double, convert result from float to double
    if (Call->getType()->getTypeID() == Type::DoubleTyID) {
      Value *ext = builder.CreateFPExt(result, Type::getDoubleTy(ctx));
      Call->replaceAllUsesWith(ext);
    }
    // else, just replace uses of Call with result
    else {
      Call->replaceAllUsesWith(result);
    }
  } else {
    // Add the function prototype to the module
    newFunction = cast<Function>(parent->getOrInsertFunction(opString, fcType));
    // Create the instruction call and place it before Call
    newCall = builder.CreateCall(newFunction, argsArray);
    // ensure conventions, attributes and metadata are equal
    copyCallMetadata(Call, newCall);
    // Replace the old instruction with the new call, including its uses. If
    // Call is doubleType, create FPExtention instruction to receive the result
    // of newCall
    if (Call->getType()->getTypeID() == Type::DoubleTyID) {
      Value *ext = builder.CreateFPExt(newCall, Type::getDoubleTy(ctx));
      Call->replaceAllUsesWith(ext);
    } else {
      Call->replaceAllUsesWith(newCall);
    }
  }
  // remove Call from the basic block
  Call->eraseFromParent();
}

char FunctionApprox::ID = 0;
FunctionPass *llvm::createFunctionApproxPass() {
  return new FunctionApprox();
}
