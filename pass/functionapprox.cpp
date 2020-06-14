#include "llvm/Module.h"
#include "llvm/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <sstream>

#include "accept.h"

using namespace llvm;

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
// tus, remove the final f
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
  struct FunctionApprox : public FunctionPass {
    static char ID;
    ACCEPTPass *transformPass;
    ApproxInfo *AI;
    Module *module;

    FunctionApprox() : FunctionPass(ID) {
      initializeFunctionApproxPass(*PassRegistry::getPassRegistry());
      module = 0;
    }

    virtual bool doInitialization(Module &M) {
      module = &M;
      transformPass = (ACCEPTPass *)sharedAcceptTransformPass;
      AI = transformPass->AI;
      return false;
    }

    const char *getPassName() const {
      return "ACCEPT Function Approximation";
    }

    virtual bool runOnFunction(Function &F) {
      AI = &getAnalysis<ApproxInfo>();
      if (transformPass->shouldSkipFunc(F))
        return false;
      std::vector<CallInst *>toReplace;
      bool modified = false;
      // Search each function for instruction calls
      for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
          if (CallInst *CI = dyn_cast<CallInst>(I)) {
            StringRef functionName = CI->getCalledFunction()->getName();
            if (functionReplacementList.count(functionName)) {
              toReplace.push_back(CI);
            }
          }
        }
      }


      for(int i = 0 ; i < toReplace.size() ; ++i) {
        modified |= tryToOptimizeCall(toReplace[i]);
      }
      return modified;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<ApproxInfo>();
    }


    virtual bool doFinalization(Module &M) {
      return false;
    }

    bool tryToOptimizeCall(CallInst *Call) {
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
      
      // Since these mathematical functions are precise-pure by nature,
      // all we need is to check if they store to an approx variable
      // if so, we can approx it
      Instruction *storeCall = cast<Instruction>(*Call->use_begin());
      if(!isApprox(storeCall)) {
        return false;
      }

      if (transformPass->relax) {
        int param = transformPass->relaxConfig[callName];
        if (param) {
          ACCEPT_LOG << "replacing fuction with version " 
                     << approxVersion(param)
                     << '\n';
          replaceFunction(Call,param);
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

    void replaceFunction(CallInst *CI, unsigned int option) {
      Function *newFunction, *currentFunction;
      // This pointer is used in case we need two functions
      //(See pow2() and sin,cos,tan)
      Function *otherNewFunction;
      // Function Type of the newfunction
      FunctionType *fcType;
      // Needed when using two functions
      FunctionType *otherFcType;
      CallInst *newCall;
      // Needed when using two functions
      CallInst *otherNewCall;
      // Value of the comparison used for creating branch instruction
      Value *comparison;
      std::string opString, functionName;
      // Needed when using two functions
      std::string otherOpString;
      // Global context for creation of types
      currentFunction = CI->getCalledFunction();
      Module *parent = currentFunction->getParent();
      LLVMContext &ctx = parent->getContext();
      functionName = formatString(currentFunction->getName());
      bool createCheck = false;
      // Default string for new prototype
      opString = approxVersion(option);
      // the exp2() function is called pow2() in fastapprox
      if (functionName == "exp2")
        opString += "pow2";
      else
        opString += functionName;

      // Start the IR builder before CI
      IRBuilder<> builder(CI);
      // Create prototype Types
      std::vector<Type *> tp;
      // some functions require 2 args, some only one
      // whenever we need only one, just slice the Array
      for (int i = 0; i < currentFunction->getFunctionType()->getNumParams(); i++)
        tp.push_back(Type::getFloatTy(ctx));
      ArrayRef<Type *> typeArray(tp);
      Type *returnType = Type::getFloatTy(ctx);
      // Create arguments array
      std::vector<Value *> argumentsVector;
      for (int i = 0; i < CI->getNumArgOperands(); i++) {
        // The arguments must be converted to float (in case they came from a
        // double standard)
        Value *oldArgument, *newArgument;
        oldArgument = CI->getArgOperand(i);
        // If they are single precision type, we are ok
        if (oldArgument->getType()->getTypeID() == Type::FloatTyID) {
          newArgument = oldArgument;
        }
        // else, use the IRBuilder to create a truncate instruction and use it
        // as
        // the argument
        else {
          newArgument = builder.CreateFPTrunc(oldArgument, typeArray[i]);
        }
        argumentsVector.push_back(newArgument);
      }
      ArrayRef<Value *> argsArray(argumentsVector);
      // Needed when using two functions
      ArrayRef<Value *> otherArgsArray;
      switch (resolveOptions(functionName)) {
      case POW: {
        ConstantFP *arg0 = dyn_cast<ConstantFP>(CI->getArgOperand(0));
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
          comparison = builder.CreateFCmpOEQ(argsArray[0], valueTwo);

          otherOpString = opString;
          opString += '2';
          fcType = FunctionType::get(returnType, typeArray.slice(1), false);
          otherFcType = FunctionType::get(returnType, typeArray, false);
          otherArgsArray = argsArray;
          argsArray = argsArray.slice(1);
          createCheck = true;
        }
        break;
      }
      // for sin, cos and tan, the fast/faster versions
      // should be used only on specific angle intervals
      case SINorCOS: { //[-\pi,pi]
        ConstantFP *arg0 = dyn_cast<ConstantFP>(CI->getOperand(0));
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
          comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

          otherOpString = opString + "full";
          fcType = FunctionType::get(returnType, typeArray, false);
          otherFcType = FunctionType::get(returnType, typeArray, false);
          otherArgsArray = argsArray;
          createCheck = true;
        }
        break;
      }
      case TAN: { //[-\pi / 2, pi / 2]
        ConstantFP *arg0 = dyn_cast<ConstantFP>(CI->getOperand(0));
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
          comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

          otherOpString = opString + "full";
          fcType = FunctionType::get(returnType, typeArray, false);
          otherFcType = FunctionType::get(returnType, typeArray, false);
          otherArgsArray = argsArray;
          createCheck = true;
        }
        break;
      }
      case OTHER: {
        fcType = FunctionType::get(returnType, typeArray, false);
        break;
      }
      }
      SmallVector<std::pair<unsigned, MDNode *>, 0> MD;
      CI->getAllMetadata(MD);
      // check that denotes we are creating two functions
      if (createCheck) {
        BasicBlock::iterator bit = ++BasicBlock::iterator(CI);
        // Create a basic block for after adding the two calls after CI
        BasicBlock *afterIfElse =
            CI->getParent()->splitBasicBlock(bit, "afterIfElse");
        // Create basic blocks required for ifelse comparison
        BasicBlock *ifBB = BasicBlock::Create(
            ctx, "ifComp.then", afterIfElse->getParent(), afterIfElse);
        BasicBlock *elseBB = BasicBlock::Create(
            ctx, "ifComp.else", afterIfElse->getParent(), afterIfElse);
        // Create the branch function and replace
        // the one created automatically by splitBasicBlock()
        BranchInst *condJump = BranchInst::Create(ifBB, elseBB, comparison);
        Instruction *toReplace = CI->getParent()->getTerminator();
        ReplaceInstWithInst(toReplace, condJump);
        // Create both functions
        const AttrListPtr atrPtr = currentFunction->getAttributes();
        newFunction =
            cast<Function>(parent->getOrInsertFunction(opString, fcType, atrPtr));
        otherNewFunction = cast<Function>(
            parent->getOrInsertFunction(otherOpString, otherFcType, atrPtr));
        // Insert the call and jump into
        // both basic blocks
        builder.SetInsertPoint(ifBB);
        newCall = builder.CreateCall(newFunction, argsArray);
        builder.CreateBr(afterIfElse);
        builder.SetInsertPoint(elseBB);
        otherNewCall = builder.CreateCall(otherNewFunction, otherArgsArray);
        builder.CreateBr(afterIfElse);
        // Set calling metadata IF any
        for (SmallVector<std::pair<unsigned, MDNode *>, 0>::iterator mdit =
                 MD.begin();
             mdit != MD.end(); ++mdit) {
          newCall->setMetadata(mdit->first, mdit->second);
          otherNewCall->setMetadata(mdit->first, mdit->second);
        }
        // Set calling conventions to mirror CI
        newCall->setCallingConv(CI->getCallingConv());
        otherNewCall->setCallingConv(CI->getCallingConv());
        // Allocate a new pointer to receive result from both newCalls
        BasicBlock::iterator pointToInsert = CI->getParent()->getFirstInsertionPt();
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
        // If CI is a double, convert result from float to double
        if (CI->getType()->getTypeID() == Type::DoubleTyID) {
          Value *ext = builder.CreateFPExt(result, Type::getDoubleTy(ctx));
          CI->replaceAllUsesWith(ext);
        }
        // else, just replace uses of CI with result
        else
          CI->replaceAllUsesWith(result);
      } else {
        // Add the function prototype to the module
        const AttrListPtr atrPtr = currentFunction->getAttributes();
        newFunction =
            cast<Function>(parent->getOrInsertFunction(opString, fcType, atrPtr));
        // Create the instruction call and place it before CI
        newCall = builder.CreateCall(newFunction, argsArray);
        // ensure conventions and attributes are equal
        newCall->setCallingConv(CI->getCallingConv());
        newCall->setAttributes(CI->getAttributes());
        // ensure metadata are equal (if there are any)
        for (SmallVector<std::pair<unsigned, MDNode *>, 0>::iterator mdit =
                 MD.begin();
             mdit != MD.end(); ++mdit)
          newCall->setMetadata(mdit->first, mdit->second);
        // Replace the old instruction with the new call, including its uses if
        // CI is doubleType, create FPExtention instruction to receive the result
        // of newCall
        if (CI->getType()->getTypeID() == Type::DoubleTyID) {
          Value *ext = builder.CreateFPExt(newCall, Type::getDoubleTy(ctx));
          CI->replaceAllUsesWith(ext); // pass the extension to the uses of CI
        } else
          // else, just replace all uses with the new call
          CI->replaceAllUsesWith(newCall);
      }
      // remove CI from the basic block
      CI->eraseFromParent();
    }
  };
}

char FunctionApprox::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionApprox,"fapprox", "ACCEPT function approximation pass",false,false)
INITIALIZE_PASS_DEPENDENCY(ApproxInfo)
INITIALIZE_PASS_END(FunctionApprox,"fapprox", "ACCEPT function approximation pass",false,false)
FunctionPass *llvm::createFunctionApproxPass() {
  return new FunctionApprox();
}
