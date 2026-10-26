from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams as ParamsEnum
    print("Using SingleTurnParams (DeepEval 4.x)")
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as ParamsEnum
    print("Using LLMTestCaseParams (DeepEval 3.x)")

from deepeval.metrics import GEval
print("Core imports OK")
print("Params enum members:", list(ParamsEnum))