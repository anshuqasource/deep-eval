from openrouter_judge import OpenRouterJudge
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval

try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as P

judge = OpenRouterJudge()

correctness = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct based on the expected output.",
    evaluation_params=[P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
    model=judge,
    threshold=0.5,
)

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="The capital of France is Paris",
)

correctness.measure(test_case)
print(f"Score: {correctness.score}")
print(f"Reason: {correctness.reason}")
print(f"success: {correctness.success}")