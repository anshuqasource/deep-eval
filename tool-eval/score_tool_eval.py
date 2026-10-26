import json, csv
from openrouter_judge import OpenRouterJudge
from deepeval.test_case import LLMTestCase, ToolCall
try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as P
from deepeval.metrics import ToolCorrectnessMetric, GEval

judge = OpenRouterJudge()

outcome_correctness = GEval(
    name="OutcomeCorrectness",
    criteria=(
        "Given the expected outcome description, determine whether the "
        "actual output correctly reflects what should have happened - "
        "e.g. correctly confirming or denying stock/purchase availability, "
        "or appropriately handling the complaint. The actual output does "
        "not need to match word-for-word, only be factually and "
        "functionally correct given the expected outcome."
    ),
    evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
    model=judge,
    threshold=0.5,
)

with open("golden_tool_dataset.json", encoding="utf-8") as f:
    cases = json.load(f)
    
results = []
for c in cases:
    print(f"\n{'='*60}\n[{c['case_id']}] {c['user_input']}")

    tool_metric = ToolCorrectnessMetric(model=judge)
    tool_test_case = LLMTestCase(
        input=c["user_input"],
        actual_output=c["actual_output"],
        tools_called=[ToolCall(name=c["actual_tool_called"])],
        expected_tools=[ToolCall(name=c["expected_tool"])],
    )
    tool_metric.measure(tool_test_case)
    print(f"  ToolCorrectness: {tool_metric.score:.3f} - {tool_metric.reason}")

    outcome_test_case = LLMTestCase(
        input=c["user_input"],
        actual_output=c["actual_output"],
        expected_output=c["expected_outcome"],
    )
    try:
        outcome_correctness.measure(outcome_test_case)
        outcome_score = outcome_correctness.score
        outcome_reason = outcome_correctness.reason
        print(f"  OutcomeCorrectness: {outcome_score:.3f} - {outcome_reason}")
    except Exception as e:
        outcome_score = None
        outcome_reason = str(e)
        print(f"  OutcomeCorrectness FAILED: {e}")

    results.append({
        "case_id": c["case_id"],
        "user_input": c["user_input"],
        "expected_tool": c["expected_tool"],
        "actual_tool_called": c["actual_tool_called"],
        "tool_correctness": tool_metric.score,
        "outcome_correctness": outcome_score,
        "outcome_reason": outcome_reason,
    })

with open("tool_eval_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n\nSaved to tool_eval_report.csv")
print("\n=== AVERAGES ===")
tc_scores = [r["tool_correctness"] for r in results]
oc_scores = [r["outcome_correctness"] for r in results if r["outcome_correctness"] is not None]
print(f"Tool Correctness avg: {sum(tc_scores)/len(tc_scores):.3f}")
if oc_scores:
    print(f"Outcome Correctness avg: {sum(oc_scores)/len(oc_scores):.3f}")