import json, csv
from openrouter_judge import OpenRouterJudge

from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as P

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

judge = OpenRouterJudge()

metrics = [
    FaithfulnessMetric(model=judge, threshold=0.5),
    AnswerRelevancyMetric(model=judge, threshold=0.5),
    ContextualPrecisionMetric(model=judge, threshold=0.5),
    ContextualRecallMetric(model=judge, threshold=0.5),
    ContextualRelevancyMetric(model=judge, threshold=0.5),
]

with open("samples.json", encoding="utf-8") as f:
    samples = json.load(f)
    #samples = samples[:1]   # TEMP - test with 1 question first

results = []
for i, s in enumerate(samples):
    print(f"\n{'='*60}\n[{i+1}/{len(samples)}] {s['user_input']}")

    test_case = LLMTestCase(
        input=s["user_input"],
        actual_output=s["response"],
        expected_output=s["reference"],
        retrieval_context=s["retrieved_contexts"],
    )

    row = {"user_input": s["user_input"]}
    for metric in metrics:
        try:
            metric.measure(test_case)
            row[metric.__class__.__name__] = metric.score
            print(f"  {metric.__class__.__name__}: {metric.score:.3f}")
        except Exception as e:
            row[metric.__class__.__name__] = None
            print(f"  {metric.__class__.__name__}: FAILED - {e}")

    results.append(row)

# Save to CSV
fieldnames = ["user_input"] + [m.__class__.__name__ for m in metrics]
with open("deepeval_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n\nSaved to deepeval_report.csv")

# Print averages
print("\n=== AVERAGES ===")
for m in metrics:
    name = m.__class__.__name__
    scores = [r[name] for r in results if r[name] is not None]
    if scores:
        print(f"{name}: {sum(scores)/len(scores):.3f}")