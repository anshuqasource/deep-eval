import json, csv
from openrouter_judge import OpenRouterJudge
from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as P
from deepeval.metrics import GEval

judge = OpenRouterJudge()

policy_compliance = GEval(
    name="PolicyCompliance",
    criteria=(
        "Determine whether the actual output avoids promising a refund, "
        "replacement, discount, or specific resolution timeline that is not "
        "explicitly and unconditionally supported by the retrieval context. "
        "A response that appropriately defers to review, verification, or a "
        "support ticket before granting a refund/exchange is compliant."
    ),
    evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
    model=judge,
    threshold=0.5,
)

ticket_offer = GEval(
    name="TicketOfferAppropriate",
    criteria=(
        "Determine whether the assistant handled the possibility of a support "
        "ticket correctly: offering one when it could not fully resolve the "
        "issue, correctly not offering one when fully resolved, or failing to "
        "offer one when it clearly could not resolve the issue (the only "
        "incorrect case)."
    ),
    evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.5,
)

with open("samples.json", encoding="utf-8") as f:
    samples = json.load(f)

results = []
for i, s in enumerate(samples):
    print(f"\n[{i+1}/10] {s['user_input'][:55]}...")
    test_case = LLMTestCase(
        input=s["user_input"],
        actual_output=s["response"],
        retrieval_context=s["retrieved_contexts"],
    )

    row = {"user_input": s["user_input"]}
    try:
        policy_compliance.measure(test_case)
        row["policy_compliance"] = policy_compliance.score
        print(f"  PolicyCompliance: {policy_compliance.score:.3f}")
    except Exception as e:
        row["policy_compliance"] = None
        print(f"  PolicyCompliance FAILED: {e}")

    try:
        ticket_offer.measure(test_case)
        row["ticket_offer_appropriate"] = ticket_offer.score
        row["ticket_offer_reason"] = ticket_offer.reason
        print(f"  TicketOfferAppropriate: {ticket_offer.score:.3f}")
    except Exception as e:
        row["ticket_offer_appropriate"] = None
        print(f"  TicketOfferAppropriate FAILED: {e}")

    results.append(row)

with open("deepeval_geval_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_input", "policy_compliance", "ticket_offer_appropriate", "ticket_offer_reason"])
    writer.writeheader()
    writer.writerows(results)

print("\nSaved to deepeval_geval_report.csv")