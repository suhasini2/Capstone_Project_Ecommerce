"""All evaluation queries live here. Calibration queries are kept separate from the demo/eval queries
so the threshold is not tuned on the very queries used to demonstrate it."""
CALIBRATION_IN = [
    "What is the return period for shoes?",
    "Can I cancel after the order has shipped?",
    "Can I redeem reward points on a small order?",
    "How many days does delivery take?",
]
CALIBRATION_OUT = [
    "Who won the cricket world cup?",
    "Give me a recipe for butter chicken",
    "How do I reset my Netflix password?",
]
# Tasks 4 + 5: same queries; relevant = parent documents that truly answer the question.
RETRIEVAL_QUERIES = [
    ("How many days do I have to return a lipstick I bought?", {"KB-01"}),
    ("When will my COD refund reach my bank account?", {"KB-02"}),
    ("How long does standard delivery take to a metro city?", {"KB-03"}),
    ("Is a product eligible for reverse pickup if I opened it?", {"KB-04"}),
    ("What warranty comes with a laptop?", {"KB-05"}),
    ("My parcel arrived broken, how do I claim?", {"KB-10"}),
]
DEMO_OUT_OF_SCOPE = "What is the capital of Australia?"

# Task 13: 12 topic queries (one per required KB topic) + 2 out-of-scope + 1 edge case = 15.
# (query, expected_doc_or_None, expect_refusal)
TRIAD_QUERIES = [
    ("What is the return window for beauty products?", "KB-01", False),
    ("How long does a COD refund take?", "KB-02", False),
    ("What are the delivery timelines for non-metro cities?", "KB-03", False),
    ("Which items qualify for reverse pickup?", "KB-04", False),
    ("What warranty do electronics have?", "KB-05", False),
    ("Can I cancel my order after it is shipped?", "KB-06", False),
    ("How do I redeem my reward points?", "KB-07", False),
    ("What happens if money is deducted but my payment failed?", "KB-08", False),
    ("How does a size exchange work for shoes?", "KB-09", False),
    ("How do I claim a damaged item?", "KB-10", False),
    ("Can I ship perfume internationally?", "KB-11", False),
    ("How are unresolved issues escalated to a manager?", "KB-12", False),
    ("What is the capital of France?", None, True),          # out of scope
    ("Write me a poem about the monsoon.", None, True),      # out of scope
    ("Can I return an opened lipstick?", "KB-04", False),    # edge case: hygiene exclusion
]
