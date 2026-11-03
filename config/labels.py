"""
config/labels.py
~~~~~~~~~~~~~~~~
Canonical label definitions for every classification / analysis task in the
Customer Complaint Root-Cause Classifier project.

Sorted lists are used so that label-to-index mappings are deterministic
across runs and machines.
"""

from typing import List

# ---------------------------------------------------------------------------
# 1.  Issue categories  (21 unique)
# ---------------------------------------------------------------------------
ISSUE_LABELS: List[str] = sorted([
    "Fees",
    "Payments",
    "Account management",
    "Credit reporting",
    "Billing dispute",
    "Account access",
    "Fraud or scam",
    "Collection practices",
    "Deposits and withdrawals",
    "Communication",
    "Interest rate",
    "Servicing",
    "Payment arrangement",
    "Application",
    "Foreclosure",
    "Transfer pending",
    "Transfer error",
    "Repossession",
    "Repayment",
    "Forbearance",
    "Card activation",
])

# ---------------------------------------------------------------------------
# 2.  Sub-issue categories  (38 unique)
# ---------------------------------------------------------------------------
SUB_ISSUE_LABELS: List[str] = sorted([
    "Cash withdrawal",
    "Unauthorized transaction",
    "Payment processing",
    "Account closed",
    "Rate increase",
    "Payment not credited",
    "Duplicate charge",
    "Unexpected fee",
    "Incorrect charge",
    "Incorrect account information",
    "Account closure",
    "Debit card payment",
    "Overdraft fee",
    "Online banking access",
    "Repeated calls",
    "Incorrect collection record",
    "Written notice",
    "Payment plan",
    "Debt not owed",
    "Application delay",
    "Loss mitigation",
    "Escrow balance",
    "Recipient did not receive funds",
    "Wallet access",
    "Incorrect amount",
    "Unauthorized transfer",
    "Transfer fee",
    "Late fee",
    "Payoff quote",
    "Repossession notice",
    "Payment allocation",
    "Forbearance processing",
    "Confusing statement",
    "Incorrect balance",
    "Incorrect payment status",
    "Monthly fee",
    "Card locked",
    "Activation failure",
])

# ---------------------------------------------------------------------------
# 3.  Sentiment labels  (3 classes)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS: List[str] = ["Negative", "Neutral", "Positive"]

# ---------------------------------------------------------------------------
# 4.  Resolution status labels  (4 classes)
# ---------------------------------------------------------------------------
RESOLUTION_LABELS: List[str] = [
    "Resolved",
    "In review",
    "Unresolved",
    "Closed with explanation",
]

# ---------------------------------------------------------------------------
# 5.  Product labels  (8 unique)
# ---------------------------------------------------------------------------
PRODUCT_LABELS: List[str] = sorted([
    "Checking or savings account",
    "Credit card",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
])

# ---------------------------------------------------------------------------
# 6.  Channel labels  (6 unique)
# ---------------------------------------------------------------------------
CHANNEL_LABELS: List[str] = sorted([
    "Web",
    "Mobile app",
    "Phone",
    "Email",
    "Mail",
    "In person",
])

# ---------------------------------------------------------------------------
# 7.  Issue to Sub-Issue Mapping  (Hierarchical taxonomy)
# ---------------------------------------------------------------------------
from typing import Dict

ISSUE_TO_SUB_ISSUES: Dict[str, List[str]] = {
    "Account access": [
        "Card locked",
        "Online banking access",
        "Wallet access",
    ],
    "Account management": [
        "Account closed",
        "Account closure",
        "Payoff quote",
    ],
    "Application": [
        "Application delay",
    ],
    "Billing dispute": [
        "Duplicate charge",
        "Incorrect charge",
    ],
    "Card activation": [
        "Activation failure",
    ],
    "Collection practices": [
        "Debt not owed",
        "Repeated calls",
    ],
    "Communication": [
        "Confusing statement",
        "Written notice",
    ],
    "Credit reporting": [
        "Incorrect account information",
        "Incorrect collection record",
        "Incorrect payment status",
    ],
    "Deposits and withdrawals": [
        "Cash withdrawal",
    ],
    "Fees": [
        "Late fee",
        "Monthly fee",
        "Overdraft fee",
        "Transfer fee",
        "Unexpected fee",
    ],
    "Forbearance": [
        "Forbearance processing",
    ],
    "Foreclosure": [
        "Loss mitigation",
    ],
    "Fraud or scam": [
        "Unauthorized transaction",
        "Unauthorized transfer",
    ],
    "Interest rate": [
        "Rate increase",
    ],
    "Payment arrangement": [
        "Payment plan",
    ],
    "Payments": [
        "Debit card payment",
        "Payment not credited",
        "Payment processing",
    ],
    "Repayment": [
        "Payment allocation",
    ],
    "Repossession": [
        "Repossession notice",
    ],
    "Servicing": [
        "Escrow balance",
        "Incorrect balance",
    ],
    "Transfer error": [
        "Incorrect amount",
    ],
    "Transfer pending": [
        "Recipient did not receive funds",
    ],
}

SUB_ISSUE_TO_ISSUE: Dict[str, str] = {
    sub: issue
    for issue, subs in ISSUE_TO_SUB_ISSUES.items()
    for sub in subs
}

