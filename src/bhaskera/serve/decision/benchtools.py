"""Request sets, parity comparison and latency summaries for the decision benchmarks.

Standard library only. Request sets are deterministic (seeded) so every run and every
server sees exactly the same requests.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

SHORT_STATES = [
    "I was charged twice for the same order.",
    "The box arrived empty. This is the second time!",
    "Can you tell me when my subscription renews? I might upgrade to the annual plan.",
    "Your app keeps crashing every time I open the camera. Fix it or I'm switching to a competitor.",
    "Thanks so much, the replacement arrived yesterday and works perfectly.",
    "I ordered a blue jacket in size M but received a red one in size L.",
    "Please delete my account and all the data you hold about me.",
    "My card was declined but the money still left my bank account.",
    "Is there a student discount for the pro plan?",
    "The package says delivered but I never got it. The tracking photo shows a different door.",
    "I can't log in, the password reset email never arrives.",
    "Cancel my order #4471 immediately, I found it cheaper elsewhere.",
]

JSON_STATES = [
    {"item": "wireless mouse", "delivered": "5 days ago",
     "customer_message": "The box arrived empty. This is the second time!"},
    {"plan": "Business, monthly, 12 seats",
     "ticket": "Hi, we were billed twice for March. Refund the duplicate today or we cancel."},
    {"order_id": "A-1029", "status": "shipped", "days_since_order": 14,
     "customer_message": "Where is my order? It was supposed to arrive last week."},
    {"user": "new signup", "message": "How do I export my data to CSV?", "account_age_days": 2},
    {"review": "Battery died after two days. Waste of money.", "rating": 1, "product": "smartwatch"},
    {"email_subject": "URGENT: verify your account",
     "email_body": "Click this link within 24 hours or your account will be suspended: "
                   "http://secure-login.example.co",
     "sender": "support@examp1e.com"},
    {"message": "Great service, I will recommend you to my friends.", "channel": "chat"},
    {"contract_clause": "Either party may terminate this agreement with 30 days written notice.",
     "document": "MSA v3"},
]

AGENT_REPLIES = [
    "Thanks for reaching out, I'm looking into this now.",
    "Could you share your order number so I can check?",
    "I'm sorry about that. I've escalated this to our billing team.",
    "I understand the frustration; a replacement can be sent within 3 business days.",
    "Our records show the payment was processed on the 3rd.",
    "I've reset your account settings; please try again and let me know.",
]

QUESTIONS = [
    "Is this a billing problem?",
    "Is the customer upset?",
    "Does the customer ask for a refund?",
    "Does the customer threaten to leave or cancel?",
    "Does the customer say they received the wrong item?",
    "Is this a technical problem with the product?",
    "Is the customer asking a question about pricing?",
    "Does the message contain a request to delete personal data?",
    "Is the customer satisfied?",
    "Does the message mention a delivery problem?",
    "Is this message likely a phishing attempt?",
    "Does the customer mention a specific order number?",
    "Is the customer unable to access their account?",
    "Should this be escalated to a human agent?",
    "Is the tone of the message polite?",
    "Does the customer mention a competitor or a cheaper alternative?",
    "Is the message about a subscription?",
    "Does the customer report being charged more than once?",
    "Our policy refunds items reported missing within 30 days of delivery. "
    "Should this customer get a refund?",
    "Does the message contain a URL?",
    "Is the customer a new user?",
    "Is this a product review?",
    "Does the text describe a legal or contractual term?",
    "Is the message written in English?",
    "Does the customer ask about a discount?",
    "Is urgent action requested?",
    "Does the customer praise the service?",
    "Is the problem already resolved according to the text?",
    "Does the text mention money leaving a bank account?",
    "Would a spam filter be right to block this message?",
]

LENGTHS = ("short", "long", "document")
PARITY_KINDS = ("short",) * 4 + ("json",) * 3 + ("long",) * 2 + ("document",)
PARITY_COUNTS = (1, 1, 2, 3, 5, 10)


def _thread(rng: random.Random, turns: int) -> str:
    lines = ["Support thread:"]
    for _ in range(turns):
        lines.append(f"Customer: {rng.choice(SHORT_STATES)}")
        lines.append(f"Agent: {rng.choice(AGENT_REPLIES)}")
    return "\n".join(lines)


def _state(kind: str, rng: random.Random):
    if kind == "short":
        return rng.choice(SHORT_STATES)
    if kind == "json":
        return dict(rng.choice(JSON_STATES))
    if kind == "long":
        return _thread(rng, 4)  # ~190 tokens
    if kind == "document":
        return _thread(rng, 40)  # ~2k tokens
    raise ValueError(f"unknown state kind {kind!r}")


def _request(state, questions: list[str]) -> dict:
    return {
        "model": "jev-latest",
        "state": state,
        "questions": {
            f"q{index}": {"type": "noul", "instructions": text}
            for index, text in enumerate(questions)
        },
    }


def parity_requests(n: int = 300, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "id": f"p{index:03d}",
            "request": _request(
                _state(PARITY_KINDS[index % 10], rng),
                rng.sample(QUESTIONS, PARITY_COUNTS[index % 6]),
            ),
        }
        for index in range(n)
    ]


def bench_requests(length: str, count: int, n: int = 50, seed: int = 1) -> list[dict]:
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {LENGTHS}")
    rng = random.Random(f"{seed}-{length}-{count}")
    return [
        {"id": f"{length}-q{count}-{index:03d}",
         "request": _request(_state(length, rng), rng.sample(QUESTIONS, count))}
        for index in range(n)
    ]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(reference: list[dict], candidate: list[dict], tolerance: float = 0.01) -> dict:
    """Per-answer |ΔP(yes)| between two recordings of the same request set."""
    by_id = {row["id"]: row for row in candidate}
    deltas, missing, mismatched = [], [], []
    for ref in reference:
        other = by_id.get(ref["id"])
        if other is None:
            missing.append(ref["id"])
            continue
        if ref["status"] != 200 or other["status"] != 200:
            if ref["status"] != other["status"]:
                mismatched.append(ref["id"])
            continue
        for qid, answer in ref["response"]["answers"].items():
            got = other["response"]["answers"].get(qid)
            if got is None:
                missing.append(f"{ref['id']}/{qid}")
                continue
            a, b = answer["noul"], got["noul"]
            deltas.append((abs(a - b), ref["id"], qid, a, b))
    deltas.sort(reverse=True)
    worst = deltas[0][0] if deltas else 0.0
    return {
        "answers": len(deltas),
        "max_delta": worst,
        "mean_delta": math.fsum(d[0] for d in deltas) / len(deltas) if deltas else 0.0,
        "decisions_flipped": sum(1 for d in deltas if (d[3] > 0.5) != (d[4] > 0.5)),
        "worst": [
            {"id": i, "question": q, "reference": a, "candidate": b, "delta": d}
            for d, i, q, a, b in deltas[:10]
        ],
        "missing": missing,
        "status_mismatch": mismatched,
        "tolerance": tolerance,
        "passed": not missing and not mismatched and worst <= tolerance,
    }


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; NaN for no values."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def summarize(latencies: list[float], errors: int, questions: int, seconds: float) -> dict:
    return {
        "requests": len(latencies),
        "errors": errors,
        "seconds": seconds,
        "req_per_s": len(latencies) / seconds,
        "questions_per_s": questions / seconds,
        "p50_ms": 1000 * percentile(latencies, 50),
        "p90_ms": 1000 * percentile(latencies, 90),
        "p99_ms": 1000 * percentile(latencies, 99),
    }


COLUMNS = ("label", "requests", "concurrency", "completed", "req_per_s", "questions_per_s", "p50_ms",
           "p90_ms", "p99_ms", "errors", "max_memory_mib", "mean_util_pct")


def table(rows: list[dict]) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.1f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines)
