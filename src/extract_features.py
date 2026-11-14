"""
Text feature extraction + customer segmentation.

Two interchangeable back-ends produce the SAME output columns:
  * extract_offline(df)  - lightweight rule/lexicon "SLM stand-in" (runs anywhere, no downloads)
  * extract_with_hf(df)  - real small language models via Hugging Face (see extract_features_hf.py)

Output columns: sentiment_score, sentiment_label, text_topic, text_risk_score, text_risk_category
Segmentation  : customer_segment (KMeans on behaviour + sentiment, named from centroids)
"""
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

POS = {"happy", "great", "thank", "thanks", "pleased", "glad", "impressed", "excellent", "smooth", "instant", "friendly",
       "fast", "competitive", "worth", "easy", "transparent", "quickly", "well", "recommend", "satisfied", "healthy",
       "terima", "kasih", "good", "clearly", "cooperative", "strong"}
NEG = {"disappointed", "frustrating", "unhappy", "unacceptable", "not acceptable", "upset", "terrible", "fed", "stressed",
       "crashing", "cannot", "cant", "never arrives", "unfair", "unexplained", "overdue", "behind", "struggling", "worried",
       "drained", "retrenched", "missed", "slow", "lost", "charged", "complain", "closing", "suspicious", "fraud", "lambat",
       "heavy", "evasive", "anxious", "broken", "loss", "sangat", "nothing changed", "fix", "wait", "waited", "twice"}
NEGATORS = {"not", "never", "no", "cannot", "cant", "without"}
INTENS = {"very": 1.3, "really": 1.3, "extremely": 1.5, "so": 1.2, "quite": 1.1}

TOPIC_KEYWORDS = {
    "repayment_hardship": ["salary was cut", "retrench", "defer", "restructur", "behind on", "overdue", "drained", "reschedule",
                           "cannot keep up", "struggling", "too heavy", "out of control", "reduced", "lower payment", "not be able to pay",
                           "missed the last payment", "income came in late", "relief", "instalment is", "installment is"],
    "fees_charges": ["fee", "charge", "interest charged", "overlimit", "billed", "service charge"],
    "app_service_issue": ["app keeps", "crash", "log in", "login", "otp", "hotline", "queue", "pending", "customer service",
                          "servise", "different answers", "online banking", "ticket"],
    "fraud_security": ["did not make", "suspicious", "sms", "overseas", "tac", "unknown online", "block", "fraud"],
    "product_inquiry": ["refinance", "documents do i need", "early settlement", "fixed deposit", "auto-debit", "promotion",
                        "eligibility", "top-up", "how do i", "interested in", "enquiry", "brochure"],
    "praise_loyalty": ["helped me", "smooth", "friendly", "approved fast", "never had any problem", "cashback", "easy and transparent",
                       "recommend", "well done", "great service", "compliment", "satisfied", "loyalty", "courtesy"],
}
HARDSHIP = ["retrench", "salary was cut", "defer", "restructur", "behind on", "overdue", "drained", "cannot keep up", "struggling",
            "too heavy", "reduced", "not be able to pay", "missed the last payment", "income came in late", "income loss", "watchlist",
            "promise-to-pay broken", "broken", "anxious", "2nd request", "at risk", "evasive", "missed contact", "collections",
            "akpk", "out of control", "payment relief", "revised payment plan", "deferment", "expenses are now higher", "lower payment plan"]
PROTECTIVE = ["good standing", "no concerns", "strong repayment", "cooperative", "healthy", "never had any problem", "upsell",
              "satisfied", "loyalty", "recommend", "well done", "slightly irregular"]
PROTECTIVE = [p for p in PROTECTIVE if p != "slightly irregular"]
MILD = ["slightly irregular", "mild concern", "monitor next cycle", "asked for more time"]


def _tok(t):
    return re.findall(r"[a-z\-']+", t.lower())


def _sentiment(text):
    toks = _tok(text)
    s = 0.0
    for i, w in enumerate(toks):
        v = 1.0 if w in POS else (-1.0 if w in NEG else 0.0)
        if v == 0:
            continue
        window = toks[max(0, i - 2):i]
        if any(x in NEGATORS for x in window) and w not in NEGATORS:
            v = -v * 0.7
        for x in window:
            v *= INTENS.get(x, 1.0)
        s += v
    return s / np.sqrt(s * s + 6)        # squash to (-1, 1)


def _topic(text):
    t = text.lower()
    sc = {k: sum(kw in t for kw in kws) for k, kws in TOPIC_KEYWORDS.items()}
    best = max(sc, key=sc.get)
    return best if sc[best] > 0 else "product_inquiry"


def _risk(text, sent):
    t = text.lower()
    h = sum(k in t for k in HARDSHIP)
    p = sum(k in t for k in PROTECTIVE)
    m = sum(k in t for k in MILD)
    raw = 1.0 * h + 0.35 * m - 0.8 * p - 0.9 * sent - 0.4
    return 1 / (1 + np.exp(-raw))


def extract_offline(df):
    out = df.copy()
    fb = out["customer_feedback"].fillna("")
    nt = out["agent_note"].fillna("")
    full = (fb + " " + nt).str.strip()
    out["sentiment_score"] = fb.map(_sentiment).round(3)
    out.loc[fb == "", "sentiment_score"] = 0.0
    out["sentiment_label"] = pd.cut(out["sentiment_score"], [-1.01, -0.15, 0.15, 1.01], labels=["Negative", "Neutral", "Positive"]).astype(str)
    out["text_topic"] = fb.map(_topic)
    out.loc[fb == "", "text_topic"] = "no_feedback"
    out["text_risk_score"] = [round(_risk(t, s), 3) for t, s in zip(full, out["sentiment_score"])]
    out["text_risk_category"] = pd.cut(out["text_risk_score"], [-0.01, 0.35, 0.60, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    out["has_feedback"] = (fb != "").astype(int)
    out["has_agent_note"] = (nt != "").astype(int)
    return out


# ------------------------------- segmentation -------------------------------
SEG_FEATURES = ["monthly_income", "savings_balance", "txn_count_monthly", "digital_channel_ratio", "tenure_months",
                "credit_utilization", "num_products_held", "dti_ratio", "sentiment_score"]


def segment_customers(df, k=5, k_range=range(3, 8), seed=42):
    """KMeans segmentation. Silhouette is reported for k_range but is flat on behavioural data,
    so k is fixed (default 5) for business interpretability."""
    X = df[SEG_FEATURES].copy()
    X = X.fillna(X.median())
    X["monthly_income"] = np.log1p(X["monthly_income"])
    X["savings_balance"] = np.log1p(X["savings_balance"])
    Xs = StandardScaler().fit_transform(X)
    scores = {kk: silhouette_score(Xs, KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(Xs)) for kk in k_range}
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    out = df.copy()
    out["segment_id"] = km.labels_
    prof = out.groupby("segment_id")[SEG_FEATURES].mean()
    out["customer_segment"] = out["segment_id"].map(_name_segments(prof))
    return out, scores, k, prof


ARCHETYPES = {
    "Stretched Borrowers":     {"dti_ratio": 1, "credit_utilization": 1, "sentiment_score": -1},
    "Affluent Active":         {"monthly_income": 1, "savings_balance": 1, "txn_count_monthly": 1},
    "Traditional Branch Users": {"digital_channel_ratio": -1, "txn_count_monthly": -0.5},
    "Loyal Multi-Product":     {"tenure_months": 1.5},
    "Entry-Level Everyday":    {"monthly_income": -1, "savings_balance": -1},
}


def _name_segments(prof):
    """One-to-one assignment of clusters to business archetypes (Hungarian algorithm on centroid z-scores)."""
    from scipy.optimize import linear_sum_assignment
    z = (prof - prof.mean()) / (prof.std() + 1e-9)
    arch = list(ARCHETYPES)
    cost = np.array([[-sum(w * z.loc[sid, f] for f, w in ARCHETYPES[a].items()) for a in arch] for sid in z.index])
    r, c = linear_sum_assignment(cost)
    names = {z.index[i]: arch[j] for i, j in zip(r, c)}
    for sid in z.index:                       # if k > number of archetypes
        names.setdefault(sid, f"Other Segment {sid}")
    return names
