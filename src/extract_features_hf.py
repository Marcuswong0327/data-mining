"""
Real SLM/LLM feature extraction via Hugging Face (drop-in replacement for extract_offline).

    pip install transformers torch
    from extract_features_hf import extract_with_hf
    enriched = extract_with_hf(raw_df)

NOTE: written for the group to run with internet access. It was NOT executed in the sandbox where the
offline dataset was built (no network). Expect ~5-15 min on CPU for 1,500 rows; use batch_size / GPU to speed up.
Models are small (<= ~270M params) => "SLMs" in the sense of the project brief.
"""
import pandas as pd

SENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"       # multilingual: copes with English/Malay code-mixing
ZS_MODEL = "typeform/distilbert-base-uncased-mnli"                  # light zero-shot classifier
TOPICS = {"repayment_hardship": "customer cannot afford loan repayment", "fees_charges": "dispute about fees or charges",
          "app_service_issue": "app or customer service problem", "fraud_security": "fraud or suspicious transaction",
          "product_inquiry": "question about a banking product", "praise_loyalty": "praise or satisfaction"}
RISK_LABELS = ["low credit risk, customer is stable", "medium credit risk, some concern", "high credit risk, financial distress"]


def extract_with_hf(df, batch_size=16):
    from transformers import pipeline
    out = df.copy()
    fb = out["customer_feedback"].fillna("")
    nt = out["agent_note"].fillna("")
    full = (fb + " " + nt).str.strip().tolist()

    sent = pipeline("sentiment-analysis", model=SENT_MODEL, top_k=None, truncation=True)
    res = sent(fb.replace("", "no comment").tolist(), batch_size=batch_size)
    sc, lab = [], []
    for r in res:
        d = {x["label"].lower(): x["score"] for x in r}
        sc.append(d.get("positive", 0) - d.get("negative", 0))
        lab.append(max(d, key=d.get).capitalize())
    out["sentiment_score"], out["sentiment_label"] = pd.Series(sc).round(3).values, lab

    zs = pipeline("zero-shot-classification", model=ZS_MODEL)
    inv = {v: k for k, v in TOPICS.items()}
    t = zs(fb.replace("", "no comment").tolist(), candidate_labels=list(TOPICS.values()), batch_size=batch_size)
    out["text_topic"] = [inv[r["labels"][0]] for r in t]
    out.loc[fb == "", "text_topic"] = "no_feedback"

    r = zs(full, candidate_labels=RISK_LABELS, batch_size=batch_size)
    probs = [dict(zip(x["labels"], x["scores"])) for x in r]
    out["text_risk_score"] = [round(p[RISK_LABELS[1]] * 0.5 + p[RISK_LABELS[2]], 3) for p in probs]
    out["text_risk_category"] = pd.cut(out["text_risk_score"], [-0.01, 0.35, 0.60, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    out["has_feedback"] = (fb != "").astype(int)
    out["has_agent_note"] = (nt != "").astype(int)
    return out


# Optional: scale up the TEXT generation with an LLM (needs an API key; not used for the submitted CSV)
LLM_TEXT_PROMPT = """You are simulating a Malaysian retail-banking customer. Write ONE short feedback message (1-3 sentences).
Profile: age {age}, {employment_status}, product {product_type}, DTI {dti_ratio}, late payments last 12m {late_payments_12m}.
Hidden financial-stress level (0 calm - 10 distressed): {stress}.
Style: natural, occasionally code-mixed English/Malay, may contain a typo. Do NOT mention the words 'default' or 'stress level'."""
