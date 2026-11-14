"""
Synthetic retail-banking dataset generator (WIE3007 Group Project, Component 1)

Design
------
1. Numerical layer : correlated customer / credit / behaviour variables (RM, Malaysian context).
2. Latent layer    : a hidden "financial stress" score and a hidden "recent life-event shock".
                     The default flag is generated from numerics + shock.
3. Text layer      : customer_feedback + agent_note are generated from the latent text-stress
                     (stress + shock + noise), NOT from the default flag -> no target leakage,
                     but the text carries information the numeric columns do not (the shock).
The phrase pools below were authored with GenAI (Claude) and are composed by a grammar so that
1,500 records are linguistically varied (code-mixed English/Malay, typos, casing noise, missing notes).
"""
import numpy as np
import pandas as pd

SEED = 42
TARGET_DEFAULT_RATE = 0.13

REGIONS = ["Selangor", "Kuala Lumpur", "Johor", "Penang", "Perak", "Sabah", "Sarawak",
           "Kedah", "Negeri Sembilan", "Melaka", "Pahang"]
REGION_P = [0.22, 0.16, 0.13, 0.09, 0.08, 0.07, 0.07, 0.06, 0.04, 0.04, 0.04]
REGION_INCOME_FACTOR = {"Kuala Lumpur": 1.30, "Selangor": 1.18, "Penang": 1.10, "Johor": 1.05,
                        "Melaka": 0.98, "Negeri Sembilan": 0.97, "Perak": 0.92, "Pahang": 0.92,
                        "Sarawak": 0.95, "Sabah": 0.90, "Kedah": 0.88}
EMPLOYMENT = ["Salaried", "Self-employed", "Civil servant", "Contract/Gig", "Unemployed/Retired"]
EMPLOYMENT_P = [0.52, 0.18, 0.13, 0.12, 0.05]
EMP_BASE_INCOME = {"Salaried": 5200, "Self-employed": 6000, "Civil servant": 4800,
                   "Contract/Gig": 3600, "Unemployed/Retired": 2200}
EMP_STRESS = {"Salaried": 0.0, "Self-employed": 0.25, "Civil servant": -0.3,
              "Contract/Gig": 0.35, "Unemployed/Retired": 0.6}
EDU = ["SPM/Below", "Diploma", "Degree", "Postgraduate"]
EDU_P = [0.25, 0.25, 0.40, 0.10]
EDU_INCOME = {"SPM/Below": 0.80, "Diploma": 0.95, "Degree": 1.15, "Postgraduate": 1.45}
PRODUCTS = ["Credit Card", "Personal Loan", "Auto Loan", "Home Loan", "SME Loan"]
PRODUCT_P = [0.35, 0.25, 0.18, 0.12, 0.10]


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------------------
# 1. NUMERICAL LAYER
# --------------------------------------------------------------------------------------
def make_numeric(n, rng):
    df = pd.DataFrame({"customer_id": [f"CUST{100000 + i}" for i in range(n)]})
    df["age"] = np.clip(rng.normal(38, 11, n), 21, 70).round().astype(int)
    df["gender"] = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    df["marital_status"] = rng.choice(["Single", "Married", "Divorced"], n, p=[0.38, 0.55, 0.07])
    df["dependents"] = np.clip(rng.poisson(1.3, n) * (df["marital_status"] != "Single") + rng.poisson(0.2, n), 0, 6)
    df["education"] = rng.choice(EDU, n, p=EDU_P)
    df["employment_status"] = rng.choice(EMPLOYMENT, n, p=EMPLOYMENT_P)
    df["region"] = rng.choice(REGIONS, n, p=REGION_P)

    base = df["employment_status"].map(EMP_BASE_INCOME).values
    income = (base * rng.lognormal(0, 0.45, n)
              * (1 + 0.012 * (df["age"].values - 30))
              * df["region"].map(REGION_INCOME_FACTOR).values
              * df["education"].map(EDU_INCOME).values)
    df["monthly_income"] = np.clip(income, 1200, 60000).round(-1)

    max_tenure = (df["age"].values - 18) * 12
    df["tenure_months"] = np.minimum(np.clip(rng.gamma(2.2, 30, n), 3, 240), max_tenure).round().astype(int)

    # credit bureau style score (CTOS range 300-850)
    score = (665 + 38 * (np.log(df["monthly_income"].values) - np.log(4500))
             + 0.25 * np.minimum(df["tenure_months"].values, 120) + rng.normal(0, 55, n))
    df["ctos_score"] = np.clip(score, 300, 850).round().astype(int)

    # product & exposure
    prod = rng.choice(PRODUCTS, n, p=PRODUCT_P)
    prod = np.where((prod == "SME Loan") & (df["employment_status"] != "Self-employed"), "Personal Loan", prod)
    df["product_type"] = prod
    inc = df["monthly_income"].values
    mult = {"Credit Card": (1.5, 4), "Personal Loan": (3, 12), "Auto Loan": (8, 18),
            "Home Loan": (30, 70), "SME Loan": (8, 30)}
    df["loan_amount"] = [round(inc[i] * rng.uniform(*mult[p]), -2) for i, p in enumerate(prod)]
    term_choices = {"Credit Card": [12], "Personal Loan": [12, 24, 36, 48, 60, 84],
                    "Auto Loan": [60, 72, 84, 108], "Home Loan": [240, 300, 360, 420],
                    "SME Loan": [24, 36, 60, 84]}
    df["loan_term_months"] = [int(rng.choice(term_choices[p])) for p in prod]
    base_rate = {"Credit Card": 15.0, "Personal Loan": 8.5, "Auto Loan": 3.8, "Home Loan": 4.2, "SME Loan": 6.5}
    rate = (np.array([base_rate[p] for p in prod]) + np.clip((700 - df["ctos_score"].values) / 100, -0.5, 3) * 1.3
            + rng.normal(0, 0.4, n))
    df["interest_rate_pct"] = np.clip(rate, 2.5, 18).round(2)

    r = df["interest_rate_pct"].values / 1200
    L = df["loan_amount"].values
    N = df["loan_term_months"].values
    inst = L * r / (1 - (1 + r) ** (-N))
    df["credit_utilization"] = np.clip(rng.beta(2, 3.2, n), 0.01, 1.0).round(3)
    is_cc = prod == "Credit Card"
    inst = np.where(is_cc, 0.05 * L * df["credit_utilization"].values, inst)
    df["monthly_installment"] = inst.round(2)
    other = np.where(rng.random(n) < 0.6, inc * rng.uniform(0, 0.25, n), 0)
    df["other_monthly_debt"] = other.round(2)
    df["dti_ratio"] = ((inst + other) / inc).round(3)

    df["late_payments_12m"] = rng.poisson(
        0.15 + 1.6 * _sigmoid(-(df["ctos_score"].values - 600) / 45) + 0.8 * df["credit_utilization"].values ** 2)
    df["savings_balance"] = (inc * rng.lognormal(1.0, 1.0, n)).round(2)
    dr_mean = np.clip(0.9 - 0.012 * (df["age"].values - 20), 0.2, 0.95)
    df["digital_channel_ratio"] = np.clip(rng.normal(dr_mean, 0.15), 0.02, 1.0).round(3)
    df["num_products_held"] = 1 + rng.poisson(1.2, n)
    df["txn_count_monthly"] = rng.poisson(18 + 14 * df["digital_channel_ratio"].values + inc / 1500)
    df["avg_txn_amount"] = (inc * 0.02 * rng.lognormal(0, 0.6, n)).round(2)
    return df


# --------------------------------------------------------------------------------------
# 2. LATENT LAYER  (stress, shock, default)
# --------------------------------------------------------------------------------------
def add_latent_and_target(df, rng):
    n = len(df)
    emp_eff = df["employment_status"].map(EMP_STRESS).values
    sav_ratio = np.log1p(df["savings_balance"].values / df["monthly_income"].values)
    z = (-0.9 * _z(df["ctos_score"]) + 0.7 * _z(df["dti_ratio"]) + 0.6 * _z(df["credit_utilization"])
         + 0.5 * _z(df["late_payments_12m"]) - 0.25 * _z(sav_ratio) + emp_eff + 0.6 * rng.normal(0, 1, n))
    z = _z(z)
    shock = rng.normal(0, 1, n)          # unobserved life event (job loss, illness...) -> mostly visible only in text

    logit_core = 1.5 * z + 0.8 * shock + 0.25 * (df["employment_status"] == "Unemployed/Retired").values
    lo, hi = -8, 2                        # calibrate intercept to hit the target default rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(mid + logit_core).mean() > TARGET_DEFAULT_RATE:
            hi = mid
        else:
            lo = mid
    p_default = _sigmoid(mid + logit_core)
    df["default_flag"] = (rng.random(n) < p_default).astype(int)

    text_stress = _z(0.6 * z + 0.6 * shock + 0.5 * rng.normal(0, 1, n))
    return df, z, shock, text_stress


# --------------------------------------------------------------------------------------
# 3. TEXT LAYER  (GenAI-authored phrase pools + grammar)
# --------------------------------------------------------------------------------------
OPENERS = {
    "neg": ["Honestly very disappointed.", "This is getting really frustrating.", "I am not happy with this at all.",
            "Not acceptable.", "I have raised this before and nothing changed.", "Quite upset right now.",
            "Terrible experience this month.", "I'm fed up.", "Very stressed about this."],
    "neu": ["Hi team,", "Just a note from my side.", "Writing in regarding my account.", "Sharing some feedback.",
            "Following up on my last contact.", "Hello,", "Quick message about my account.", "To whom it may concern,"],
    "pos": ["Really happy to share this.", "Great service as always!", "Thank you so much.", "Very pleased with the bank.",
            "Just wanted to say well done.", "Glad I chose this bank.", "Impressed with the team.", "Excellent experience."],
}
CLOSERS = {
    "neg": ["Please fix this soon.", "I expect a proper response.", "If not I will consider closing my account.",
            "Really hope someone takes this seriously.", "I don't know what else to do.", ""],
    "neu": ["Please advise.", "Thanks.", "Let me know the next step.", "Appreciate a reply.", "", ""],
    "pos": ["Keep it up!", "Will recommend to my friends.", "Thanks again.", "Appreciate it.", "", ""],
}

BODIES = {
    "repayment_hardship": [
        "My salary was cut and I cannot keep up with the RM{amount} instalment.",
        "I was retrenched in {month} and I am struggling to make the monthly payment.",
        "Business has been slow lately, can I defer my repayment for {n} months?",
        "I missed the last payment by {days} days because my income came in late.",
        "Please advise on restructuring options, my expenses are now higher than my income.",
        "Medical bills have drained my savings and I am behind on my loan.",
        "I'm worried I will not be able to pay this month, can we reschedule?",
        "I had to use my credit card for daily needs and the balance is getting out of control.",
        "My hours were reduced at work and the instalment is becoming too heavy.",
        "I am already {days} days overdue and need help with a lower payment plan.",
    ],
    "fees_charges": [
        "You charged me RM{fee} late fee even though I paid on time.",
        "The annual fee was billed again after I asked to waive it.",
        "Why is there an unexplained RM{fee} service charge on my statement?",
        "The interest charged is higher than what the officer quoted me.",
        "The processing fee was not clearly explained when I signed up.",
        "An overlimit charge of RM{fee} is unfair.",
        "I was charged twice for the same fee this month.",
    ],
    "app_service_issue": [
        "The mobile app keeps crashing when I try to transfer money.",
        "I cannot log in to online banking for {n} days now.",
        "The OTP never arrives so I cannot complete my payment.",
        "I waited {mins} minutes on the hotline before anyone answered.",
        "The branch queue at {region} was very long and the staff seemed lost.",
        "Payment in the app shows pending but the money was already deducted.",
        "Customer service gave me three different answers on the same issue.",
    ],
    "fraud_security": [
        "I saw a transaction of RM{amount} that I did not make.",
        "Got a suspicious SMS pretending to be from the bank and I am worried about my account.",
        "My card was used overseas but I never travelled, please block it.",
        "Someone called asking for my TAC number, is this legitimate?",
        "There is an unknown online purchase on my card statement.",
    ],
    "product_inquiry": [
        "Is there a lower interest rate if I refinance my {product}?",
        "What documents do I need to increase my credit limit?",
        "Can I make an early settlement without penalty?",
        "Interested in a fixed deposit, what are the current rates?",
        "How do I set up auto-debit for my monthly instalment?",
        "Do you have any promotion for balance transfer?",
        "I would like to know the eligibility for a top-up loan.",
    ],
    "praise_loyalty": [
        "My relationship manager helped me settle the paperwork very quickly.",
        "The app is smooth and transfers are instant.",
        "The staff at the {region} branch were friendly and explained everything clearly.",
        "I got approved fast and the rate is competitive.",
        "I have never had any problem with my payments or the service.",
        "The cashback on my card has been really worth it.",
        "Setting up my instalment plan was easy and transparent.",
    ],
}
TOPICS = list(BODIES.keys())
MANGLISH = ["lah", "la", "boleh tolong?", "tolong check ya", "macam mana?", "please lah", "terima kasih", "sangat lambat",
            "harap cepat sikit", "kenapa macam ni?"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September"]
CHANNELS = ["Hotline", "WhatsApp", "Branch", "App chat", "Email"]

NOTE_TOPIC_PHRASE = {
    "repayment_hardship": ["repayment difficulty", "inability to meet instalment", "request for payment relief"],
    "fees_charges": ["fee dispute", "charges query", "interest rate complaint"],
    "app_service_issue": ["app/login issue", "service delay complaint", "OTP problem"],
    "fraud_security": ["suspicious transaction", "fraud concern", "card block request"],
    "product_inquiry": ["product enquiry", "refinancing enquiry", "limit increase enquiry"],
    "praise_loyalty": ["positive feedback", "courtesy call", "satisfaction check"],
}
NOTE_ACTION = {
    "repayment_hardship": ["Offered 3-month deferment", "Referred to collections/AKPK restructuring", "Arranged revised payment plan",
                           "Advised to submit income proof"],
    "fees_charges": ["Fee reversed as goodwill", "Explained fee schedule", "Escalated to billing team"],
    "app_service_issue": ["Logged ticket with IT", "Reset online credentials", "Advised to reinstall app"],
    "fraud_security": ["Card blocked, new card issued", "Referred to fraud desk", "Advised to change PIN"],
    "product_inquiry": ["Sent product brochure", "Scheduled RM callback", "Checked eligibility"],
    "praise_loyalty": ["Logged compliment", "Thanked customer", "Offered loyalty promo"],
}
NOTE_RISK = {
    "high": ["Cust sounds anxious, 2nd request this quarter", "Previous promise-to-pay broken", "Cust reports income loss, watchlist suggested",
             "Repeated missed contact, account at risk", "Cust evasive about repayment date"],
    "mid": ["No major concerns, follow up in 7 days", "Cust asked for more time, plan reasonable", "Mild concern, monitor next cycle",
            "Payment pattern slightly irregular"],
    "low": ["Cust cooperative, account in good standing", "No concerns noted", "Cust satisfied, strong repayment record",
            "Account healthy, upsell candidate"],
}
TYPO_MAP = {"payment": "paymnet", "because": "becuase", "instalment": "installment", "account": "acount",
            "please": "plz", "cannot": "cant", "receive": "recieve", "service": "servise", "charged": "charged"}


def _sample_topic(ts, rng):
    w = np.array([0.8 * np.exp(1.25 * ts), 1.0 * np.exp(0.3 * ts), 1.0, 0.4, 1.2 * np.exp(-0.4 * ts), 1.3 * np.exp(-0.85 * ts)])
    return rng.choice(TOPICS, p=w / w.sum())


def _sample_tone(topic, ts, rng):
    if topic == "praise_loyalty":
        return rng.choice(["pos", "neu"], p=[0.88, 0.12])
    if topic == "product_inquiry":
        return rng.choice(["neu", "pos", "neg"], p=[0.75, 0.15, 0.10])
    p_neg = float(np.clip(0.55 + 0.15 * ts, 0.25, 0.9))
    return rng.choice(["neg", "neu"], p=[p_neg, 1 - p_neg])


def _fill(t, row, rng):
    return t.format(amount=int(rng.choice([150, 280, 450, 780, 1200, 2300])), fee=int(rng.choice([10, 25, 50, 80, 120])),
                    days=int(rng.integers(3, 60)), n=int(rng.integers(2, 6)), mins=int(rng.choice([20, 35, 45, 60])),
                    month=rng.choice(MONTHS), region=row["region"], product=row["product_type"].lower())


def _noise(text, rng):
    if rng.random() < 0.20:
        text = text.rstrip(".!?") + " " + rng.choice(MANGLISH)
    if rng.random() < 0.06:
        for k, v in TYPO_MAP.items():
            if k in text:
                text = text.replace(k, v, 1)
                break
    if rng.random() < 0.10:
        text = text.lower()
    elif rng.random() < 0.03:
        text = text.upper()
    return text


def make_text(df, text_stress, rng):
    fb, nt, truth = [], [], []
    for i, row in df.iterrows():
        ts = text_stress[i]
        topic = _sample_topic(ts, rng)
        tone = _sample_tone(topic, ts, rng)
        body_t = rng.choice(BODIES[topic])
        body = _fill(body_t, row, rng)
        parts = [rng.choice(OPENERS[tone]), body]
        if rng.random() < 0.45:                       # optional second sentence
            topic2 = topic if rng.random() < 0.6 else rng.choice(TOPICS)
            t1 = body_t
            t2 = rng.choice(BODIES[topic2])
            while t2 == t1:
                t2 = rng.choice(BODIES[topic2])
            body2 = _fill(t2, row, rng)
            parts.append(body2)
        parts.append(rng.choice(CLOSERS[tone]))
        feedback = _noise(" ".join(p for p in parts if p), rng)
        fb.append(None if rng.random() < 0.04 else feedback)

        band = "high" if ts > 0.45 else ("low" if ts < -0.45 else "mid")
        # agent notes are noisy: 20% of the time the agent writes a risk comment from the adjacent band
        shown = band if rng.random() > 0.20 else rng.choice(["low", "mid", "high"])
        note = (f"[{rng.choice(CHANNELS)}] Cust contacted re {rng.choice(NOTE_TOPIC_PHRASE[topic])}. "
                f"{rng.choice(NOTE_ACTION[topic])}. {rng.choice(NOTE_RISK[shown])}.")
        nt.append(None if rng.random() < 0.22 else note)
        truth.append({"customer_id": row["customer_id"], "true_topic": topic,
                      "true_sentiment": {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}[tone],
                      "true_text_stress_band": band, "true_text_stress": round(float(ts), 3)})
    return fb, nt, pd.DataFrame(truth)


# --------------------------------------------------------------------------------------
# 4. DATA-QUALITY NOISE + ASSEMBLY
# --------------------------------------------------------------------------------------
def inject_quality_issues(df, rng):
    n = len(df)
    for col, frac in [("savings_balance", 0.04), ("ctos_score", 0.02), ("employment_status", 0.01), ("digital_channel_ratio", 0.02)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df[col] = df[col].astype(object) if col == "employment_status" else df[col].astype(float)
        df.loc[idx, col] = np.nan
    idx = rng.choice(n, 8, replace=False)                 # a few extreme outliers
    df.loc[idx, "avg_txn_amount"] = (df.loc[idx, "avg_txn_amount"] * rng.uniform(8, 15, 8)).round(2)
    return df


def generate(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    df = make_numeric(n, rng)
    df, z, shock, text_stress = add_latent_and_target(df, rng)
    df["num_service_calls_90d"] = rng.poisson(np.exp(-0.3 + 0.35 * text_stress))
    fb, nt, truth = make_text(df, text_stress, rng)
    df["customer_feedback"] = fb
    df["agent_note"] = nt
    df = inject_quality_issues(df, rng)
    cols = [c for c in df.columns if c != "default_flag"] + ["default_flag"]
    return df[cols], truth


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    raw, truth = generate()
    raw.to_csv("data/bank_customers_raw.csv", index=False)
    truth.to_csv("data/ground_truth_text_labels.csv", index=False)
    print(raw.shape, "default rate:", raw["default_flag"].mean().round(3))
