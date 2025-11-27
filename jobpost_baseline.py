# jobpost_baseline.py
"""
Improved Fake Job/Internship Detector

Features added:
- Expanded metadata features (phone, url, email, generic email, whatsapp mention, wfh, upfront fee, text length)
- Scam keyword score (counts of suspicious keywords)
- Short-scam data augmentation injected into training set
- Auto-threshold in prediction based on text length (short texts use lower threshold)
- Saves: tfidf_vectorizer.joblib, meta_scaler.joblib, logreg_model_with_meta.joblib
"""

import os
import re
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc as sk_auc,
)
from sklearn.preprocessing import StandardScaler
from scipy import sparse

# -------- Config --------
CSV_PATH = "data/fake_job_postings.csv"   # update if needed
MODEL_DIR = "models"
RANDOM_STATE = 42
AUGMENT_REPEAT = 8        # how many times to replicate each short scam template in training
SHORT_TEXT_WORDS = 30     # threshold for "short" text (words)
AUTO_THRESH_SHORT = 0.35  # default threshold for short messages
AUTO_THRESH_LONG = 0.6    # default threshold for long messages
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
LABEL_COL = "fraudulent"  # 1=scam/fake, 0=real

# -------- Suspicious keyword lists (expandable) --------
SUSPICIOUS_KEYWORDS = [
    "earn", "earn daily", "earn per", "work from home", "work from", "data entry",
    "apply now", "registration fee", "security deposit", "pay", "payment", "daily",
    "guarantee", "get paid", "get started", "no experience", "urgent hiring",
    "whatsapp", "apply today", "limited seats", "click here", "start earning",
    "refundable", "commission", "instant joining", "no interview"
]
UPFRONT_FEE_KEYWORDS = ["registration fee", "security deposit", "pay", "payment", "fee", "deposit"]
WFH_KEYWORDS = ["work from home", "work from", "wfh"]
GENERIC_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]

# -------- Sample short scam messages for augmentation ----------
SHORT_SCAM_SAMPLES = [
    "We are urgently hiring Data Entry Operators to work from home. Earn up to Rs. 5,000 per day. No experience needed. Apply now!",
    "Earn daily income by doing simple tasks on your phone. No interview. Pay registration fee Rs. 999 to start. Contact WhatsApp 9876543210.",
    "Congratulations! You are selected for remote job. Salary ₹50,000/week. Send payment screenshot to confirm.",
    "Work from home, earn ₹2000/day. Limited seats. Visit www.quickearn-online.site to register.",
    "Paid internship at leading MNC. Stipend ₹1,20,000. Pay registration fee to activate account.",
]

# -------- Helpers --------
def load_data(path):
    df = pd.read_csv(path)
    keep = [c for c in TEXT_FIELDS if c in df.columns] + [LABEL_COL]
    df = df[keep].copy()
    df.fillna("", inplace=True)
    # combine relevant text fields into one string
    df["text"] = df[[c for c in TEXT_FIELDS if c in df.columns]].agg(" ".join, axis=1)
    return df

def clean_text(s: str):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s@.+:/-]", " ", s)  # allow @ . + : / - for detecting emails/urls/phones
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- metadata & feature extractors ---
phone_re = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})")
email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
url_re = re.compile(r"http[s]?://|www\.|\.in|\.com|\.site|\.online|\.biz")

def has_salary(text):
    return int(bool(re.search(r"\b(inr|rs\.?|rupees|\$|per month|per annum|/month|/year|per year|monthly|annum|₹)\b", text, flags=re.I)))

def has_phone(text):
    return int(bool(phone_re.search(text)))

def has_whatsapp(text):
    return int(bool(re.search(r"\bwhatsapp\b", text, flags=re.I)) or bool(re.search(r"\bwa[\s:-]?\d{8,}\b", text, flags=re.I)))

def has_url(text):
    return int(bool(url_re.search(text)))

def has_email(text):
    return int(bool(email_re.search(text)))

def generic_email(text):
    m = email_re.search(text)
    if not m:
        return 0
    domain = m.group(0).split("@")[-1].lower()
    return int(any(d in domain for d in GENERIC_EMAIL_DOMAINS))

def wfh_flag(text):
    return int(any(k in text for k in WFH_KEYWORDS))

def upfront_fee_flag(text):
    return int(any(k in text for k in UPFRONT_FEE_KEYWORDS))

def scam_keyword_count(text):
    cnt = 0
    for k in SUSPICIOUS_KEYWORDS:
        # count overlapping occurrences approximately
        cnt += len(re.findall(re.escape(k), text))
    return cnt

def text_length_words(text):
    return len(text.split())

def build_meta_matrix(texts):
    rows = []
    for t in texts:
        rows.append([
            has_salary(t),
            has_phone(t),
            has_whatsapp(t),
            has_url(t),
            has_email(t),
            generic_email(t),
            wfh_flag(t),
            upfront_fee_flag(t),
            scam_keyword_count(t),
            text_length_words(t),
        ])
    return np.array(rows, dtype=float)

# --- explainability helper (only uses TF-IDF vocab part as model includes meta) ---
def top_keywords_for_text(text, vectorizer, model, top_k=8):
    tf = vectorizer.transform([text])
    vocab = vectorizer.get_feature_names_out()
    vocab_size = len(vocab)
    coef = model.coef_[0][:vocab_size]
    scores = tf.toarray()[0] * coef
    feature_names = np.array(vocab)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return list(zip(feature_names[top_idx], scores[top_idx]))

def evaluate_at_thresholds(y_true, probs, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    print("\nThreshold analysis:")
    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds_t, average="binary", zero_division=0)
        print(f" threshold={t:.2f} -> Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

# -------- Augmentation helper (adds short scam samples into training set) --------
def augment_short_scams(X_train, y_train, repeat=AUGMENT_REPEAT):
    aug_texts = []
    aug_labels = []
    for s in SHORT_SCAM_SAMPLES:
        s_clean = clean_text(s)
        for i in range(repeat):
            # small random perturbations: add/remove a short phrase or number to diversify
            perturb = ""
            if random.random() < 0.4:
                perturb = " " + random.choice(["Apply now", "Limited seats", "Contact WhatsApp", "Registration fee"])
            if random.random() < 0.2:
                perturb += " " + str(random.randint(900, 9999))
            aug_texts.append((s_clean + perturb).strip())
            aug_labels.append(1)
    # append to training
    X_aug = np.concatenate([X_train, np.array(aug_texts)])
    y_aug = np.concatenate([y_train, np.array(aug_labels)])
    # shuffle
    perm = np.random.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]

# -------- Main pipeline --------
if __name__ == "__main__":
    print("Loading data:", CSV_PATH)
    df = load_data(CSV_PATH)
    print("Rows:", len(df))
    # Clean text column
    df["text"] = df["text"].apply(clean_text)
    X_all = df["text"].values
    y_all = df[LABEL_COL].astype(int).values

    # Train/test split before augmentation
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )
    print("Train / Test sizes:", X_train.shape[0], X_test.shape[0], "Label dist (train):", np.bincount(y_train))

    # Augment training with short scam messages (only to training split)
    X_train, y_train = augment_short_scams(X_train, y_train, repeat=AUGMENT_REPEAT)
    print("After augmentation, Train size:", len(X_train), "Label dist (train):", np.bincount(y_train.astype(int)))

    # Vectorize text
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    # Build metadata matrices (on cleaned text)
    meta_train = build_meta_matrix(X_train)
    meta_test  = build_meta_matrix(X_test)

    # Standard scale metadata
    scaler = StandardScaler()
    meta_train_scaled = scaler.fit_transform(meta_train)
    meta_test_scaled  = scaler.transform(meta_test)

    # Combine TF-IDF + metadata
    X_train_all = sparse.hstack([X_train_tfidf, sparse.csr_matrix(meta_train_scaled)], format="csr")
    X_test_all  = sparse.hstack([X_test_tfidf,  sparse.csr_matrix(meta_test_scaled)],  format="csr")

    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train_all, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test_all)[:,1]
    y_pred = (y_proba >= AUTO_THRESH_LONG).astype(int)  # eval default at long threshold

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"\nWITH METADATA+AUG -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}, ROC-AUC: {auc_score:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    joblib.dump(vect, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "meta_scaler.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "logreg_model_with_meta.joblib"))
    print("Saved vectorizer, scaler and model to", MODEL_DIR)

    # Example explainability: top tokens for some predicted scams
    print("\nExample explainability: top tokens for some predicted scams (pred==1)")
    shown = 0
    for i, (txt, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
        if pred == 1:
            toks = top_keywords_for_text(txt, vect, clf, top_k=6)
            print(f"\nExample idx {i} (true={true}, pred={pred})")
            print("Text snippet:", txt[:300])
            print("Top tokens and scores:", toks)
            # show metadata values for this sample
            m = meta_test[i]
            meta_names = ["has_salary","has_phone","has_whatsapp","has_url","has_email","generic_email","wfh_flag","upfront_fee","scam_kw_count","text_len"]
            print("Metadata:", dict(zip(meta_names, m.tolist())))
            shown += 1
        if shown >= 6:
            break

    # Predict helper with auto-threshold based on length (can override threshold param)
    def predict_text(text, threshold=None):
        t = clean_text(text)
        t_vec = vect.transform([t])
        m = np.array([[has_salary(t), has_phone(t), has_whatsapp(t), has_url(t), has_email(t),
                       generic_email(t), wfh_flag(t), upfront_fee_flag(t), scam_keyword_count(t),
                       text_length_words(t)]], dtype=float)
        m_scaled = scaler.transform(m)
        x_all = sparse.hstack([t_vec, sparse.csr_matrix(m_scaled)], format="csr")
        prob = float(clf.predict_proba(x_all)[0,1])
        # auto threshold if not provided
        if threshold is None:
            tl = int(text_length_words(t))
            threshold = AUTO_THRESH_SHORT if tl < SHORT_TEXT_WORDS else AUTO_THRESH_LONG
        label = int(prob >= threshold)
        toks = top_keywords_for_text(t, vect, clf, top_k=6)
        meta_names = ["has_salary","has_phone","has_whatsapp","has_url","has_email","generic_email","wfh_flag","upfront_fee","scam_kw_count","text_len"]
        meta_vals = m[0].tolist()
        meta_dict = dict(zip(meta_names, meta_vals))
        return {"label": label, "scam_score": prob, "top_tokens": toks, "meta": meta_dict, "used_threshold": threshold}

    # Demo predict for several short scam samples
    print("\nDemo predictions (auto-thresholding) for short scam templates:")
    for s in SHORT_SCAM_SAMPLES[:4]:
        print("\nText:", s)
        print("Prediction:", predict_text(s))

    # PR-AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = sk_auc(recall, precision)
    print(f"\nPR-AUC: {pr_auc:.3f}")

    evaluate_at_thresholds(y_test, y_proba, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])

    print("\nSaved model artifacts and helper predict_text() ready for use in backend/frontend.")
