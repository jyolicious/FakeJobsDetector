# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os, re, numpy as np
from scipy import sparse

# -----------------------------------------------
# Load model artifacts (exact same names as training)
# -----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

vect = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "meta_scaler.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "logreg_model_with_meta.joblib"))

# -----------------------------------------------
# Config (must match jobpost_baseline.py)
# -----------------------------------------------
SHORT_TEXT_WORDS = 30
AUTO_THRESH_SHORT = 0.35
AUTO_THRESH_LONG = 0.6

# -----------------------------------------------
# Regex + keyword lists (same as training)
# -----------------------------------------------
email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
phone_re = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})")
url_re = re.compile(r"http[s]?://|www\.|\.in|\.com|\.site|\.online|\.biz")

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

# -----------------------------------------------
# Helper functions (EXACT same as training)
# -----------------------------------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s@.+:/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_salary(text):
    return int(bool(re.search(r"\b(inr|rs\.?|rupees|\$|per month|per annum|monthly|annum|₹)\b", text)))

def has_phone(text):
    return int(bool(phone_re.search(text)))

def has_whatsapp(text):
    # guard in case text is None (shouldn't be after cleaning, but safe)
    if text is None:
        return 0
    # check either the literal word "whatsapp" or patterns like "wa:98765..."
    m = re.search(r"\bwa[\s:-]?\d{8,}", text)
    return int(bool("whatsapp" in text or m))

def has_url(text):
    return int(bool(url_re.search(text)))

def has_email(text):
    return int(bool(email_re.search(text)))

def generic_email(text):
    m = email_re.search(text)
    if not m:
        return 0
    domain = m.group(0).split("@")[-1]
    return int(domain in GENERIC_EMAIL_DOMAINS)

def wfh_flag(text):
    return int(any(k in text for k in WFH_KEYWORDS))

def upfront_fee_flag(text):
    return int(any(k in text for k in UPFRONT_FEE_KEYWORDS))

def scam_keyword_count(text):
    total = 0
    for k in SUSPICIOUS_KEYWORDS:
        total += len(re.findall(re.escape(k), text))
    return total

def text_length_words(text):
    return len(text.split())

def build_meta(text):
    return np.array([[
        has_salary(text),
        has_phone(text),
        has_whatsapp(text),
        has_url(text),
        has_email(text),
        generic_email(text),
        wfh_flag(text),
        upfront_fee_flag(text),
        scam_keyword_count(text),
        text_length_words(text)
    ]], dtype=float)

# ------------------ Explainability --------------------
def top_keywords_for_text(text, vectorizer, model, top_k=6):
    tfidf = vectorizer.transform([text])
    vocab = vectorizer.get_feature_names_out()
    vocab_size = len(vocab)

    coef = model.coef_[0][:vocab_size]  
    scores = tfidf.toarray()[0] * coef

    top_idx = scores.argsort()[-top_k:][::-1]
    
    return [(vocab[i], float(scores[i])) for i in top_idx]

# -----------------------------------------------
# Flask app
# -----------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    manual_threshold = data.get("threshold", None)

    cleaned = clean_text(text)

    # Vectorize text
    text_vec = vect.transform([cleaned])

    # Metadata
    meta = build_meta(cleaned)
    meta_scaled = scaler.transform(meta)

    # Combine features
    X = sparse.hstack([text_vec, sparse.csr_matrix(meta_scaled)], format="csr")

    scam_score = float(clf.predict_proba(X)[0][1])

    # Decide threshold
    if manual_threshold is None:
        used_threshold = AUTO_THRESH_SHORT if meta[0, 9] < SHORT_TEXT_WORDS else AUTO_THRESH_LONG
    else:
        used_threshold = float(manual_threshold)

    label = int(scam_score >= used_threshold)

    # Explainability
    tokens = top_keywords_for_text(cleaned, vect, clf, top_k=6)

    meta_names = ["has_salary","has_phone","has_whatsapp","has_url","has_email",
                  "generic_email","wfh_flag","upfront_fee","scam_kw_count","text_len"]
    meta_dict = dict(zip(meta_names, meta[0].astype(int).tolist()))

    return jsonify({
        "label": label,
        "scam_score": scam_score,
        "used_threshold": used_threshold,
        "top_tokens": tokens,
        "meta": meta_dict
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
