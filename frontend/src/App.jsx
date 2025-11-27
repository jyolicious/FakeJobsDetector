import React, { useState } from "react";
import "./styles.css"; // make sure this file exists in src/

export default function App() {
  const [text, setText] = useState(`Congratulations! You have been shortlisted for a remote Customer Support role with guaranteed weekly pay of ₹35,000. This is an exclusive work-from-home opportunity with immediate joining. To confirm your spot you must pay a refundable security deposit of ₹1,500 and provide bank account details for verification. Once verified, training will begin and payments are instant. For faster processing click www.quick-apply-join.info or message us on WhatsApp at +91-9988776655.`);
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyzeJob() {
    if (!text.trim()) {
      alert("Please enter a job description to analyze");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    // attempt to call backend predict; if it fails, fallback to local scoring visualization
    try {
      const payload = { text };
      // If user typed a manual threshold, send it
      if (threshold !== null && threshold !== undefined) payload.threshold = parseFloat(threshold);

      const resp = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(`Server returned ${resp.status}`);
      }
      const data = await resp.json();
      // ensure numbers are numbers for display
      data.scam_score = Number(data.scam_score);
      data.used_threshold = Number(data.used_threshold ?? threshold);
      setResult(data);
    } catch (err) {
      // fallback: compute a quick client-side pseudo-score so UI still shows something useful
      console.warn("Backend call failed, using local heuristic:", err.message);
      const local = localHeuristic(text, threshold);
      setResult(local);
      setError("Backend not reachable — showing local heuristic result.");
    } finally {
      setLoading(false);
    }
  }

  // local lightweight heuristic used as fallback when backend unavailable
  function localHeuristic(text, userThreshold) {
    const scamWords = {
      "000": 0.359, "confirm": 0.220, "pay": 0.217, "deposit": 0.195,
      "bank account": 0.208, "verification": 0.156, "congratulations": 0.189,
      "guaranteed": 0.185, "refundable": 0.178, "work from home": 0.135,
      "whatsapp": 0.142, "limited": 0.162, "registration fee": 0.2, "urgent": 0.174
    };
    const lowerText = text.toLowerCase();
    const found = [];
    let total = 0;
    for (const [k, v] of Object.entries(scamWords)) {
      if (lowerText.includes(k)) {
        found.push([k, v]);
        total += v;
      }
    }
    const avg = found.length ? (total / found.length) : 0;
    const scam_score = Math.min(avg * 1.2, 1.0); // scale similar to your HTML heuristic
    const tokens = found.sort((a, b) => b[1] - a[1]).slice(0, 12);
    // metadata heuristics
    const hasEmail = /@/.test(text) ? 1 : 0;
    const hasGenericEmail = /(gmail|yahoo|hotmail|outlook)\.com/i.test(text) ? 1 : 0;
    const hasPhone = /(\+?\d{1,3}[-\s]?)?\d{10}/.test(text) ? 1 : 0;
    const hasSuspiciousURL = /(https?:\/\/[^\s]+)|(www\.[^\s]+)/i.test(text) && !/linkedin|indeed|naukri|glassdoor/i.test(text) ? 1 : 0;

    return {
      label: scam_score >= (userThreshold ?? 0.6) ? 1 : 0,
      scam_score,
      used_threshold: userThreshold ?? 0.6,
      top_tokens: tokens.map(t => [t[0], t[1]]),
      meta: {
        has_email: hasEmail,
        generic_email: hasGenericEmail,
        has_phone: hasPhone,
        suspicious_link: hasSuspiciousURL ? 1 : 0
      }
    };
  }

  function renderBadge(score, thresholdUsed) {
    if (score >= thresholdUsed) {
      return <div className="status-badge critical">CRITICAL</div>;
    } else if (score >= thresholdUsed * 0.7) {
      return <div className="status-badge alert">ALERT</div>;
    } else {
      return <div className="status-badge clear">CLEAR</div>;
    }
  }

  return (
    <div>
      <div className="bg-gradient" />
      <div className="container">
        <div className="header">
          <div className="logo-mark">S</div>
          <h1 className="main-title">SCAM DETECTOR</h1>
          <p className="subtitle">Advanced Job Fraud Analysis System</p>
        </div>

        <div className="main-content">
          <div className="form-section">
            <label className="form-label">
              <span className="label-accent"></span>
              Job Description Input
            </label>
            <textarea
              id="jobText"
              className="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className="form-section">
            <label className="form-label">
              <span className="label-accent"></span>
              Detection Threshold
            </label>
            <div className="threshold-group">
              <input
                type="number"
                className="number-input"
                id="threshold"
                value={threshold}
                step="0.01"
                min="0"
                max="1"
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <span className="hint-text">Sensitivity level for fraud detection</span>
            </div>
          </div>

          <button className="scan-button" onClick={analyzeJob} disabled={loading}>
            {loading ? "Analyzing..." : "Initiate Analysis"}
          </button>

          <div className={`results-container ${result ? "active" : ""}`} id="resultsContainer">
            {result && (
              <>
                <div className="results-header">
                  <h2 className="results-title">Analysis Results</h2>
                  {renderBadge(result.scam_score, result.used_threshold)}
                </div>

                <div className="score-display">
                  <div className="score-value" id="scoreValue">{(result.scam_score * 100).toFixed(2)}%</div>
                  <div className="score-description">Fraud Probability Score</div>
                  <div className="threshold-display">Detection Threshold: <span id="thresholdDisplay">{result.used_threshold.toFixed(2)}</span></div>
                </div>

                <div className="analysis-grid">
                  <div className="analysis-card">
                    <h3 className="card-title"><span className="card-icon" />Fraud Indicators</h3>
                    <div className="data-list" id="indicatorsList">
                      {result.top_tokens && result.top_tokens.length > 0 ? (
                        result.top_tokens.slice(0, 12).map((t, idx) => (
                          <div className="data-item" key={idx}>
                            <span className="item-name">{t[0]}</span>
                            <span className="item-value">{Number(t[1]).toFixed(3)}</span>
                          </div>
                        ))
                      ) : (
                        <div style={{ color: "#6b7280" }}>No indicators detected</div>
                      )}
                    </div>
                  </div>

                  <div className="analysis-card">
                    <h3 className="card-title"><span className="card-icon" />Metadata Analysis</h3>
                    <div className="data-list" id="metadataList">
                      <div className="data-item">
                        <span className="item-name">Email Address</span>
                        <span className={result.meta?.has_email ? "indicator-positive" : "indicator-negative"}>
                          {result.meta?.has_email ? "✓" : "✗"}
                        </span>
                      </div>
                      <div className="data-item">
                        <span className="item-name">Generic Email</span>
                        <span className={result.meta?.generic_email ? "indicator-warning" : "indicator-positive"}>
                          {result.meta?.generic_email ? "!" : "✓"}
                        </span>
                      </div>
                      <div className="data-item">
                        <span className="item-name">Phone Number</span>
                        <span className={result.meta?.has_phone ? "indicator-warning" : "indicator-positive"}>
                          {result.meta?.has_phone ? "!" : "✓"}
                        </span>
                      </div>
                      <div className="data-item">
                        <span className="item-name">Suspicious Link</span>
                        <span className={result.meta?.suspicious_link ? "indicator-negative" : "indicator-positive"}>
                          {result.meta?.suspicious_link ? "✗" : "✓"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}

            {error && <p style={{ color: "#fb7185", marginTop: 12 }}>{error}</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
