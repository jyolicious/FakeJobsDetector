import React, { useState } from "react";

function ResultCard({ result }) {
  if (!result) return null;

  const meta = result.meta || {};

  return (
    <div className="mt-6 p-5 rounded-lg bg-white shadow">
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-semibold">Result</h3>
        <span className={`px-3 py-1 text-sm rounded ${result.label ? "bg-red-200 text-red-800" : "bg-green-200 text-green-800"}`}>
          {result.label ? "SCAM" : "REAL"}
        </span>
      </div>

      <div className="mt-4">
        <p className="text-gray-600 text-sm">Scam Score:</p>
        <p className="text-3xl font-bold">{(result.scam_score * 100).toFixed(2)}%</p>
        <p className="text-xs text-gray-500 mt-1">
          Threshold Used: {result.used_threshold.toFixed(2)}
        </p>
      </div>

      <div className="mt-4">
        <h4 className="font-medium text-gray-700">Top Tokens</h4>
        <ul className="mt-2 text-sm text-gray-800 space-y-1">
          {result.top_tokens.map(([token, score], idx) => (
            <li key={idx}>
              <span className="font-semibold">{token}</span> — {score.toFixed(3)}
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-4">
        <h4 className="font-medium text-gray-700">Metadata Flags</h4>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-6 text-sm mt-2">
          {Object.entries(meta).map(([k, v]) => (
            <li key={k}>
              <span className="font-semibold">{k.replace(/_/g, " ")}:</span>{" "}
              {typeof v === "number" ? v : v ? "Yes" : "No"}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [threshold, setThreshold] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = { text };
    if (threshold) payload.threshold = parseFloat(threshold);

    try {
      const resp = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) throw new Error("Server error");

      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    }

    setLoading(false);
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex justify-center">
      <div className="w-full max-w-4xl">
        <h1 className="text-2xl font-bold mb-4">Fake Job / Internship Detector</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            rows="8"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste job or internship text here..."
            className="w-full p-4 border rounded-md bg-white shadow-sm"
            required
          />

          <div>
            <label className="block text-gray-700 font-medium text-sm mb-1">
              Manual Threshold (optional)
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              placeholder="Leave empty for auto-threshold"
              className="border rounded-md p-2 w-40 bg-white shadow-sm"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-5 py-2 bg-indigo-600 text-white rounded-md shadow hover:bg-indigo-700"
          >
            {loading ? "Checking..." : "Check"}
          </button>
        </form>

        {error && <p className="text-red-600 mt-4">{error}</p>}

        <ResultCard result={result} />
      </div>
    </div>
  );
}
