"""
app.py
------
Mall Customer Segmentation — Flask Web Dashboard
Mukundan Hackathon Project

Routes:
  GET  /              → Home dashboard (charts + stats)
  GET  /customers     → Full customer table with segment filter
  GET  /segments      → Segment profiles with strategies
  POST /predict       → Predict segment for a new customer
  GET  /api/scatter   → JSON data for interactive scatter plot
  GET  /api/stats     → JSON summary statistics
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ─────────────────────────────────────────────
# INIT APP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS (once at startup)
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "segmented.csv")

kmeans        = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
segment_map   = joblib.load(os.path.join(MODEL_DIR, "segment_map.pkl"))
cluster_stats = joblib.load(os.path.join(MODEL_DIR, "cluster_stats.pkl"))
rfm_name_map  = joblib.load(os.path.join(MODEL_DIR, "rfm_name_map.pkl"))
rfm_behavior  = joblib.load(os.path.join(MODEL_DIR, "rfm_behavior.pkl"))
rfm_marketing = joblib.load(os.path.join(MODEL_DIR, "rfm_marketing.pkl"))
biz_impacts   = joblib.load(os.path.join(MODEL_DIR, "business_impacts.pkl"))
df            = pd.read_csv(DATA_PATH)

SEGMENT_COLORS = {s["name"]: s["color"] for s in cluster_stats}
SEGMENT_STRATEGIES = {s["name"]: s["strategy"] for s in cluster_stats}
SEGMENT_ACTIONS    = {s["name"]: s["action"]   for s in cluster_stats}
FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[FEATURE_COLS].values
X_scaled = scaler.transform(X)
pred_labels = kmeans.predict(X_scaled)

# Model quality + efficiency metrics (computed once on startup)
def _benchmark_inference(batch, runs=40):
    start = time.perf_counter()
    for _ in range(runs):
        kmeans.predict(batch)
    duration = time.perf_counter() - start
    avg = duration / runs
    return {
        "pred_ms": round(avg * 1000, 3),
        "pred_per_sec": int(len(batch) / avg) if avg else 0,
    }

_bench = _benchmark_inference(X_scaled)
MODEL_METRICS = {
    "silhouette": round(float(silhouette_score(X_scaled, pred_labels)), 4),
    "davies": round(float(davies_bouldin_score(X_scaled, pred_labels)), 3),
    "calinski": round(float(calinski_harabasz_score(X_scaled, pred_labels)), 1),
    "inertia": round(float(kmeans.inertia_), 1),
    "n_iter": int(getattr(kmeans, "n_iter_", 0)),
    "dataset_rows": len(df),
    "dataset_cols": len(df.columns),
    "dataset_mem_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
    "missing_values": int(df.isnull().sum().sum()),
    "cluster_balance": [
        {
            "name": s["name"],
            "count": int(s["n"]),
            "pct": round(s["n"] / len(df) * 100, 1),
            "color": s["color"],
        }
        for s in cluster_stats
    ],
    **_bench,
}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard — overview metrics + charts."""
    total = len(df)
    avg_income = round(df["Annual Income (k$)"].mean(), 1)
    avg_score  = round(df["Spending Score (1-100)"].mean(), 1)
    avg_age    = round(df["Age"].mean(), 1)
    female_pct = round((df["Gender"] == "Female").mean() * 100, 1)

    return render_template(
        "index.html",
        total=total,
        avg_income=avg_income,
        avg_score=avg_score,
        avg_age=avg_age,
        female_pct=female_pct,
        cluster_stats=cluster_stats,
    )


@app.route("/customers")
def customers():
    """Full customer table with optional segment filter."""
    seg_filter = request.args.get("segment", "All")
    search_id  = request.args.get("search", "").strip()

    data = df.copy()
    if seg_filter != "All":
        data = data[data["Segment"] == seg_filter]
    if search_id:
        try:
            data = data[data["CustomerID"] == int(search_id)]
        except ValueError:
            pass

    records = data[["CustomerID", "Gender", "Age",
                    "Annual Income (k$)", "Spending Score (1-100)",
                    "Segment"]].to_dict(orient="records")

    segments_list = ["All"] + sorted(df["Segment"].unique().tolist())
    return render_template(
        "customers.html",
        records=records,
        segments_list=segments_list,
        active_segment=seg_filter,
        segment_colors=SEGMENT_COLORS,
        segment_actions=SEGMENT_ACTIONS,
        total=len(records),
    )


@app.route("/segments")
def segments():
    """Segment profiles with stats and marketing strategies."""
    return render_template(
        "segments.html",
        cluster_stats=cluster_stats,
        segment_colors=SEGMENT_COLORS,
    )


@app.route("/metrics")
def metrics():
    """Model accuracy/efficiency dashboard."""
    return render_template(
        "metrics.html",
        metrics=MODEL_METRICS,
        cluster_stats=cluster_stats,
        segment_colors=SEGMENT_COLORS,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Predict segment for a new customer input."""
    result = None
    error  = None

    if request.method == "POST":
        try:
            age    = int(request.form["age"])
            gender = request.form["gender"]
            income = float(request.form["income"])
            score  = float(request.form["score"])

            # Validate
            if not (1 <= age <= 100):
                raise ValueError("Age must be between 1 and 100.")
            if not (0 <= income <= 300):
                raise ValueError("Income must be between 0 and 300 k$.")
            if not (1 <= score <= 100):
                raise ValueError("Score must be between 1 and 100.")

            X_new     = np.array([[income, score]])
            X_scaled  = scaler.transform(X_new)
            cluster   = int(kmeans.predict(X_scaled)[0])
            seg_name  = segment_map[cluster]

            # Distance to centroid (confidence proxy)
            centroid  = scaler.inverse_transform([kmeans.cluster_centers_[cluster]])[0]
            dist      = float(np.sqrt((income - centroid[0])**2 + (score - centroid[1])**2))
            confidence = max(0, min(100, round(100 - dist * 2, 1)))

            # Similar customers in same cluster
            similar = df[df["Cluster"] == cluster].head(5)[
                ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
            ].to_dict(orient="records")

            result = {
                "segment":    seg_name,
                "color":      SEGMENT_COLORS[seg_name],
                "strategy":   SEGMENT_STRATEGIES[seg_name],
                "action":     SEGMENT_ACTIONS[seg_name],
                "confidence": confidence,
                "cluster":    cluster,
                "similar":    similar,
                "input": {"age": age, "gender": gender,
                          "income": income, "score": score},
            }

        except (ValueError, KeyError) as e:
            error = str(e)

    return render_template(
        "predict.html",
        result=result,
        error=error,
        segment_colors=SEGMENT_COLORS,
        cluster_stats=cluster_stats,
    )


@app.route("/rfm")
def rfm():
    """RFM Metrics page — explains R, F, M and maps segments to case study criteria."""
    rfm_stats = []
    for s in cluster_stats:
        sub = df[df["Segment"] == s["name"]]
        high_r = round((sub["Spending Score (1-100)"] >= 60).mean() * 100, 1)
        high_f = round((sub["Age"] <= 30).mean() * 100, 1)
        high_m = round((sub["Annual Income (k$)"] >= 70).mean() * 100, 1)
        rfm_stats.append({**s, "high_r_pct": high_r, "high_f_pct": high_f, "high_m_pct": high_m})
    return render_template(
        "rfm.html",
        rfm_stats=rfm_stats,
        rfm_behavior=rfm_behavior,
        rfm_marketing=rfm_marketing,
        cluster_stats=cluster_stats,
    )


@app.route("/impact")
def impact():
    """Business Impact page — Efficiency, Retention Growth, Scalability."""
    at_risk_count = len(df[df["Segment"] == "At-Risk / Churn"])
    vip_count     = len(df[df["Segment"] == "VIP Customers"])
    total         = len(df)
    vip_pct       = round(vip_count / total * 100, 1)

    # Segment revenue proxy (income × score as simple proxy)
    df["rev_proxy"] = df["Annual Income (k$)"] * df["Spending Score (1-100)"] / 100
    total_rev = df["rev_proxy"].sum()

    seg_rev = {}
    for s in cluster_stats:
        sub = df[df["Segment"] == s["name"]]
        seg_rev[s["name"]] = round(sub["rev_proxy"].sum() / total_rev * 100, 1)

    return render_template(
        "impact.html",
        biz_impacts=biz_impacts,
        cluster_stats=cluster_stats,
        at_risk_count=at_risk_count,
        vip_count=vip_count,
        vip_pct=vip_pct,
        seg_rev=seg_rev,
        total=total,
    )


# ─────────────────────────────────────────────
# API ENDPOINTS (JSON for JS charts)
# ─────────────────────────────────────────────

@app.route("/api/scatter")
def api_scatter():
    """Return all customers as JSON for Chart.js scatter."""
    records = df[["CustomerID", "Gender", "Age",
                  "Annual Income (k$)", "Spending Score (1-100)",
                  "Cluster", "Segment"]].copy()
    records["color"] = records["Segment"].map(SEGMENT_COLORS)
    return jsonify(records.to_dict(orient="records"))


@app.route("/api/stats")
def api_stats():
    """Return summary stats JSON."""
    return jsonify({
        "total":      len(df),
        "segments":   cluster_stats,
        "avg_income": round(df["Annual Income (k$)"].mean(), 1),
        "avg_score":  round(df["Spending Score (1-100)"].mean(), 1),
        "avg_age":    round(df["Age"].mean(), 1),
        "female_pct": round((df["Gender"] == "Female").mean() * 100, 1),
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Mall Customer Segmentation — Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
