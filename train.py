"""
train.py
--------
Mall Customer Segmentation - ML Training Pipeline
 Hackathon Project

Runs K-Means clustering on Mall_Customers.csv and saves:
  - models/kmeans_model.pkl   (trained KMeans)
  - models/scaler.pkl         (fitted StandardScaler)
  - models/segment_map.pkl    (cluster → segment name mapping)
  - models/cluster_stats.pkl  (per-cluster summary stats)
  - data/segmented.csv        (original data + Segment column)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  MALL CUSTOMER SEGMENTATION — TRAINING PIPELINE")
print("=" * 55)

DATA_PATH   = os.path.join("data", "Mall_Customers.csv")
MODEL_DIR   = "models"
STATIC_DIR  = "static"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")
print(f"    Null values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2] Feature engineering...")

# Encode gender: Female=1, Male=0
df["Gender_Enc"] = (df["Gender"] == "Female").astype(int)

# Features for clustering
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[FEATURES].values

print(f"    Clustering features: {FEATURES}")
print(f"    Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# 3. SCALING
# ─────────────────────────────────────────────
print("\n[3] Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"    Scaled mean ≈ {X_scaled.mean():.4f}, std ≈ {X_scaled.std():.4f}")

# ─────────────────────────────────────────────
# 4. ELBOW + SILHOUETTE — FIND OPTIMAL K
# ─────────────────────────────────────────────
print("\n[4] Running elbow + silhouette analysis (k=2 to 10)...")
inertias, silhouettes = [], []
K_RANGE = range(2, 11)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"    k={k}  inertia={km.inertia_:.1f}  silhouette={silhouette_score(X_scaled, labels):.4f}")

best_k = K_RANGE[silhouettes.index(max(silhouettes))]
print(f"\n    Best k by silhouette: k={best_k}")

# Save elbow chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_RANGE), inertias, "o-", color="#378ADD", linewidth=2, markersize=7)
ax1.axvline(x=5, color="#E24B4A", linestyle="--", alpha=0.6, label="k=5 (chosen)")
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("Inertia (WCSS)", fontsize=11)
ax1.set_title("Elbow Curve", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(list(K_RANGE), silhouettes, "o-", color="#1D9E75", linewidth=2, markersize=7)
ax2.axvline(x=5, color="#E24B4A", linestyle="--", alpha=0.6, label="k=5 (chosen)")
ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=11)
ax2.set_title("Silhouette Score", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "elbow_curve.png"), dpi=130, bbox_inches="tight")
plt.close()
print("    Elbow chart saved → static/elbow_curve.png")

# ─────────────────────────────────────────────
# 5. FINAL K-MEANS WITH k=5
# ─────────────────────────────────────────────
BEST_K = 5
print(f"\n[5] Fitting final K-Means with k={BEST_K}...")
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"    Final silhouette score: {final_sil:.4f}")
print(f"    Cluster sizes: {df['Cluster'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 6. NAME SEGMENTS FROM CENTROIDS
# ─────────────────────────────────────────────
print("\n[6] Naming segments from cluster centroids...")

# Get original-scale centroids
centroids_scaled = kmeans.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)

# ── Exact RFM segment names from case study document ──
# Mapping based on centroid (Income, Score) analysis:
#   High Income + High Score  → VIP Customers    (High R, High F, High M)
#   Mid  Income + Mid  Score  → Loyal Customers  (Moderate R, High F)
#   Low  Income + High Score  → Discount Seekers (Low M, High F during sales)
#   Low  Income + Low  Score  → At-Risk / Churn  (Low R, Moderate F)
#   High Income + Low  Score  → New Customers    (High R, Low F/M — untapped)

SEGMENT_COLORS = {
    "VIP Customers":    "#1D9E75",
    "Loyal Customers":  "#378ADD",
    "Discount Seekers": "#E24B4A",
    "At-Risk / Churn":  "#BA7517",
    "New Customers":    "#7F77DD",
}

SEGMENT_STRATEGIES = {
    "VIP Customers":    "Exclusive loyalty rewards and early access to new products.",
    "Loyal Customers":  "Personalized product recommendations to maintain engagement.",
    "Discount Seekers": "Targeted coupon campaigns and seasonal promotion alerts.",
    "At-Risk / Churn":  "Re-engagement \"win-back\" emails and special \"we miss you\" offers.",
    "New Customers":    "Welcome discounts and onboarding series to trigger a second purchase.",
}

SEGMENT_ACTIONS = {
    "VIP Customers":    "Retain & upsell — exclusive events, early product access",
    "Loyal Customers":  "Increase basket size with personalised cross-sell & upsell",
    "Discount Seekers": "Maximise frequency with flash sales and seasonal alerts",
    "At-Risk / Churn":  "Win-back campaign — \"we miss you\" offer before permanent loss",
    "New Customers":    "Onboard with welcome discount, trigger second purchase",
}

SEGMENT_BEHAVIOR = {
    "VIP Customers":    "High R, High F, High M",
    "Loyal Customers":  "Moderate R, High F",
    "Discount Seekers": "Low M, High F (during sales)",
    "At-Risk / Churn":  "Low R, Moderate F",
    "New Customers":    "High R, Low F/M",
}

# Direct cluster → RFM segment mapping (validated against centroids)
segment_map = {}
for k, (inc, sc) in enumerate(centroids_orig):
    if   inc >= 70 and sc >= 60:   segment_map[k] = "VIP Customers"
    elif inc >= 40 and inc < 70:   segment_map[k] = "Loyal Customers"
    elif inc <  40 and sc >= 55:   segment_map[k] = "Discount Seekers"
    elif inc <  40 and sc <  55:   segment_map[k] = "At-Risk / Churn"
    elif inc >= 70 and sc <  60:   segment_map[k] = "New Customers"
    else:                          segment_map[k] = "Loyal Customers"

df["Segment"] = df["Cluster"].map(segment_map)

print("    Segment mapping:")
for k, name in segment_map.items():
    inc, sc = centroids_orig[k]
    n = (df["Cluster"] == k).sum()
    print(f"    Cluster {k} → {name:<22} | Income=${inc:.0f}k  Score={sc:.0f}  n={n}  Behavior: {SEGMENT_BEHAVIOR[name]}")

# ─────────────────────────────────────────────
# 6b. RFM PROXY MAPPING (case study alignment)
# ─────────────────────────────────────────────
print("\n[6b] Adding RFM proxy columns...")

# R proxy — Spending Score: high score = recently active buyer
df['R_proxy'] = df['Spending Score (1-100)'].apply(
    lambda x: 'High' if x >= 60 else ('Moderate' if x >= 35 else 'Low'))

# F proxy — Age: younger customers tend to shop more frequently
df['F_proxy'] = df['Age'].apply(
    lambda x: 'High' if x <= 30 else ('Moderate' if x <= 45 else 'Low'))

# M proxy — Annual Income: higher income = higher monetary capacity
df['M_proxy'] = df['Annual Income (k$)'].apply(
    lambda x: 'High' if x >= 70 else ('Moderate' if x >= 45 else 'Low'))

# Segment IS the RFM name — already using case study names
df['RFM_Segment'] = df['Segment']

# RFM behaviour + marketing per segment — direct from case study doc
RFM_NAME_MAP  = {s: s for s in SEGMENT_BEHAVIOR.keys()}   # identity map
RFM_BEHAVIOR  = SEGMENT_BEHAVIOR
RFM_MARKETING = SEGMENT_STRATEGIES

# Business impact definitions (from case study doc)
BUSINESS_IMPACTS = [
    {
        'title':  'Increased Efficiency',
        'icon':   'efficiency',
        'color':  '#1D9E75',
        'desc':   'Drastically reduces marketing waste by excluding segments that don\'t respond to specific offers.',
        'metric': '57% waste eliminated',
        'detail': 'Without segmentation, 57% of campaign budget reaches the wrong audience. Targeted campaigns ensure every offer reaches only receptive customers, multiplying ROI 3–5×.',
    },
    {
        'title':  'Retention Growth',
        'icon':   'retention',
        'color':  '#378ADD',
        'desc':   'Early detection of "At-Risk" behavior allows for proactive intervention before a customer churns.',
        'metric': f'{len(df[df["Segment"] == "At-Risk / Churn"])} at-risk customers identified',
        'detail': f'{len(df[df["Segment"] == "At-Risk / Churn"])} customers show Low R, Moderate F — disengaged but recoverable. Proactive win-back campaigns can retain these customers before permanent loss.',
    },
    {
        'title':  'Scalability',
        'icon':   'scalability',
        'color':  '#7F77DD',
        'desc':   'The data-driven pipeline handles large-scale transaction growth, providing real-time insights into shifting customer behaviors.',
        'metric': '200 → millions of customers',
        'detail': 'K-Means scoring is a single matrix multiplication — handles millions of customers in milliseconds. Segment labels update automatically as behaviour shifts.',
    },
]

# Save RFM artifacts
joblib.dump(RFM_NAME_MAP,     os.path.join(MODEL_DIR, "rfm_name_map.pkl"))
joblib.dump(RFM_BEHAVIOR,     os.path.join(MODEL_DIR, "rfm_behavior.pkl"))
joblib.dump(RFM_MARKETING,    os.path.join(MODEL_DIR, "rfm_marketing.pkl"))
joblib.dump(BUSINESS_IMPACTS, os.path.join(MODEL_DIR, "business_impacts.pkl"))
df.to_csv(os.path.join("data", "segmented.csv"), index=False)

print(f"    RFM segments: {df['Segment'].value_counts().to_dict()}")
print("    models/rfm_name_map.pkl     ✓")
print("    models/rfm_behavior.pkl     ✓")
print("    models/rfm_marketing.pkl    ✓")
print("    models/business_impacts.pkl ✓")

# ─────────────────────────────────────────────
# 7. CLUSTER STATS
# ─────────────────────────────────────────────
print("\n[7] Computing cluster statistics...")
cluster_stats = []
for k in range(BEST_K):
    sub = df[df["Cluster"] == k]
    inc_orig, sc_orig = centroids_orig[k]
    name = segment_map[k]
    rfm_name = name   # Segment IS the RFM name now
    cluster_stats.append({
        "cluster":      k,
        "name":         name,
        "rfm_name":     rfm_name,
        "rfm_behavior": SEGMENT_BEHAVIOR.get(name, ""),
        "rfm_strategy": SEGMENT_STRATEGIES.get(name, ""),
        "color":        SEGMENT_COLORS[name],
        "strategy":     SEGMENT_STRATEGIES[name],
        "action":       SEGMENT_ACTIONS[name],
        "n":            len(sub),
        "pct":          round(len(sub) / len(df) * 100, 1),
        "avg_income":   round(sub["Annual Income (k$)"].mean(), 1),
        "avg_score":    round(sub["Spending Score (1-100)"].mean(), 1),
        "avg_age":      round(sub["Age"].mean(), 1),
        "female_pct":   round((sub["Gender"] == "Female").mean() * 100, 1),
        "cx":           round(inc_orig, 1),
        "cy":           round(sc_orig, 1),
    })
    print(f"    {name:<22} n={len(sub):>3}  Income=${sub['Annual Income (k$)'].mean():.0f}k  Score={sub['Spending Score (1-100)'].mean():.0f}  Age={sub['Age'].mean():.0f}")

# ─────────────────────────────────────────────
# 8. GENERATE CHARTS
# ─────────────────────────────────────────────
print("\n[8] Generating visualisation charts...")

# -- Scatter plot --
fig, ax = plt.subplots(figsize=(10, 7))
for stat in cluster_stats:
    sub = df[df["Cluster"] == stat["cluster"]]
    ax.scatter(
        sub["Annual Income (k$)"], sub["Spending Score (1-100)"],
        c=stat["color"], label=stat["name"], s=80, alpha=0.80, edgecolors="white", linewidth=0.5
    )
    ax.scatter(stat["cx"], stat["cy"], c=stat["color"],
               s=250, marker="*", edgecolors="black", linewidth=1.2, zorder=5)

ax.set_xlabel("Annual Income (k$)", fontsize=12)
ax.set_ylabel("Spending Score (1–100)", fontsize=12)
ax.set_title("Mall Customer Segments\n(K-Means, k=5) — ★ = Centroid", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "scatter.png"), dpi=130, bbox_inches="tight")
plt.close()
print("    Scatter plot saved → static/scatter.png")

# -- Bar charts --
names  = [s["name"]       for s in cluster_stats]
colors = [s["color"]      for s in cluster_stats]
incomes = [s["avg_income"] for s in cluster_stats]
scores  = [s["avg_score"]  for s in cluster_stats]
ages    = [s["avg_age"]    for s in cluster_stats]
counts  = [s["n"]          for s in cluster_stats]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

# Customer count
axes[0].bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)
axes[0].set_title("Customers per Segment", fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
axes[0].tick_params(axis="x", labelrotation=15, labelsize=9)
axes[0].grid(axis="y", alpha=0.3)

# Avg income
axes[1].bar(names, incomes, color=colors, edgecolor="white", linewidth=0.8)
axes[1].set_title("Avg Annual Income (k$)", fontweight="bold")
axes[1].set_ylabel("Income (k$)")
for i, v in enumerate(incomes):
    axes[1].text(i, v + 0.5, f"${v}k", ha="center", fontsize=10, fontweight="bold")
axes[1].tick_params(axis="x", labelrotation=15, labelsize=9)
axes[1].grid(axis="y", alpha=0.3)

# Avg spending score
axes[2].bar(names, scores, color=colors, edgecolor="white", linewidth=0.8)
axes[2].set_title("Avg Spending Score", fontweight="bold")
axes[2].set_ylabel("Score (1–100)")
for i, v in enumerate(scores):
    axes[2].text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
axes[2].tick_params(axis="x", labelrotation=15, labelsize=9)
axes[2].grid(axis="y", alpha=0.3)

# Avg age
axes[3].bar(names, ages, color=colors, edgecolor="white", linewidth=0.8)
axes[3].set_title("Avg Customer Age", fontweight="bold")
axes[3].set_ylabel("Age (years)")
for i, v in enumerate(ages):
    axes[3].text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")
axes[3].tick_params(axis="x", labelrotation=15, labelsize=9)
axes[3].grid(axis="y", alpha=0.3)

plt.suptitle("Segment Analysis — Mall Customers Dataset", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "bar_charts.png"), dpi=130, bbox_inches="tight")
plt.close()
print("    Bar charts saved → static/bar_charts.png")

# -- Gender pie --
fig, axes = plt.subplots(1, BEST_K, figsize=(16, 4))
for i, stat in enumerate(cluster_stats):
    sub = df[df["Cluster"] == stat["cluster"]]
    female = (sub["Gender"] == "Female").sum()
    male   = len(sub) - female
    axes[i].pie([female, male], labels=["Female", "Male"],
                colors=[stat["color"], stat["color"] + "55"],
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[i].set_title(stat["name"], fontsize=10, fontweight="bold")

plt.suptitle("Gender Distribution per Segment", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "gender_pie.png"), dpi=130, bbox_inches="tight")
plt.close()
print("    Gender pie chart saved → static/gender_pie.png")

# ─────────────────────────────────────────────
# 9. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
print("\n[9] Saving model artifacts...")
joblib.dump(kmeans,       os.path.join(MODEL_DIR, "kmeans_model.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(segment_map,  os.path.join(MODEL_DIR, "segment_map.pkl"))
joblib.dump(cluster_stats,os.path.join(MODEL_DIR, "cluster_stats.pkl"))
df.to_csv(os.path.join("data", "segmented.csv"), index=False)

print("    models/kmeans_model.pkl  ✓")
print("    models/scaler.pkl        ✓")
print("    models/segment_map.pkl   ✓")
print("    models/cluster_stats.pkl ✓")
print("    data/segmented.csv       ✓")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE — Run app.py to start dashboard")
print("=" * 55 + "\n")
