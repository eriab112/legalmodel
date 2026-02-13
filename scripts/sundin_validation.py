#!/usr/bin/env python3
"""
Sundin validation: clustering check and feature importance (read-only on feature file).

Reads:
  - Data/processed/decision_features_sundin2026.json  (create via sundin_feature_extraction.py)

Writes (NEW files only):
  - Data/processed/sundin_feature_importance.json
  - Data/processed/clustering_validation_report.json

Does NOT modify cleaned_court_texts.json, labeled_dataset.json, or any pipeline data.
Safe to run after sundin_feature_extraction.py.
"""

import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "Data" / "processed" / "decision_features_sundin2026.json"
IMPORTANCE_PATH = BASE_DIR / "Data" / "processed" / "sundin_feature_importance.json"
CLUSTER_REPORT_PATH = BASE_DIR / "Data" / "processed" / "clustering_validation_report.json"

# Order of numeric/one-hot features for matrix (must match build_feature_matrix)
FEATURE_ORDER = [
    "downstream_has_screen",
    "downstream_gap_mm",
    "downstream_angle_degrees",
    "downstream_bypass_ls",
    "upstream_has_fishway",
    "upstream_type_int",
    "upstream_slope_pct",
    "upstream_discharge_ls",
    "upstream_has_eel_ramp",
    "flow_min_ls",
    "flow_hydropeaking_banned",
    "flow_percent_mq",
    "monitoring_required",
    "monitoring_functional",
    "cost_msek",
    "timeline_years",
]

UPSTREAM_TYPE_MAP = {None: 0, "nature-like": 1, "vertical-slot": 2, "eel-ramp": 3, "undefined": 4}


def build_feature_matrix(decisions: list) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build numeric matrix from decision feature dicts.
    Returns (X, feature_names, labels).
    Missing numeric -> -1; booleans -> 0/1; upstream_type -> int.
    """
    rows = []
    labels = []
    for d in decisions:
        f = d["features"]
        row = []
        row.append(1 if f.get("downstream_has_screen") else 0)
        row.append(float(f.get("downstream_gap_mm")) if f.get("downstream_gap_mm") is not None else -1.0)
        row.append(float(f.get("downstream_angle_degrees")) if f.get("downstream_angle_degrees") is not None else -1.0)
        row.append(float(f.get("downstream_bypass_ls")) if f.get("downstream_bypass_ls") is not None else -1.0)
        row.append(1 if f.get("upstream_has_fishway") else 0)
        row.append(UPSTREAM_TYPE_MAP.get(f.get("upstream_type"), 0))
        row.append(float(f.get("upstream_slope_pct")) if f.get("upstream_slope_pct") is not None else -1.0)
        row.append(float(f.get("upstream_discharge_ls")) if f.get("upstream_discharge_ls") is not None else -1.0)
        row.append(1 if f.get("upstream_has_eel_ramp") else 0)
        row.append(float(f.get("flow_min_ls")) if f.get("flow_min_ls") is not None else -1.0)
        row.append(1 if f.get("flow_hydropeaking_banned") else 0)
        row.append(float(f.get("flow_percent_mq")) if f.get("flow_percent_mq") is not None else -1.0)
        row.append(1 if f.get("monitoring_required") else 0)
        row.append(1 if f.get("monitoring_functional") else 0)
        row.append(float(f.get("cost_msek")) if f.get("cost_msek") is not None else -1.0)
        row.append(float(f.get("timeline_years")) if f.get("timeline_years") is not None else -1.0)
        rows.append(row)
        labels.append(d["label"])
    return np.array(rows, dtype=np.float64), FEATURE_ORDER.copy(), labels


def run_clustering(X: np.ndarray, labels: list[str], ids: list[str]) -> dict:
    """KMeans(3) + ARI; report as dict for saving."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    ari = adjusted_rand_score(labels, cluster_labels)

    # Heuristic: map cluster to risk by mean cost (higher cost -> higher risk)
    cost_col = FEATURE_ORDER.index("cost_msek")
    cluster_costs = []
    for c in range(3):
        mask = cluster_labels == c
        vals = X[mask, cost_col]
        vals = vals[vals >= 0]
        cluster_costs.append((c, float(np.mean(vals)) if len(vals) else 0))
    cluster_costs.sort(key=lambda x: x[1], reverse=True)
    cluster_to_risk = {cluster_costs[0][0]: "HIGH_RISK", cluster_costs[1][0]: "MEDIUM_RISK", cluster_costs[2][0]: "LOW_RISK"}

    disagreements = []
    for i, (cid, lid) in enumerate(zip(cluster_labels, labels)):
        if cluster_to_risk[cid] != lid:
            disagreements.append({"id": ids[i], "label": lid, "cluster_risk": cluster_to_risk[cid]})

    return {
        "ari": round(ari, 4),
        "n_samples": len(labels),
        "cluster_to_risk_mapping": cluster_to_risk,
        "disagreements": disagreements,
        "message": "Low ARI suggests labels may not align with feature clusters; review disagreements manually.",
    }


def run_rf_importance(X: np.ndarray, labels: list[str]) -> dict:
    """5-fold stratified RF, aggregate feature importances."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    label_map = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
    y = np.array([label_map[l] for l in labels])
    n = len(FEATURE_ORDER)
    importances = np.zeros(n)
    fold_importances = []

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        rf = RandomForestClassifier(max_depth=4, min_samples_leaf=3, random_state=42 + fold, n_estimators=50)
        rf.fit(X[train_idx], y[train_idx])
        fold_importances.append(rf.feature_importances_.tolist())
        importances += rf.feature_importances_

    importances /= 5
    std = np.std(fold_importances, axis=0).tolist() if len(fold_importances) > 1 else [0.0] * n
    return {
        "feature_importance": dict(zip(FEATURE_ORDER, [round(x, 4) for x in importances.tolist()])),
        "feature_importance_std": dict(zip(FEATURE_ORDER, [round(s, 4) for s in std])),
        "note": f"Averaged over 5-fold stratified CV; n_samples={len(labels)}. Interpret as correlation with risk in this set.",
    }


def main():
    if not FEATURES_PATH.exists():
        print(f"Missing {FEATURES_PATH}. Run: python scripts/sundin_feature_extraction.py")
        return

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    decisions = data.get("decisions", [])
    if not decisions:
        print("No decisions in feature file.")
        return

    X, feat_names, labels = build_feature_matrix(decisions)
    ids = [d["id"] for d in decisions]

    print("--- Clustering validation ---")
    cluster_report = run_clustering(X, labels, ids)
    print(f"ARI (labels vs 3 clusters): {cluster_report['ari']}")
    print(f"Disagreements: {len(cluster_report['disagreements'])}")
    for d in cluster_report["disagreements"][:10]:
        print(f"  {d['id']}: label={d['label']}, cluster suggests {d['cluster_risk']}")
    if len(cluster_report["disagreements"]) > 10:
        print(f"  ... and {len(cluster_report['disagreements']) - 10} more")

    print("\n--- Feature importance (5-fold RF) ---")
    importance_report = run_rf_importance(X, labels)
    for name, imp in sorted(importance_report["feature_importance"].items(), key=lambda x: -x[1]):
        std = importance_report["feature_importance_std"].get(name, 0)
        print(f"  {name}: {imp:.3f} ± {std:.3f}")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_report, f, ensure_ascii=False, indent=2)
    with open(IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(importance_report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {CLUSTER_REPORT_PATH}")
    print(f"Wrote {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
