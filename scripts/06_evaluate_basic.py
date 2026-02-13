#!/usr/bin/env python3
"""
Task 3: Evaluate Fine-tuned KB-BERT Results

Reads training_metrics.json from 05_finetune_legalbert.py and generates:
1. Summary table (per-fold accuracy, F1, precision, recall)
2. Normalized confusion matrix visualization
3. Per-class precision/recall/F1 breakdown
4. Error analysis (misclassified documents)
5. Overfitting assessment (train loss vs val accuracy)

Outputs:
- models/nap_legalbert_cv/evaluation_report.json
- models/nap_legalbert_cv/performance_summary.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "models" / "nap_legalbert_cv" / "training_metrics.json"
OUTPUT_DIR = BASE_DIR / "models" / "nap_legalbert_cv"

LABEL_NAMES = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]


def load_metrics():
    """Load training metrics from JSON."""
    if not METRICS_PATH.exists():
        print(f"ERROR: {METRICS_PATH} not found. Run 05_finetune_legalbert.py first.")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary_table(metrics):
    """Generate per-fold and aggregate summary table."""
    lines = []
    lines.append("# NAP LegalBERT - Performance Summary")
    lines.append("")
    lines.append(f"**Model**: {metrics['model']}")
    lines.append(f"**Documents**: {metrics['n_documents']} ({metrics['label_distribution']})")
    lines.append(f"**Chunks**: {metrics['n_total_chunks']} (window={metrics['chunk_size']}, stride={metrics['stride']})")
    lines.append(f"**Folds**: {metrics['n_folds']}-fold stratified CV")
    lines.append(f"**Device**: {metrics['device']}")
    lines.append(f"**Training time**: {metrics['total_time_seconds']:.0f}s ({metrics['total_time_seconds']/60:.1f} min)")
    lines.append("")

    # Hyperparameters
    hp = metrics["hyperparameters"]
    lines.append("## Hyperparameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    for k, v in hp.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Class weights
    lines.append("## Class Weights")
    lines.append("")
    for cls, w in metrics["class_weights"].items():
        lines.append(f"- {cls}: {w:.3f}")
    lines.append("")

    # Per-fold results
    lines.append("## Per-Fold Results (Document-Level)")
    lines.append("")
    lines.append("| Fold | Accuracy | F1 (macro) | Precision | Recall | Train Loss | Val Docs | Val Chunks |")
    lines.append("|------|----------|------------|-----------|--------|------------|----------|------------|")

    for r in metrics["per_fold"]:
        lines.append(
            f"| {r['fold']} | {r['doc_accuracy']:.3f} | {r['doc_f1_macro']:.3f} | "
            f"{r['doc_precision_macro']:.3f} | {r['doc_recall_macro']:.3f} | "
            f"{r['train_loss']:.4f} | {r['n_val_docs']} | {r['n_val_chunks']} |"
        )

    agg = metrics["aggregate"]
    lines.append(
        f"| **Mean** | **{agg['doc_accuracy_mean']:.3f}** | **{agg['doc_f1_macro_mean']:.3f}** | "
        f"**{agg['doc_precision_macro_mean']:.3f}** | **{agg['doc_recall_macro_mean']:.3f}** | "
        f"**{agg['train_loss_mean']:.4f}** | - | - |"
    )
    lines.append(
        f"| **Std** | {agg['doc_accuracy_std']:.3f} | {agg['doc_f1_macro_std']:.3f} | "
        f"{agg['doc_precision_macro_std']:.3f} | {agg['doc_recall_macro_std']:.3f} | "
        f"{agg['train_loss_std']:.4f} | - | - |"
    )
    lines.append("")

    # Chunk-level comparison
    lines.append("## Chunk vs Document Level Comparison")
    lines.append("")
    lines.append("| Fold | Chunk Acc | Chunk F1 | Doc Acc | Doc F1 |")
    lines.append("|------|-----------|----------|---------|--------|")
    for r in metrics["per_fold"]:
        lines.append(
            f"| {r['fold']} | {r['chunk_accuracy']:.3f} | {r['chunk_f1_macro']:.3f} | "
            f"{r['doc_accuracy']:.3f} | {r['doc_f1_macro']:.3f} |"
        )
    lines.append("")

    return "\n".join(lines)


def generate_per_class_breakdown(metrics):
    """Generate per-class precision/recall/F1 across folds."""
    lines = []
    lines.append("## Per-Class Performance (Averaged Across Folds)")
    lines.append("")

    # Collect per-class metrics from classification reports
    class_metrics = {cls: {"precision": [], "recall": [], "f1-score": [], "support": []}
                     for cls in LABEL_NAMES}

    for r in metrics["per_fold"]:
        report = r["doc_classification_report"]
        for cls in LABEL_NAMES:
            if cls in report:
                for metric_name in ["precision", "recall", "f1-score", "support"]:
                    class_metrics[cls][metric_name].append(report[cls][metric_name])

    lines.append("| Class | Precision | Recall | F1-Score | Support (avg) |")
    lines.append("|-------|-----------|--------|----------|---------------|")

    for cls in LABEL_NAMES:
        m = class_metrics[cls]
        if m["precision"]:
            lines.append(
                f"| {cls} | {np.mean(m['precision']):.3f} +/- {np.std(m['precision']):.3f} | "
                f"{np.mean(m['recall']):.3f} +/- {np.std(m['recall']):.3f} | "
                f"{np.mean(m['f1-score']):.3f} +/- {np.std(m['f1-score']):.3f} | "
                f"{np.mean(m['support']):.1f} |"
            )
    lines.append("")

    return "\n".join(lines), class_metrics


def generate_error_analysis(metrics):
    """Identify consistently misclassified documents."""
    lines = []
    lines.append("## Error Analysis")
    lines.append("")

    # Track per-document predictions across folds
    doc_results = defaultdict(list)
    for r in metrics["per_fold"]:
        for doc_id, pred_info in r["doc_predictions"].items():
            doc_results[doc_id].append({
                "fold": r["fold"],
                "pred": pred_info["pred"],
                "true": pred_info["true"],
                "correct": pred_info["pred"] == pred_info["true"],
            })

    # Find documents that are misclassified in any fold
    misclassified = {}
    for doc_id, results in doc_results.items():
        n_wrong = sum(1 for r in results if not r["correct"])
        if n_wrong > 0:
            misclassified[doc_id] = {
                "true_label": results[0]["true"],
                "times_wrong": n_wrong,
                "times_seen": len(results),
                "predictions": [(r["fold"], r["pred"]) for r in results if not r["correct"]],
            }

    if misclassified:
        # Sort by number of times wrong
        sorted_mis = sorted(misclassified.items(), key=lambda x: -x[1]["times_wrong"])

        lines.append(f"**{len(sorted_mis)} documents misclassified in at least one fold:**")
        lines.append("")
        lines.append("| Document | True Label | Times Wrong | Wrong Predictions |")
        lines.append("|----------|-----------|-------------|-------------------|")

        for doc_id, info in sorted_mis:
            wrong_preds = ", ".join(f"F{fold}:{pred}" for fold, pred in info["predictions"])
            lines.append(
                f"| {doc_id} | {info['true_label']} | "
                f"{info['times_wrong']}/{info['times_seen']} | {wrong_preds} |"
            )
    else:
        lines.append("All documents correctly classified in all folds.")

    lines.append("")

    # Confusion patterns
    lines.append("### Common Confusion Patterns")
    lines.append("")
    confusion_counts = defaultdict(int)
    for doc_id, info in misclassified.items():
        for _, pred in info["predictions"]:
            confusion_counts[(info["true_label"], pred)] += 1

    if confusion_counts:
        lines.append("| True -> Predicted | Count |")
        lines.append("|-------------------|-------|")
        for (true, pred), count in sorted(confusion_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {true} -> {pred} | {count} |")
    lines.append("")

    return "\n".join(lines), misclassified


def generate_overfitting_assessment(metrics):
    """Assess overfitting by comparing train loss and val accuracy."""
    lines = []
    lines.append("## Overfitting Assessment")
    lines.append("")

    agg = metrics["aggregate"]
    mean_acc = agg["doc_accuracy_mean"]
    mean_loss = agg["train_loss_mean"]

    # Heuristic: if accuracy is low but loss is very low, likely overfitting
    lines.append(f"- Mean train loss: {mean_loss:.4f}")
    lines.append(f"- Mean val accuracy: {mean_acc:.3f}")
    lines.append(f"- Mean val F1 macro: {agg['doc_f1_macro_mean']:.3f}")
    lines.append("")

    # Check variance across folds
    acc_std = agg["doc_accuracy_std"]
    if acc_std > 0.15:
        lines.append("**WARNING**: High variance across folds (std > 0.15). "
                     "Results may be unstable due to small dataset size.")
    elif acc_std > 0.10:
        lines.append("**NOTE**: Moderate variance across folds. "
                     "Consider this when interpreting results.")
    else:
        lines.append("Fold variance is acceptable (std <= 0.10).")

    lines.append("")

    # Random baseline comparison
    random_acc = 1.0 / 3.0
    if mean_acc > random_acc:
        improvement = (mean_acc - random_acc) / random_acc * 100
        lines.append(f"Model accuracy ({mean_acc:.3f}) is {improvement:.1f}% above "
                     f"random baseline ({random_acc:.3f}).")
    else:
        lines.append(f"**WARNING**: Model accuracy ({mean_acc:.3f}) is at or below "
                     f"random baseline ({random_acc:.3f}).")

    # Majority class baseline
    label_dist = metrics["label_distribution"]
    majority_count = max(label_dist.values())
    total = sum(label_dist.values())
    majority_acc = majority_count / total
    majority_class = max(label_dist, key=label_dist.get)

    if mean_acc > majority_acc:
        lines.append(f"Model accuracy ({mean_acc:.3f}) exceeds majority-class baseline "
                     f"({majority_acc:.3f}, always predict {majority_class}).")
    else:
        lines.append(f"**WARNING**: Model accuracy ({mean_acc:.3f}) does not exceed "
                     f"majority-class baseline ({majority_acc:.3f}, {majority_class}).")

    lines.append("")
    return "\n".join(lines)


def plot_fold_comparison(metrics, output_path):
    """Plot bar chart comparing fold performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    folds = [r["fold"] for r in metrics["per_fold"]]
    doc_accs = [r["doc_accuracy"] for r in metrics["per_fold"]]
    doc_f1s = [r["doc_f1_macro"] for r in metrics["per_fold"]]
    train_losses = [r["train_loss"] for r in metrics["per_fold"]]

    # Accuracy and F1 per fold
    x = np.arange(len(folds))
    width = 0.35
    axes[0].bar(x - width/2, doc_accs, width, label="Accuracy", color="steelblue")
    axes[0].bar(x + width/2, doc_f1s, width, label="F1 (macro)", color="darkorange")
    axes[0].axhline(y=1/3, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Document-Level Performance per Fold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Train loss per fold
    axes[1].bar(folds, train_losses, color="steelblue")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss per Fold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved fold comparison plot to {output_path}")


def main():
    print("=" * 60)
    print("Task 3: Evaluate Fine-tuned KB-BERT Results")
    print("=" * 60)

    # Load metrics
    metrics = load_metrics()
    print(f"Loaded metrics for {metrics['n_folds']}-fold CV on {metrics['n_documents']} documents")

    # Generate all sections
    summary = generate_summary_table(metrics)
    per_class_text, class_metrics = generate_per_class_breakdown(metrics)
    error_text, misclassified = generate_error_analysis(metrics)
    overfit_text = generate_overfitting_assessment(metrics)

    # Combine into markdown report
    full_report = "\n".join([summary, per_class_text, error_text, overfit_text])

    # Save markdown report
    md_path = OUTPUT_DIR / "performance_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Saved performance summary to {md_path}")

    # Save evaluation report JSON
    eval_report = {
        "aggregate": metrics["aggregate"],
        "per_class_metrics": {
            cls: {
                metric: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": [float(v) for v in vals],
                }
                for metric, vals in class_data.items()
                if vals
            }
            for cls, class_data in class_metrics.items()
        },
        "misclassified_documents": {
            doc_id: info for doc_id, info in misclassified.items()
        },
        "confusion_patterns": {},
        "model_vs_baselines": {
            "model_accuracy": metrics["aggregate"]["doc_accuracy_mean"],
            "random_baseline": 1.0 / 3.0,
            "majority_baseline": max(metrics["label_distribution"].values()) / sum(metrics["label_distribution"].values()),
        },
    }

    eval_path = OUTPUT_DIR / "evaluation_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation report to {eval_path}")

    # Generate plots
    plot_path = OUTPUT_DIR / "fold_comparison.png"
    plot_fold_comparison(metrics, plot_path)

    # Print summary to console
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    agg = metrics["aggregate"]
    print(f"  Doc Accuracy: {agg['doc_accuracy_mean']:.3f} +/- {agg['doc_accuracy_std']:.3f}")
    print(f"  Doc F1 macro: {agg['doc_f1_macro_mean']:.3f} +/- {agg['doc_f1_macro_std']:.3f}")
    print(f"  Random baseline: {1/3:.3f}")
    print(f"  Majority baseline: {max(metrics['label_distribution'].values()) / sum(metrics['label_distribution'].values()):.3f}")
    print(f"  Misclassified docs: {len(misclassified)}/{metrics['n_documents']}")
    print()

    # Print the full markdown report
    print(full_report)


if __name__ == "__main__":
    main()
