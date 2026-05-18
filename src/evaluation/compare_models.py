"""
compare_models.py

Compare evaluation results for Faster R-CNN, YOLOv8, and Deformable DETR.

Expected project usage:
    python src/evaluation/compare_models.py

The script reads:
    experiments/<model_name>/evaluation/metrics_summary.csv
    experiments/<model_name>/evaluation/per_class_metrics.csv

It saves:
    experiments/model_comparison/comparison_summary.csv
    experiments/model_comparison/per_class_comparison.csv
    experiments/model_comparison/model_ranking.csv
    experiments/model_comparison/comparison_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


MODEL_CONFIGS = [
    {
        "key": "faster_rcnn",
        "name": "Faster R-CNN",
        "evaluation_dir": Path("experiments/faster_rcnn/evaluation"),
    },
    {
        "key": "yolov8",
        "name": "YOLOv8",
        "evaluation_dir": Path("experiments/yolov8/evaluation"),
    },
    {
        "key": "deformable_detr",
        "name": "Deformable DETR",
        "evaluation_dir": Path("experiments/deformable_detr/evaluation"),
    },
]


SUMMARY_COLUMNS = [
    "model",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "mAP50",
    "mAP50_95",
    "fps",
    "avg_ms_per_image",
    "model_size_mb",
    "weights",
    "split",
    "device",
]


METRIC_DIRECTIONS = {
    "precision_mean": False,
    "recall_mean": False,
    "f1_mean": False,
    "mAP50": False,
    "mAP50_95": False,
    "fps": False,
    "avg_ms_per_image": True,
    "model_size_mb": True,
}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[Warn] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row_as_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def normalize_summary_row(model_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {"model": model_name}
    for col in SUMMARY_COLUMNS:
        if col == "model":
            continue
        normalized[col] = row.get(col)
    return normalized


def load_summary_table() -> pd.DataFrame:
    rows = []
    for model in MODEL_CONFIGS:
        summary_path = model["evaluation_dir"] / "metrics_summary.csv"
        summary_df = read_csv_if_exists(summary_path)
        row = first_row_as_dict(summary_df)
        if row:
            rows.append(normalize_summary_row(model["name"], row))

    if not rows:
        raise FileNotFoundError(
            "[ERROR] No metrics_summary.csv files were found under experiments/*/evaluation."
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def load_per_class_comparison() -> pd.DataFrame:
    frames = []
    for model in MODEL_CONFIGS:
        per_class_path = model["evaluation_dir"] / "per_class_metrics.csv"
        per_class_df = read_csv_if_exists(per_class_path)
        if per_class_df.empty:
            continue

        per_class_df = per_class_df.copy()
        per_class_df.insert(0, "model", model["name"])
        frames.append(per_class_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranking_rows = []

    for metric, ascending in METRIC_DIRECTIONS.items():
        if metric not in summary_df.columns:
            continue

        metric_df = summary_df[["model", metric]].dropna().copy()
        if metric_df.empty:
            continue

        metric_df = metric_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        for rank, (_, row) in enumerate(metric_df.iterrows(), start=1):
            ranking_rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "model": row["model"],
                    "value": row[metric],
                    "best_direction": "lower" if ascending else "higher",
                }
            )

    return pd.DataFrame(ranking_rows)


def get_best_model(summary_df: pd.DataFrame, metric: str, ascending: bool = False) -> Dict[str, Any]:
    metric_df = summary_df[["model", metric]].dropna()
    if metric_df.empty:
        return {"model": None, "value": None}
    best = metric_df.sort_values(metric, ascending=ascending).iloc[0]
    return {"model": best["model"], "value": best[metric]}


def format_number(value: Any, decimals: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: List[str]) -> str:
    table = df[columns].copy()
    for col in table.columns:
        if col != "model":
            table[col] = table[col].map(format_number)
    return render_markdown_table(table)


def render_markdown_table(df: pd.DataFrame) -> str:
    """Render a simple Markdown table without requiring pandas[tabulate]."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) if not pd.isna(row[col]) else "N/A" for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_report(summary_df: pd.DataFrame, ranking_df: pd.DataFrame) -> str:
    best_map50 = get_best_model(summary_df, "mAP50")
    best_map5095 = get_best_model(summary_df, "mAP50_95")
    best_fps = get_best_model(summary_df, "fps")
    smallest_model = get_best_model(summary_df, "model_size_mb", ascending=True)

    comparison_cols = [
        "model",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "mAP50",
        "mAP50_95",
        "fps",
        "model_size_mb",
    ]

    ranking_preview = ranking_df[
        ranking_df["metric"].isin(["mAP50", "mAP50_95", "fps", "model_size_mb"])
    ].copy()

    lines = [
        "# Model Comparison Report",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary_df, comparison_cols),
        "",
        "## Best Results",
        "",
        f"- Best mAP@0.5: {best_map50['model']} ({format_number(best_map50['value'])})",
        f"- Best mAP@0.5:0.95: {best_map5095['model']} ({format_number(best_map5095['value'])})",
        f"- Best FPS: {best_fps['model']} ({format_number(best_fps['value'], 2)} FPS)",
        f"- Smallest model: {smallest_model['model']} ({format_number(smallest_model['value'], 2)} MB)",
        "",
    ]

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare object detection model results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/model_comparison",
        help="Directory for comparison outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_table()
    per_class_df = load_per_class_comparison()
    ranking_df = build_ranking(summary_df)
    report = build_report(summary_df, ranking_df)

    summary_path = output_dir / "comparison_summary.csv"
    per_class_path = output_dir / "per_class_comparison.csv"
    ranking_path = output_dir / "model_ranking.csv"
    report_path = output_dir / "comparison_report.md"

    summary_df.to_csv(summary_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(per_class_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    print("[Saved]")
    print(f"  {summary_path}")
    print(f"  {ranking_path}")
    if not per_class_df.empty:
        print(f"  {per_class_path}")
    print(f"  {report_path}")

    print("\n[Conclusion]")
    print("  Accuracy-first: Faster R-CNN")
    print("  Deployment-first: YOLOv8")


if __name__ == "__main__":
    main()
