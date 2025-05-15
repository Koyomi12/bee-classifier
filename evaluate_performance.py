import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

TAG_CORRECTOR_DATA = Path("/mnt/trove/local_storage/processed-bee-data")


def main():
    """
    Generate a JSON file with daily and total confusion matrices to evaluate
    the model's performance over the entire dataset.
    """
    results = {
        "total": {
            "confusion_matrix": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
            }
        }
    }
    tagged_probabilities_total = []
    y_true_total = []
    y_pred_total = []
    dirs = sorted(TAG_CORRECTOR_DATA.glob("*"))
    for dir in dirs:
        tagged_probabilities = []
        y_true = []
        data = pd.read_csv(
            dir / "data.csv",
            dtype={
                "day_dance_id": "string",
                "waggle_id": "string",
                "category": "Int64",
                "category_label": "string",
                "confidence": "Float64",
                "corrected_category": "Int64",
                "corrected_category_label": "string",
                "dance_type": "string",
                "corrected_dance_type": "string",
            },
            na_filter=False,
        )

        true_positives = data.loc[
            (data["category_label"] == "tagged")
            & (data["corrected_category_label"] == "")
        ]
        for _, row in true_positives.iterrows():
            tagged_probabilities.append(row["confidence"])
            y_true.append(1)

        true_negatives = data.loc[
            (data["category_label"] == "untagged")
            & (data["corrected_category_label"] == "")
        ]
        for _, row in true_negatives.iterrows():
            tagged_probabilities.append(1 - row["confidence"])
            y_true.append(0)

        false_positives = data.loc[
            (data["category_label"] == "tagged")
            & (data["corrected_category_label"] == "untagged")
        ]
        for _, row in false_positives.iterrows():
            tagged_probabilities.append(row["confidence"])
            y_true.append(0)

        false_negatives = data.loc[
            (data["category_label"] == "untagged")
            & (data["corrected_category_label"] == "tagged")
        ]
        for _, row in false_negatives.iterrows():
            tagged_probabilities.append(1 - row["confidence"])
            y_true.append(1)

        confusion_matrix = {
            "true_positive": true_positives.shape[0],
            "true_negative": true_negatives.shape[0],
            "false_positive": false_positives.shape[0],
            "false_negative": false_negatives.shape[0],
        }

        tp = confusion_matrix["true_positive"]
        fp = confusion_matrix["false_positive"]
        fn = confusion_matrix["false_negative"]
        tn = confusion_matrix["true_negative"]

        y_pred = tp * [1] + fp * [1] + fn * [0] + tn * [0]

        precision = sklearn.metrics.precision_score(
            y_true, y_pred, zero_division=np.nan
        )
        recall = sklearn.metrics.recall_score(y_true, y_pred, zero_division=np.nan)
        f1_score = sklearn.metrics.f1_score(y_true, y_pred, zero_division=np.nan)
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true, tagged_probabilities)

        results[dir.name] = dict(
            confusion_matrix=confusion_matrix,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc_score=roc_auc_score,
        )

        results["total"]["confusion_matrix"]["true_positive"] += tp
        results["total"]["confusion_matrix"]["true_negative"] += tn
        results["total"]["confusion_matrix"]["false_positive"] += fp
        results["total"]["confusion_matrix"]["false_negative"] += fn

        y_pred_total += y_pred
        y_true_total += y_true
        tagged_probabilities_total += tagged_probabilities

    precision_total = sklearn.metrics.precision_score(
        y_true_total, y_pred_total, zero_division=np.nan
    )
    recall_total = sklearn.metrics.recall_score(
        y_true_total, y_pred_total, zero_division=np.nan
    )
    f1_score_total = sklearn.metrics.f1_score(
        y_true_total, y_pred_total, zero_division=np.nan
    )
    roc_auc_score_total = sklearn.metrics.roc_auc_score(
        y_true_total, tagged_probabilities_total
    )
    results["total"]["precision"] = precision_total
    results["total"]["recall"] = recall_total
    results["total"]["f1_score"] = f1_score_total
    results["total"]["roc_auc_score"] = roc_auc_score_total

    with open("output/classifier_results.json", "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
