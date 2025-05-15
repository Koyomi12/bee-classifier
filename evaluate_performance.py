import json
from pathlib import Path

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
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
    }
    tagged_probabilities = []
    y_true = []
    dirs = sorted(TAG_CORRECTOR_DATA.glob("*"))
    for dir in dirs:
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
        results[dir.name] = confusion_matrix
        results["total"]["true_positive"] += confusion_matrix["true_positive"]
        results["total"]["true_negative"] += confusion_matrix["true_negative"]
        results["total"]["false_positive"] += confusion_matrix["false_positive"]
        results["total"]["false_negative"] += confusion_matrix["false_negative"]

    roc_auc_score = sklearn.metrics.roc_auc_score(y_true, tagged_probabilities)
    print(f"ROC_AUC score: {roc_auc_score}")

    with open("output/classifier_results.json", "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
