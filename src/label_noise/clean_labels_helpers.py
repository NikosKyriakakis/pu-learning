from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
from console import *
import numpy as np


def correct_label_issues(datamodule, estimator, folds=5):
    X = np.array([datamodule.vectorizer.vectorize(x) for x in datamodule.documents["text"].tolist()])
    y = np.array(datamodule.documents["pu-label"])

    pred_probs = cross_val_predict (
        estimator,
        X,
        y,
        cv=folds,
        method="predict_proba"
    )

    ranked_label_issues = find_label_issues (
        y,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(warning(f"Cleanlab found {len(ranked_label_issues)} label issues"))

    for issue_index in ranked_label_issues:
        if datamodule.documents["pu-label"].iloc[issue_index] == 1:
            # print(datamodule.documents.at[issue_index, "text"] + "\n")
            datamodule.documents.at[issue_index, "pu-label"] = 0
        elif datamodule.documents["pu-label"].iloc[issue_index] == 0:
            datamodule.documents.at[issue_index, "pu-label"] = 1