from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
from console import *
import numpy as np


def correct_label_issues(datamodule, estimator, folds=5, n_jobs=-1):
    X = np.array([datamodule.vectorizer.vectorize(x) for x in datamodule.documents["text"].tolist()])
    y = np.array(datamodule.documents["pu-label"])

    print(hourglass(f"Performing {folds}-fold cross validation to get out-of-sample prediction probabilities ..."))
    pred_probs = cross_val_predict (
        estimator,
        X,
        y,
        cv=folds,
        method="predict_proba",
        n_jobs=n_jobs
    )

    print(success("Out-of-sample prediction probabilities computed"))

    ranked_label_issues = find_label_issues (
        y,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(warning(f"Cleanlab found {len(ranked_label_issues)} potential label issues"))

    flipped = 0
    correct_flips = 0
    for issue_index in ranked_label_issues:
        if datamodule.documents.at[issue_index, "pu-label"] == 0 \
            and datamodule.documents.at[issue_index, "label"] == 1:
            correct_flips += 1

        if datamodule.documents["pu-label"].iloc[issue_index] == 0:
            datamodule.documents.at[issue_index, "pu-label"] = 1

            flipped += 1

    print(success(f"Correct labels flipped {correct_flips / flipped * 100} %"))

    
