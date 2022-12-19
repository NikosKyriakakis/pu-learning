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

    print(warning(f"Cleanlab found {len(ranked_label_issues)} potential label issues"))

    doc_list = datamodule.documents.values.tolist()
    for i, issue_index in enumerate(ranked_label_issues):
        if doc_list[issue_index][3] == 0:
            print(doc_list[issue_index])
        # print(f"Index = {i}\n")

    for issue_index in ranked_label_issues:
        # if datamodule.documents["pu-label"].iloc[issue_index] == 1:
        #     datamodule.documents.at[issue_index, "pu-label"] = 0
        if datamodule.documents["pu-label"].iloc[issue_index] == 0:
            datamodule.documents.at[issue_index, "pu-label"] = 1

    
