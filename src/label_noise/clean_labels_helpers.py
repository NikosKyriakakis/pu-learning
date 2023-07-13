import pandas as pd

from cleanlab.filter import find_label_issues
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from console import hourglass, success, warning


def correct_label_issues(cached_set: pd.DataFrame, selector, folds: int = 10, n_jobs: int = -1):
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    
    x = tfidfvectorizer.fit_transform(cached_set["text"])
    y = pd.Series(cached_set["pu-label"])

    print(hourglass(f"Performing {folds}-fold cross validation to get out-of-sample prediction probabilities ..."))

    cv = StratifiedKFold(n_splits=folds, shuffle=True)

    pred_probs = cross_val_predict(
        selector,
        x,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs
    )

    print(success("Out-of-sample prediction probabilities computed"))

    ranked_label_issues = find_label_issues(
        y,
        pred_probs,
        return_indices_ranked_by="self_confidence"
    )

    print(warning(f"Cleanlab found {len(ranked_label_issues)} potential label issues"))

    feature_indices = []
    for issue_index in ranked_label_issues:
        if cached_set.at[issue_index, "pu-label"] == 0:
            feature_indices.append(issue_index)

    correct_flips = 0
    for issue_index in feature_indices:
        cached_set.at[issue_index, "pu-label"] = 1
        if cached_set.at[issue_index, "label"] == 1:
            correct_flips += 1

    flipped = len(feature_indices)
    purity = 0
    try:
        purity = correct_flips / flipped * 100
        print(success(f"Positive sample: {correct_flips}/{flipped} --> Purity = {purity} %"))
    except ZeroDivisionError:
        print(success(f"Correct/Total {correct_flips}/{flipped} --> Purity = NaN"))
        purity = 0

    return purity, correct_flips, flipped
