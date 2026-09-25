from pathlib import Path
import sys

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from ml_tests.utils.loader import load_metadata_from_folder
from ml_tests.utils.data_processing import group_by_samples_and_period
from ml_tests.core import get_butch_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
DERIVED_ROOT = RESEARCH_TOOL_ROOT / "Data" / "cache" / "forest" / "derived"
PATH_TO_IMAGES_DATA = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Data" / "Images"

sys.path.insert(0, str(SCRIPTS_ROOT))

def evaluate_linear_probe(X, labels, name, n_splits=5, n_repeats=20):
    X = np.asarray(X)
    labels = np.asarray(labels)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
        )),
    ])

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    scores = cross_validate(
        model,
        X,
        labels,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1_macro",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )

    print(f"\n=== {name} ===")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
    ]:
        values = scores[f"test_{metric}"]

        print(
            f"{metric:20s}: "
            f"{values.mean():.3f} +/- {values.std():.3f}"
        )

    return scores

def print_linear_probe_results(results):
    for name, scores in results.items():
        print(f"\n=== {name} ===")
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc",
        ]:
            values = scores[f"test_{metric}"]
            print(
                f"{metric:20s}: "
                f"{values.mean():.3f} +/- {values.std():.3f}"
            )


PATH_TO_IMAGES_DATA = r"Scripts\ResearchScripts\Data\Images"

images_info = load_metadata_from_folder(PATH_TO_IMAGES_DATA)
grouped_data = group_by_samples_and_period(images_info)
X_pre, labels, X_post, _, statistic_info = get_butch_embeddings(grouped_data)

X_pre = np.array(X_pre)
X_post = np.array(X_post)
X_delta = X_post - X_pre

X_pre_post = np.concatenate(
    [X_pre, X_post],
    axis=1
)
# shape: (N, 1024)

X_post_delta = np.concatenate(
    [X_post, X_delta],
    axis=1
)
# shape: (N, 1024)

X_pre_post_delta = np.concatenate(
    [X_pre, X_post, X_delta],
    axis=1
)
# shape: (N, 1536)

results = {}

results["PRE"] = evaluate_linear_probe(
    X_pre,
    labels,
    "PRE"
)

results["POST"] = evaluate_linear_probe(
    X_post,
    labels,
    "POST"
)

results["DELTA"] = evaluate_linear_probe(
    X_delta,
    labels,
    "DELTA"
)

results["PRE + POST"] = evaluate_linear_probe(
    X_pre_post,
    labels,
    "PRE + POST"
)

results["POST + DELTA"] = evaluate_linear_probe(
    X_post_delta,
    labels,
    "POST + DELTA"
)

results["PRE + POST + DELTA"] = evaluate_linear_probe(
    X_pre_post_delta,
    labels,
    "PRE + POST + DELTA"
)

print_linear_probe_results(results)