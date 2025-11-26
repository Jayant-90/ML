import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILENAME = "roses_LIBS_50000.csv"
MODEL_FILENAME = "model.pkl"


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find data file at {csv_path}. "
            "Place your 'roses_LIBS_50000.csv' in the project folder."
        )
    df = pd.read_csv(csv_path)
    # Drop any rows with missing values to mirror the notebook
    df = df.dropna()
    return df


def build_pipeline() -> Pipeline:
    """
    Base pipeline: StandardScaler -> SVC (rbf kernel).
    Hyperparameters will be tuned via GridSearchCV.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=False)),
        ]
    )
    return pipeline


def train_and_evaluate(df: pd.DataFrame):
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in data for target labels.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_pipeline = build_pipeline()

    # Hyperparameter grid for SVC
    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.001],
    }

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    print("Best params from GridSearchCV:", grid.best_params_)
    print("Best cross-val accuracy:", grid.best_score_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report (test set):\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix (test set):\n", confusion_matrix(y_test, y_pred))

    return best_model


def save_model(model, output_path: pathlib.Path):
    joblib.dump(model, output_path)
    print(f"Saved trained model to: {output_path}")


def main():
    project_root = pathlib.Path(__file__).parent
    data_path = project_root / DATA_FILENAME
    model_path = project_root / MODEL_FILENAME

    df = load_data(data_path)
    model = train_and_evaluate(df)
    save_model(model, model_path)


if __name__ == "__main__":
    main()


