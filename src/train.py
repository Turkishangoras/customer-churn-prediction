from preprocess import load_data

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def train_model():
    # 1. Load data
    df = load_data("data/Telco_customer_churn.csv")

    # 2. Split features and target
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    # 3. Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # 4. Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # 5. Model (WITH class balancing 🔥)
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Train model
    model_pipeline.fit(X_train, y_train)

    # 8. Predictions
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # 9. Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # 10. Feature Importance
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model_pipeline.named_steps["model"].feature_importances_

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(feat_imp.head(10))

    #11. Feature Importance Graph
    # Get top 10 features (sorted nicely)
    top_features = feat_imp.head(10).sort_values()

    # Create dark-themed figure
    plt.figure(figsize=(10, 6), facecolor="#0b1020")

    # Plot bars
    bars = plt.barh(
        top_features.index,
        top_features.values,
        edgecolor="white",
        linewidth=0.8
    )

    # Set axis background
    ax = plt.gca()
    ax.set_facecolor("#0b1020")

    # Title and labels (light text)
    plt.title(
        "Top 10 Features Driving Customer Churn",
        fontsize=15,
        fontweight="bold",
        color="white",
        pad=15
    )

    plt.xlabel("Importance Score", fontsize=11, color="white")
    plt.ylabel("Features", fontsize=11, color="white")

    # Style ticks (make them visible on dark background)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Add gridlines
    plt.grid(axis="x", linestyle="--", alpha=0.3, color="gray")

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="white"
        )

    # Layout and save
    plt.tight_layout()
    plt.savefig(
        "outputs/figures/feature_importance.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="#0b1020"
    )

    plt.show()
    plt.close()

    # 12. Save model
    joblib.dump(model_pipeline, "outputs/models/churn_model.joblib")

    print("\nModel trained and saved successfully!")

if __name__ == "__main__":
    train_model()