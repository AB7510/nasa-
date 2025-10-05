"""
Streamlit Exoplanet Classifier (robust fallback)

This single-file app was originally written as a Streamlit demo. In some execution
environments (like the one you ran in), the `streamlit` package may not be available
and attempting to `import streamlit as st` will raise ModuleNotFoundError.

This rewritten file fixes that by:
 - Detecting whether `streamlit` is installed.
 - If `streamlit` is available, it starts the interactive Streamlit app (same UX as
   the original demo).
 - If `streamlit` is NOT available, it falls back to a command-line interface (CLI)
   so the script can still be executed and the ML pipeline (train/evaluate/predict)
   runs correctly without Streamlit.

Why this change?
 - The error you hit was `ModuleNotFoundError: No module named 'streamlit'` which
   happens at import-time. To make the script importable in environments without
   Streamlit, the decorators and direct top-level Streamlit calls were moved into
   a function that's only executed if Streamlit is present.

How to use:
 - To run in Streamlit (interactive UI): first install Streamlit in your environment
   (e.g. `pip install streamlit`) then:
       streamlit run streamlit_exoplanet_app.py

 - To run without Streamlit (CLI mode): simply run the script with Python. For example:
       python streamlit_exoplanet_app.py --demo --train
   This will run a small demo training session and write outputs to the local folder.

Additional notes:
 - SHAP is optional; if SHAP is not installed the code will skip SHAP-related plots
   and continue.
 - A tiny test/demo runner is included and can be invoked using `--run-tests`.

"""

from __future__ import annotations
import os
import io
import argparse
import tempfile
import textwrap
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import lightgbm as lgb




# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# Streamlit detection (do not raise on missing streamlit)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

# ---------------------- Core utilities (no Streamlit dependence) ----------------------

def load_demo_dataframe(n: int = 400) -> pd.DataFrame:
    """Create a small synthetic demo dataframe. Pure pandas/numpy (no streamlit).

    Parameters
    ----------
    n : int
        Number of synthetic rows to generate.
    """
    rng = np.random.RandomState(42)
    period = np.abs(rng.normal(loc=10, scale=20, size=n)) + 0.01
    duration = np.abs(rng.normal(loc=2.5, scale=1.2, size=n)) + 0.01
    depth = np.abs(rng.normal(loc=0.01, scale=0.02, size=n))
    planet_radius = np.abs(rng.normal(loc=1.5, scale=1.0, size=n))
    snr = np.abs(rng.normal(loc=15, scale=8, size=n))
    stellar_teff = rng.normal(loc=5600, scale=400, size=n)
    stellar_logg = rng.normal(loc=4.3, scale=0.25, size=n)

    labels = []
    for d, r in zip(depth, planet_radius):
        if d > 0.03 and r > 2.0:
            labels.append("CONFIRMED")
        elif d > 0.007 and r > 0.8:
            labels.append("CANDIDATE")
        else:
            labels.append("FALSE POSITIVE")

    df = pd.DataFrame({
        "period": period,
        "duration": duration,
        "depth": depth,
        "planet_radius": planet_radius,
        "snr": snr,
        "stellar_teff": stellar_teff,
        "stellar_logg": stellar_logg,
        "status": labels,
    })
    return df


def train_lgb_model(X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None, num_boost_round: int = 300) -> lgb.Booster:
    """Train a LightGBM multiclass model and return the booster object."""
    n_classes = int(np.unique(y_train).shape[0])
    params = params or {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "learning_rate": 0.05,
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return model


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "status",
    test_size: float = 0.2,
    num_rounds: int = 300,
    random_seed: int = 42,
    output_prefix: str | None = None,
    produce_shap: bool = True,
    save_artifact: bool = True,
) -> dict:
    """Train on the dataframe and return evaluation outputs.

    Returns a dict containing: model, scaler, label_encoder, X_test, y_test, predictions, metrics, file paths
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not present in dataframe")

    # select numeric features only (ignore non-numeric automatically)
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric features available for training. Please select numeric columns.")

    y_raw = df[label_col].astype(str).copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if len(np.unique(y)) > 1 else None, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_lgb_model(X_train_scaled, y_train, num_boost_round=num_rounds)

    # predictions
    probs = model.predict(X_test_scaled)
    if probs.ndim == 1:
        preds = (probs > 0.5).astype(int)
    else:
        preds = np.argmax(probs, axis=1)

    # metrics
    n_classes = len(label_encoder.classes_)
    labels = list(range(n_classes))
    target_names = list(label_encoder.classes_)

    report = classification_report(y_test, preds, labels=labels, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)

    outputs: dict = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": preds,
        "y_prob": probs,
        "report": report,
        "confusion_matrix": cm,
        "accuracy": acc,
        "files": {},
    }

    # Save artifacts and results to files if requested
    prefix = output_prefix or "exoplanet_demo"

    # predictions on the full dataframe (use model on all numeric features we can)
    X_full = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X_full_scaled = scaler.transform(X_full)
    probs_full = model.predict(X_full_scaled)
    if probs_full.ndim == 1:
        preds_full = (probs_full > 0.5).astype(int)
        pred_labels_full = label_encoder.inverse_transform(preds_full)
        max_prob_full = probs_full
    else:
        preds_full = np.argmax(probs_full, axis=1)
        pred_labels_full = label_encoder.inverse_transform(preds_full)
        max_prob_full = np.max(probs_full, axis=1)

    out_df = df.copy().reset_index(drop=True)
    out_df["prediction"] = pred_labels_full
    out_df["pred_prob"] = max_prob_full

    pred_csv = f"{prefix}_predictions.csv"
    out_df.to_csv(pred_csv, index=False)
    outputs["files"]["predictions_csv"] = os.path.abspath(pred_csv)

    # save model artifact
    if save_artifact:
        artifact = {"model": model, "scaler": scaler, "features": list(X.columns), "label_encoder": label_encoder}
        art_path = f"{prefix}_artifact.joblib"
        joblib.dump(artifact, art_path)
        outputs["files"]["artifact_joblib"] = os.path.abspath(art_path)

    # save metrics & confusion matrix figure
    metrics_txt = f"{prefix}_metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"accuracy: {acc}\n\n")
        f.write("classification_report:\n")
        f.write(textwrap.indent(pd.DataFrame(report).to_string(), "  "))
    outputs["files"]["metrics_txt"] = os.path.abspath(metrics_txt)

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    cm_png = f"{prefix}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_png)
    plt.close(fig)
    outputs["files"]["confusion_matrix_png"] = os.path.abspath(cm_png)

    # SHAP (optional)
    if produce_shap and SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            # summary plot for first class if multiclass
            fig_shap = plt.figure(figsize=(8, 4))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test, feature_names=X.columns, show=False)
            else:
                shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
            shap_png = f"{prefix}_shap_summary.png"
            plt.savefig(shap_png, bbox_inches="tight")
            plt.close()
            outputs["files"]["shap_summary_png"] = os.path.abspath(shap_png)
        except Exception as e:
            outputs["shap_error"] = str(e)
    else:
        if produce_shap and not SHAP_AVAILABLE:
            outputs["shap_error"] = "shap not installed"

    return outputs


# ---------------------- Streamlit UI (only executed if Streamlit is installed) ----------------------

def run_streamlit_app():
    """Run the Streamlit interactive UI. This function will only be called when Streamlit is available."""
    # We intentionally keep the UI logic contained here so the module is importable
    # in environments without Streamlit.
    st.set_page_config(page_title="Exoplanet Classifier", layout="wide")

    st.title("Exoplanet Classifier — Streamlit Demo (fallback-aware)")
    st.markdown(
        "Upload a CSV from Kepler/K2/TESS or use the demo dataset. Then choose features and a label column. Click **Train model** to train and evaluate a LightGBM classifier."
    )

    # Sidebar controls
    st.sidebar.title("Exoplanet Classifier - Controls")
    use_demo = st.sidebar.checkbox("Use demo dataset", value=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if use_demo or (uploaded_file is None):
        df = load_demo_dataframe()
        st.sidebar.text("Demo dataset loaded")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.text(f"Loaded {uploaded_file.name} — {df.shape[0]} rows")
        except Exception:
            st.sidebar.error("Could not read the CSV file. Make sure it's a valid CSV.")
            st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Feature & Label selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    label_col = st.sidebar.selectbox(
        "Label column (target)", options=all_cols, index=all_cols.index("status") if "status" in all_cols else 0
    )
    default_features = numeric_cols if len(numeric_cols) > 0 else [c for c in all_cols if c != label_col]
    selected_features = st.sidebar.multiselect("Feature columns", options=all_cols, default=default_features)

    st.sidebar.markdown("---")
    train_test_split_ratio = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
    num_rounds = st.sidebar.slider("LightGBM boosting rounds", 50, 2000, 300, 50)
    random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

    st.sidebar.markdown("---")
    run_training = st.sidebar.button("Train model")

    with st.expander("Preview data", expanded=True):
        st.dataframe(df.head())
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    if label_col not in df.columns:
        st.error("Selected label column not in the dataframe")
        st.stop()

    if len(selected_features) == 0:
        st.error("Please select at least one feature column")
        st.stop()

    X_raw = df[selected_features].copy()
    X = X_raw.select_dtypes(include=[np.number]).copy()
    if X.shape[1] != len(selected_features):
        missing = set(selected_features) - set(X.columns.tolist())
        st.warning(f"The following selected features are non-numeric and will be ignored: {missing}")

    y_raw = df[label_col].astype(str).copy()
    label_encoder = LabelEncoder()
    try:
        y = label_encoder.fit_transform(y_raw)
    except Exception:
        st.error("Error encoding label column — ensure it is categorical/text")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_ratio, stratify=y if len(np.unique(y)) > 1 else None, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if run_training:
        st.info("Training LightGBM model — this may take a moment")
        model = train_lgb_model(X_train_scaled, y_train, num_boost_round=num_rounds)
        st.success("Model trained")
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["features"] = X.columns.tolist()
        st.session_state["label_encoder"] = label_encoder
    else:
        if "model" not in st.session_state:
            st.info("Model not yet trained — use the controls on the left and click 'Train model'")

    if "model" in st.session_state:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        features_used = st.session_state["features"]
        label_encoder = st.session_state["label_encoder"]

        y_pred_probs = model.predict(X_test_scaled)
        if y_pred_probs.ndim == 1:
            y_pred = (y_pred_probs > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)

        st.header("Evaluation on test set")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")

        st.subheader("Classification report")
        n_classes = len(label_encoder.classes_)
        report = classification_report(y_test, y_pred, labels=list(range(n_classes)), target_names=list(label_encoder.classes_), output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion matrix")
        tick_marks = np.arange(len(label_encoder.classes_))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(label_encoder.classes_)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

        st.header("SHAP explanations")
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
                st.subheader("Global feature importance (summary plot)")
                fig_shap = plt.figure(figsize=(8, 4))
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[0], X_test, feature_names=X.columns, show=False)
                else:
                    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
                st.pyplot(fig_shap)

                st.subheader("Per-row explanation")
                row_idx = st.number_input("Select test-row index for explanation (0-based)", min_value=0, max_value=X_test.shape[0] - 1, value=0, step=1)
                if isinstance(shap_values, list):
                    single_shap = np.sum([np.abs(sv[row_idx]) for sv in shap_values], axis=0)
                    contrib = pd.Series(single_shap, index=X.columns)
                else:
                    single_shap = shap_values[row_idx]
                    contrib = pd.Series(single_shap, index=X.columns)
                contrib = contrib.abs().sort_values(ascending=False)
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                contrib.head(10).plot(kind="bar", ax=ax2)
                ax2.set_title("Top absolute SHAP contributions (test row)")
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
        else:
            st.info("SHAP not available — install shap to enable feature explanations")

        st.header("Run inference on uploaded dataset")
        run_infer = st.button("Run inference & show predictions")
        if run_infer:
            X_full = df[features_used].select_dtypes(include=[np.number]).fillna(0)
            X_full_scaled = scaler.transform(X_full)
            probs_full = model.predict(X_full_scaled)
            if probs_full.ndim == 1:
                preds_full = (probs_full > 0.5).astype(int)
                pred_labels = label_encoder.inverse_transform(preds_full)
                max_prob = probs_full
            else:
                preds_full = np.argmax(probs_full, axis=1)
                pred_labels = label_encoder.inverse_transform(preds_full)
                max_prob = np.max(probs_full, axis=1)
            out_df = df.copy()
            out_df["prediction"] = pred_labels
            out_df["pred_prob"] = max_prob
            st.write(out_df.head(50))
            towrite = io.StringIO()
            out_df.to_csv(towrite, index=False)
            st.download_button(label="Download predictions CSV", data=towrite.getvalue(), file_name="predictions.csv", mime="text/csv")

        if st.button("Save model artifact to disk (joblib)"):
            artifact = {"model": model, "scaler": scaler, "features": features_used, "label_encoder": label_encoder}
            joblib.dump(artifact, "exoplanet_model_artifact.joblib")
            st.success("Saved exoplanet_model_artifact.joblib")


# ---------------------- CLI fallback (runs when Streamlit isn't available) ----------------------

def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exoplanet classifier (CLI fallback)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--demo", action="store_true", help="Use the built-in demo dataset")
    group.add_argument("--input-csv", type=str, help="Path to input CSV file to load")
    parser.add_argument("--label-col", type=str, default="status", help="Name of the label/target column")
    parser.add_argument("--features", type=str, nargs="*", default=None, help="List of feature column names (defaults to all numeric columns)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default="exoplanet_demo")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP computation even if shap is installed")
    parser.add_argument("--no-save", action="store_true", help="Do not save artifact (joblib)")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in smoke tests and exit")

    args = parser.parse_args(argv)

    if args.run_tests:
        print("Running built-in tests...\n")
        return run_tests()

    if args.demo:
        df = load_demo_dataframe()
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        print("No input specified. Use --demo or --input-csv. Use -h for help.")
        return 2

    # decide feature columns
    if args.features and len(args.features) > 0:
        feature_cols = args.features
    else:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Using {len(feature_cols)} numeric feature columns: {feature_cols}\n")

    outputs = train_and_evaluate(
        df,
        feature_cols=feature_cols,
        label_col=args.label_col,
        test_size=args.test_size,
        num_rounds=args.num_rounds,
        random_seed=args.seed,
        output_prefix=args.output_prefix,
        produce_shap=(not args.no_shap),
        save_artifact=(not args.no_save),
    )

    print("Training & evaluation complete. Files written:")
    for k, v in outputs.get("files", {}).items():
        print(f" - {k}: {v}")
    if "shap_error" in outputs:
        print("SHAP warning:", outputs["shap_error"])

    return 0


# ---------------------- Basic tests ----------------------

def run_tests() -> int:
    """Simple smoke tests: ensure demo dataset generates and training runs (fast)."""
    try:
        df = load_demo_dataframe(n=200)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outputs = train_and_evaluate(
            df,
            feature_cols=feature_cols,
            label_col="status",
            test_size=0.25,
            num_rounds=50,
            random_seed=0,
            output_prefix=tempfile.mktemp(prefix="exo_test_"),
            produce_shap=False,
            save_artifact=False,
        )
        print("Smoke test passed. Generated files:")
        for k, v in outputs.get("files", {}).items():
            print(f" - {k}: {v}")
        return 0
    except Exception as e:
        print("Smoke test failed:", e)
        return 1


# ---------------------- Entry point ----------------------

def main(argv: list[str] | None = None) -> int:
    if STREAMLIT_AVAILABLE:
        # if Streamlit is available run the interactive app
        run_streamlit_app()
        return 0
    else:
        # fallback CLI mode
        return run_cli(argv)


def new_func(main):
    raise SystemExit(main())

if __name__ == "__main__":
    new_func(main)






