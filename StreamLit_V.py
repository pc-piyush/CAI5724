# streamlit_digital_twin.py
"""
Streamlit app wrapping the CDC Diabetes Digital Twin framework.
Features:
 - Load & preprocess data
 - Train multiple models (RF, LR, XGBoost, LightGBM, MLP)
 - Evaluate metrics
 - Feature selection via RF importance
 - Re-train on reduced feature set
 - Patient similarity (Euclidean / Cosine / Manhattan) and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Digital Twin — CDC Diabetes")

# -------------------------
# Helper functions & caching
# -------------------------
@st.cache_data(show_spinner=False)
def load_default_data():
    # fetch dataset using ucimlrepo; fallback to direct CSV if unavailable
    try:
        repo = fetch_ucirepo(id=891)
        # repo.data.features is a DataFrame, repo.data.targets is array-like
        df = repo.data.features.copy()
        df['Diabetes_binary'] = repo.data.targets
        return df
    except Exception:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
        return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def build_preprocessor(X):
    num_cols = ['BMI','MentHlth','PhysHlth','Age']#X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # OneHotEncoder(sparse=False) to get feature names easily
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="drop")
    preproc.fit(X)
    # create feature names
    cat_feature_names = []
    if len(cat_cols) > 0:
        try:
            cat_feature_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_cols)
        except Exception:
            # older sklearn versions:
            cat_feature_names = preproc.named_transformers_['cat'].get_feature_names(cat_cols)
    feature_names = np.concatenate([num_cols, cat_feature_names])
    return preproc, feature_names, num_cols, cat_cols
def train_pipelines(X_train, y_train, preproc, random_state=42):
    # Build pipelines with preprocessor included so they accept raw DataFrames
    pipelines = {
        "RandomForest": Pipeline([("preproc", preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))]),
        "LogisticRegression": Pipeline([("preproc", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))]),
        "XGBoost": Pipeline([("preproc", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, n_jobs=-1))]),
        "LightGBM": Pipeline([("preproc", preproc), ("clf", LGBMClassifier(random_state=random_state, n_jobs=-1))]),
        "MLP": Pipeline([("preproc", preproc), ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=random_state))])
    }
    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
    return trained

def evaluate_pipelines(trained_pipes, X_test, y_test):
    metrics = {}
    prob_dict = {}
    for name, pipe in trained_pipes.items():
        y_pred = pipe.predict(X_test)
        if hasattr(pipe.named_steps['clf'], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:,1]
        else:
            y_proba = pipe.predict(X_test)
        prob_dict[name] = y_proba
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = rep["1"]["precision"]
        recall = rep["1"]["recall"]
        f1 = rep["1"]["f1-score"]
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        metrics[name] = {"Precision": precision, "Recall": recall, "F1": f1, "AUROC": auroc, "AUPRC": auprc}
    return pd.DataFrame(metrics).T, prob_dict

def feature_importances_from_rf(rf_pipe, feature_names):
    rf = rf_pipe.named_steps['clf']
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["cum_importance"] = fi_df["importance"].cumsum()
    return fi_df

def retrain_on_selected(X_train_trans, X_test_trans, selected_indices, y_train, y_test, models_dict, random_state=42):
    # X_train_trans, X_test_trans are numpy arrays (preprocessor.transform)
    X_train_sel = X_train_trans[:, selected_indices]
    X_test_sel = X_test_trans[:, selected_indices]
    reduced_trained = {}
    reduced_metrics = {}
    # Train new estimators on reduced arrays: instantiate fresh classes with same params
    for name, est in models_dict.items():
        EstClass = est.__class__
        params = est.get_params()
        # instantiate with params; if some params can't be used, fallback to no params
        try:
            new_est = EstClass(**params)
        except Exception:
            new_est = EstClass()
        new_est.fit(X_train_sel, y_train)
        reduced_trained[name] = new_est
        # predict
        if hasattr(new_est, "predict_proba"):
            y_proba = new_est.predict_proba(X_test_sel)[:,1]
            y_pred = new_est.predict(X_test_sel)
        else:
            y_proba = new_est.predict(X_test_sel)
            y_pred = new_est.predict(X_test_sel)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = rep["1"]["precision"]; recall = rep["1"]["recall"]; f1 = rep["1"]["f1-score"]
        auroc = roc_auc_score(y_test, y_proba); auprc = average_precision_score(y_test, y_proba)
        reduced_metrics[name] = {"Precision": precision, "Recall": recall, "F1": f1, "AUROC": auroc, "AUPRC": auprc}
    return reduced_trained, pd.DataFrame(reduced_metrics).T

def get_similar_indices(X_sim_space, patient_idx, metric="euclidean", top_k=5):
    if metric == "euclidean":
        dists = euclidean_distances(X_sim_space[patient_idx:patient_idx+1], X_sim_space)[0]
        maxd = dists.max() if dists.max() > 0 else 1.0
        sims = (1 - (dists / maxd)) * 100
    elif metric == "cosine":
        sims_raw = cosine_similarity(X_sim_space[patient_idx:patient_idx+1], X_sim_space)[0]
        sims = ((sims_raw - sims_raw.min()) / (sims_raw.max() - sims_raw.min() + 1e-12)) * 100
    elif metric == "manhattan":
        dists = manhattan_distances(X_sim_space[patient_idx:patient_idx+1], X_sim_space)[0]
        maxd = dists.max() if dists.max() > 0 else 1.0
        sims = (1 - (dists / maxd)) * 100
    else:
        raise ValueError("Unknown metric")
    sims[patient_idx] = -np.inf
    top_idx = np.argsort(-sims)[:top_k]
    top_sims = sims[top_idx]
    return top_idx, top_sims

def apply_intervention(series, bmi_delta=-3.0, set_phys_active=True):
    s = series.copy()
    if "BMI" in s.index:
        try:
            s["BMI"] = float(s["BMI"]) + float(bmi_delta)
        except Exception:
            pass
    if set_phys_active and "PhysActivity" in s.index:
        s["PhysActivity"] = 1
    return s

# -------------------------
# App layout: sidebar + main
# -------------------------
st.sidebar.title("Controls")
st.sidebar.markdown("Load dataset / Train models / Run similarity insights")

# use_upload = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
load_default = st.sidebar.button("Load default UCI dataset")
train_button = st.sidebar.button("Train / Retrain models")

# Feature selection threshold
fi_threshold = st.sidebar.slider("Feature importance threshold (select features > value)", min_value=0.0, max_value=0.05, value=0.02, step=0.001)

# Similarity & intervention controls
st.sidebar.markdown("---")
st.sidebar.subheader("Similarity / Twin")
patient_idx = st.sidebar.number_input("Test-set patient index", value=0, min_value=0, step=1)
sim_metric = st.sidebar.selectbox("Similarity metric", ["euclidean", "cosine", "manhattan"])
top_k = st.sidebar.slider("Top-K similar patients", 1, 20, 5)
bmi_delta = st.sidebar.number_input("Intervention: BMI delta", value=-3.0, step=0.5)
set_phys_active = st.sidebar.checkbox("Set PhysActivity = 1 (active)", value=True)
run_similarity = st.sidebar.button("Run similarity insights")
PLOT_TOP_N = 5#st.sidebar.slider("Top-K similar patients", 1, 20, 5)  # use same number as similarity list

# -------------------------
# Main content tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Models & Metrics", "Feature Importance", "Similarity Insights"])

# -------------------------
# Load dataset (either upload or default)
# -------------------------
@st.cache_data(show_spinner=False)
def get_dataframe(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return load_default_data()

df = load_default_data()#get_dataframe(use_upload)

if df is None:
    st.error("No dataset loaded.")
    st.stop()

# check target
if "Diabetes_binary" not in df.columns:
    st.error("Dataset must contain 'Diabetes_binary' column.")
    st.stop()

# Show dataset info
with tab1:
    st.write("### Dataset preview")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)
    st.write("Class balance (Diabetes_binary):")
    st.write(df["Diabetes_binary"].value_counts(normalize=True).rename("proportion"))

# ---------------
# Preprocess & split
# ---------------
X = df.drop(columns=["Diabetes_binary"]).copy()
y = df["Diabetes_binary"].copy().astype(int)

preproc, feature_names, num_cols, cat_cols = build_preprocessor(X)

# train/test split (fixed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# transform arrays for reduced training and similarity
X_train_trans = preproc.transform(X_train)
X_test_trans = preproc.transform(X_test)

# min-max scale transformed arrays for similarity space
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train_trans)
X_test_mm = mm.transform(X_test_trans)

# build simple dict of base model estimators (not pipelines), used for retraining reduced feature sets
base_models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
}

# hold trained pipelines in session_state to avoid retrain on every rerun
if "trained_pipelines" not in st.session_state:
    st.session_state.trained_pipelines = None
if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None
if "baseline_probas" not in st.session_state:
    st.session_state.baseline_probas = None
if "fi_df" not in st.session_state:
    st.session_state.fi_df = None
if "reduced_trained" not in st.session_state:
    st.session_state.reduced_trained = None
if "reduced_metrics" not in st.session_state:
    st.session_state.reduced_metrics = None
if "selected_indices" not in st.session_state:
    st.session_state.selected_indices = None

# Train models if user clicked Train or if nothing trained yet
if train_button or st.session_state.trained_pipelines is None:
    with st.spinner("Training models (this may take 10-60s depending on CPU)..."):
        trained = train_pipelines(X_train, y_train, preproc, random_state=42)
        metrics_df, prob_dict = evaluate_pipelines(trained, X_test, y_test)
        # feature importances from RandomForest pipeline
        fi_df = feature_importances_from_rf(trained["RandomForest"], feature_names)
        # store
        st.session_state.trained_pipelines = trained
        st.session_state.baseline_metrics = metrics_df
        st.session_state.baseline_probas = prob_dict
        st.session_state.fi_df = fi_df
    st.success("Training complete!")

# Display model metrics
with tab2:
    st.write("### Baseline model performance (test set)")
    if st.session_state.baseline_metrics is None:
        st.info("Train models to see metrics")
    else:
        st.dataframe(st.session_state.baseline_metrics.round(4))

    # ROC / PR curves
    if st.session_state.baseline_probas is not None:
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        for name, y_proba in st.session_state.baseline_probas.items():
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                axes[0].plot(fpr, tpr, label=f"{name} (AUROC={st.session_state.baseline_metrics.loc[name,'AUROC']:.2f})")
            except Exception:
                pass
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
                axes[1].plot(recall, precision, label=f"{name} (AUPRC={pr_auc:.2f})")
            except Exception:
                pass
        axes[0].plot([0,1],[0,1],"k--")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC Curves")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall")
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        st.pyplot(fig)

# --------------------
# Feature importance & feature selection
# --------------------
with tab3:
    st.write("### Feature importance (Random Forest)")
    if st.session_state.fi_df is None:
        st.info("Train models to compute feature importances")
    else:
        fi_show = st.session_state.fi_df.copy()
        st.dataframe(fi_show.head(10))
        # bar plot
        # plt.figure(figsize=(8,8))
        # plt.subplot(1, 2, 1) 
        plt.rcParams['figure.dpi'] = 120  # or higher = smaller appearance
        fig, ax = plt.subplots(figsize=(10,10))
        plt.tight_layout()
        sns.barplot(data=fi_show.head(10), x="importance", y="feature", ax=ax)
        ax.set_title("Top 10 features by importance")
        st.pyplot(fig)

    # Feature selection by threshold
    if st.session_state.fi_df is not None:
        threshold = fi_threshold
        selected = st.session_state.fi_df[st.session_state.fi_df["importance"] > threshold]["feature"].tolist()
        st.write(f"Selected features above threshold ({threshold}): {len(selected)} features")
        # map selected features back to indices in transformed array
        selected_indices = [i for i, f in enumerate(feature_names) if f in selected]
        st.session_state.selected_indices = selected_indices

        if len(selected_indices) == 0:
            st.warning("No features selected with this threshold. Lower the threshold.")
        else:
            # retrain reduced models (on transformed arrays)
            with st.spinner("Retraining models on reduced feature set..."):
                reduced_trained, reduced_metrics = retrain_on_selected(X_train_trans, X_test_trans, selected_indices, y_train, y_test, base_models)
                st.session_state.reduced_trained = reduced_trained
                st.session_state.reduced_metrics = reduced_metrics
            st.success("Retrained on reduced feature set.")

        # Show reduced metrics if available
        if st.session_state.reduced_metrics is not None:
            st.write("### Reduced-feature model performance")
            st.dataframe(st.session_state.reduced_metrics.round(4))

# -------------------------
# Similarity & Twin insights
# -------------------------
with tab4:
    st.write("### Similar-patient insights & digital twin simulation")
    if st.session_state.trained_pipelines is None:
        st.info("Train models first")
    else:
        # ensure patient_idx valid
        max_idx = X_test.shape[0]-1
        patient_idx = min(max(0, int(patient_idx)), max_idx)
        st.write(f"Selected test-set patient index: {patient_idx} (0..{max_idx})")

        # choose which trained pipelines to use for predictions
        trained = st.session_state.trained_pipelines
        # similarity space
        X_sim_space = X_test_mm  # min-max scaled transformed test set

        if run_similarity:
            # compute top-k similar
            top_idx, top_sims = get_similar_indices(X_sim_space, patient_idx, metric=sim_metric, top_k=top_k)
            st.write("Top similar patient indices (test-set indices) and similarity %:")
            sim_df = pd.DataFrame({"test_idx": top_idx, "similarity_pct": np.round(top_sims,3)})
            st.dataframe(sim_df)

            # baseline probs for selected patient
            patient_row = X_test.iloc[[patient_idx]]
            baseline_probs = {}
            for name, pipe in trained.items():
                if hasattr(pipe.named_steps['clf'], "predict_proba"):
                    baseline_probs[name] = float(pipe.predict_proba(patient_row)[0,1])
                else:
                    baseline_probs[name] = float(pipe.predict(patient_row)[0])
            st.write("Baseline predicted probabilities (selected patient):")
            st.write(baseline_probs)

            # for each similar patient, compute prob before/after intervention for each model
            rows = []
            for idx, sim_pct in zip(top_idx, top_sims):
                sim_row = X_test.iloc[[idx]]
                info = {"test_idx": int(idx), "similarity_pct": float(np.round(sim_pct,3))}
                for name, pipe in trained.items():
                    if hasattr(pipe.named_steps['clf'], "predict_proba"):
                        prob_before = float(pipe.predict_proba(sim_row)[0,1])
                    else:
                        prob_before = float(pipe.predict(sim_row)[0])
                    sim_after_row = apply_intervention(sim_row.iloc[0], bmi_delta=bmi_delta, set_phys_active=set_phys_active).to_frame().T
                    if hasattr(pipe.named_steps['clf'], "predict_proba"):
                        prob_after = float(pipe.predict_proba(sim_after_row)[0,1])
                    else:
                        prob_after = float(pipe.predict(sim_after_row)[0])
                    info[f"{name}_before"] = round(prob_before,4)
                    info[f"{name}_after"] = round(prob_after,4)
                    info[f"{name}_delta"] = round(prob_before - prob_after,4)
                rows.append(info)
            result_df = pd.DataFrame(rows)
            st.write("Predicted probabilities for similar patients (before -> after intervention):")
            st.dataframe(result_df)

            # average improvement per model
            avg_imp = {}
            for name in trained.keys():
                avg_imp[name] = float(result_df[f"{name}_delta"].mean())
            st.write("Average risk reduction across top-K similar patients (prob_before - prob_after):")
            st.write(avg_imp)

            # plot similarity curve (top PLOT_TOP_N)
            plot_n = min(len(top_sims),  min(10, PLOT_TOP_N))
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(np.arange(1, len(top_sims)+1), top_sims, marker='o')
            ax.set_xlabel("Similar patient rank")
            ax.set_ylabel("Similarity (%)")
            ax.set_title(f"Similarity curve (metric={sim_metric})")
            ax.grid(True)
            st.pyplot(fig)

        else:
            st.info("Click 'Run similarity insights' in the sidebar to compute similar patients and interventions.")

st.write("App ready — train models, select patient, and run similarity insights.")