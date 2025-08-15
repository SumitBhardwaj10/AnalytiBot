import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
from vecstack import StackingTransformer
from sklearn.pipeline import Pipeline
# Scikit-learn and other model imports
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.model_selection import cross_validate, cross_val_predict

# Import all the models you'll be using
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.base import BaseEstimator, ClassifierMixin

# ==============================================================================
# 1. STATE INITIALIZATION
# ==============================================================================
def initialize_model_states():
    """Initializes session state for all models and their results."""
    model_keys = [
        "sgd", "log_reg", "dt", "rf", "et", "knn", "ada", "gnb",
        "svc", "mlp", "gb", "xgb", "lgb", "cat",
        "bagging", "voting", "stacking", "dnn"  # Advanced Models
    ]
    for key in model_keys:
        if f"{key}_model" not in st.session_state:
            st.session_state[f"{key}_model"] = None
        if f"{key}_eval_results" not in st.session_state:
            st.session_state[f"{key}_eval_results"] = None


class WeightedAverageMetaLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
    def fit(self, X, y):
        # Fit all models
        for model in self.models:
            model.fit(X, y)
        return self
    def predict_proba(self, X):
        # Get probabilities from all models
        all_probas = [model.predict_proba(X) for model in self.models]
        # Compute weighted average
        avg_proba = np.average(all_probas, axis=0, weights=self.weights)
        return avg_proba
    def predict(self, X):
        # Predict based on the highest probability from the weighted average
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
# ==============================================================================
# 2. EVALUATION & VISUALIZATION HELPERS
# ==============================================================================
def _display_metrics_tab(metrics, is_binary):
    """Helper function to display a set of metrics in columns."""
    prec_label = "Precision" if is_binary else "Precision (Macro)"
    rec_label = "Recall" if is_binary else "Recall (Macro)"
    f1_label = "F1 Score" if is_binary else "F1 Score (Macro)"
    auc_label = "ROC AUC" if is_binary else "ROC AUC (OvO)"

    col1, col2, col3 ,col4= st.columns(4)
    col1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}%")
    col2.metric(prec_label, f"{metrics.get('Precision', 0):.2f}%")
    col3.metric(rec_label, f"{metrics.get('Recall', 0):.2f}%")
    if 'ROC AUC' in metrics:
        col4.metric(auc_label, f"{metrics.get('ROC AUC', 0):.2f}%")


def evaluate_advanced_model(model, model_name, X_test, y_test, is_binary):
    """
    Generic evaluation function for advanced models (Bagging, Voting, Stacking).
    Calculates and displays a full suite of metrics and plots.
    """
    st.subheader(f"✅ {model_name} Trained & Evaluated")

    with st.expander("📊 Evaluation Results", expanded=True):
        avg_method = 'binary' if is_binary else 'macro'

        # --- Metrics Calculation ---
        with st.spinner("Calculating performance metrics..."):
            y_pred = model.predict(X_test)
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred) * 100,
                'Precision': precision_score(y_test, y_pred, average=avg_method) * 100,
                'Recall': recall_score(y_test, y_pred, average=avg_method) * 100,
                'F1 Score': f1_score(y_test, y_pred, average=avg_method) * 100
            }
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if is_binary:
                    metrics['ROC AUC'] = roc_auc_score(y_test, y_proba[:, 1]) * 100
                else:
                    # Check for shape mismatch in multiclass case
                    if y_proba.shape[1] == len(np.unique(y_test)):
                        metrics['ROC AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovo') * 100
                    else:
                        metrics['ROC AUC'] = None  # Cannot calculate
            else:
                metrics['ROC AUC'] = None

        # --- Display Metrics ---
        st.write("**Test Set Performance**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}%")
        col2.metric("Precision", f"{metrics.get('Precision', 0):.2f}%")
        col3.metric("Recall", f"{metrics.get('Recall', 0):.2f}%")
        if metrics.get('ROC AUC') is not None:
            col4.metric("ROC AUC", f"{metrics.get('ROC AUC', 0):.2f}%")
        else:
            col4.metric("ROC AUC", "N/A")

        st.divider()

        # --- Display Confusion Matrix and Report ---
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.write("**Classification Report**")
        st.code(classification_report(y_test, y_pred))


def _plot_test_graphs(results, model, X_train, is_binary):
    y_test = results.get("y_test")
    test_predict = results.get("test_predict")
    test_scores_for_roc = results.get("test_scores_for_roc")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout(pad=4.0)
    # Test Confusion Matrix
    cm = confusion_matrix(y_test, test_predict)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0, 0], cbar=False)
    ax[0, 0].set_title('Test Confusion Matrix', fontweight='bold')
    ax[0, 0].set_xlabel('Predicted Label');
    ax[0, 0].set_ylabel('True Label')
    # Test Precision-Recall Curve
    if is_binary and test_scores_for_roc is not None:
        precision, recall, _ = precision_recall_curve(y_test, test_scores_for_roc[:, 1])
        ax[0, 1].plot(recall, precision, color='b', lw=2)
        ax[0, 1].set_title('Test Precision-Recall Curve', fontweight='bold')
    else:
        ax[0, 1].text(0.5, 0.5, 'PR Curve for binary cases only.', ha='center')
    # Test ROC Curve
    if is_binary and test_scores_for_roc is not None:
        fpr, tpr, _ = roc_curve(y_test, test_scores_for_roc[:, 1])
        ax[1, 0].plot(fpr, tpr, color='darkorange', lw=2)
        ax[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[1, 0].set_title('Test ROC Curve', fontweight='bold')
    else:
        ax[1, 0].text(0.5, 0.5, 'ROC Curve for binary cases only.', ha='center')
    # Feature Importance
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    if importance is not None:
        feat_imp = pd.Series(importance, index=X_train.columns).sort_values(ascending=False).head(20)
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax[1, 1], palette='viridis')
        ax[1, 1].set_title('Top Feature Importances', fontweight='bold')
    else:
        ax[1, 1].text(0.5, 0.5, 'Feature importance not available.', ha='center')
    st.pyplot(fig)
    st.code(classification_report(y_test, test_predict))


def _plot_cv_graphs(results, is_binary):
    y_train = results.get("y_train")
    cv_predict = results.get("cv_predict")
    cv_scores_for_roc = results.get("cv_scores_for_roc")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.tight_layout(pad=4.0)
    # CV Confusion Matrix
    cm = confusion_matrix(y_train, cv_predict)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax[0], cbar=False)
    ax[0].set_title('CV Confusion Matrix', fontweight='bold')
    # CV Precision-Recall Curve
    if is_binary and cv_scores_for_roc is not None:
        precision, recall, _ = precision_recall_curve(y_train, cv_scores_for_roc[:, 1])
        ax[1].plot(recall, precision, color='g', lw=2)
        ax[1].set_title('CV Precision-Recall Curve', fontweight='bold')
    else:
        ax[1].text(0.5, 0.5, 'PR Curve for binary cases only.', ha='center')
    # CV ROC Curve
    if is_binary and cv_scores_for_roc is not None:
        fpr, tpr, _ = roc_curve(y_train, cv_scores_for_roc[:, 1])
        ax[2].plot(fpr, tpr, color='darkorange', lw=2)
        ax[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[2].set_title('CV ROC Curve', fontweight='bold')
    else:
        ax[2].text(0.5, 0.5, 'ROC Curve for binary cases only.', ha='center')
    st.pyplot(fig)
    st.code(classification_report(y_train, cv_predict))


def display_evaluation_from_state(eval_results, model, X_train):
    """Displays the metrics and graphs from a saved evaluation state."""
    is_binary = eval_results.get("is_binary")
    with st.expander("📊 Model Performance Summary", expanded=True):
        validation_tab, testing_tab = st.tabs(["Validation Metrics (CV)", "Testing Metrics"])
        with validation_tab:
            st.info("Performance averaged across CV folds on the training data.")
            _display_metrics_tab(eval_results["validation_metrics"], is_binary)
            st.divider()
            _plot_cv_graphs(eval_results, is_binary)
        with testing_tab:
            st.info("Performance on the held-out, unseen test data.")
            _display_metrics_tab(eval_results["test_metrics"], is_binary)
            st.divider()
            _plot_test_graphs(eval_results, model, X_train, is_binary)


def evaluate_model(X_train, X_test, y_train, y_test, model, model_key):
    st.subheader("💹 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        cv_value=st.slider("Choose No of Fold",2,15,3,1)
    with col2:
        st.warning("High Fold value increase training time")
    if st.button("▶️ Start Evaluation", type="primary", key=f"eval_{model_key}"):
        with st.spinner("⏳ Evaluating model performance..."):
            is_binary = len(np.unique(y_train)) == 2
            scorers = {'accuracy': 'accuracy', 'precision': 'precision_macro' if not is_binary else 'precision',
                       'recall': 'recall_macro' if not is_binary else 'recall'
                    }
            cv_scores = cross_validate(model, X_train, y_train, cv=cv_value, scoring=list(scorers.values()), n_jobs=-1)
            test_predict = model.predict(X_test)
            cv_predict = cross_val_predict(model, X_train, y_train, cv=cv_value, n_jobs=-1)
            cv_scores_for_roc, test_scores_for_roc = None, None
            if hasattr(model, "predict_proba"):
                cv_scores_for_roc = cross_val_predict(model, X_train, y_train, cv=cv_value, method="predict_proba", n_jobs=-1)
                test_scores_for_roc = model.predict_proba(X_test)
            elif hasattr(model, "decision_function"):
                cv_scores_for_roc = cross_val_predict(model, X_train, y_train, cv=cv_value, method="decision_function",
                                                      n_jobs=-1)
                test_scores_for_roc = model.decision_function(X_test)
            validation_metrics = {k.capitalize(): cv_scores[f"test_{v}"].mean() * 100 for k, v in scorers.items()}
            test_metrics = {'Accuracy': accuracy_score(y_test, test_predict) * 100,
                            'Precision': precision_score(y_test, test_predict,
                                                         average='macro' if not is_binary else 'binary') * 100,
                            'Recall': recall_score(y_test, test_predict,
                                                   average='macro' if not is_binary else 'binary') * 100
                            }
            if cv_scores_for_roc is not None:
                if is_binary:
                    validation_metrics['ROC AUC'] = roc_auc_score(y_train, cv_scores_for_roc[:, 1]) * 100
                    test_metrics['ROC AUC'] = roc_auc_score(y_test, test_scores_for_roc[:, 1]) * 100
                else:
                    validation_metrics['ROC AUC'] = roc_auc_score(y_train, cv_scores_for_roc, multi_class='ovo') * 100
                    test_metrics['ROC AUC'] = roc_auc_score(y_test, test_scores_for_roc, multi_class='ovo') * 100

            # Store all results in a single dictionary in session state
            st.session_state[f"{model_key}_eval_results"] = {
                "validation_metrics": validation_metrics, "test_metrics": test_metrics,
                "y_test": y_test, "test_predict": test_predict, "test_scores_for_roc": test_scores_for_roc,
                "y_train": y_train, "cv_predict": cv_predict, "cv_scores_for_roc": cv_scores_for_roc,
                "is_binary": is_binary
            }
        st.rerun()


def evaluate_dnn_model(model, history, X_test, y_test):
    """
    Smarter evaluation for TensorFlow models.
    It automatically detects binary vs. multiclass output from the model's architecture.
    """
    st.subheader("✅ Model Trained & Evaluated")
    with st.expander("🧠 Model Summary", expanded=False):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_str = "\n".join(stringlist)
        st.code(summary_str)

    with st.expander("📈 Training History", expanded=True):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        # Plot Accuracy and Loss if they exist in history
        if 'val_accuracy' in history.history:
            ax[0].plot(history.history['accuracy'], label='Train Accuracy')
            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax[0].set_title('Model Accuracy')
            ax[0].legend()
        if 'val_loss' in history.history:
            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Validation Loss')
            ax[1].set_title('Model Loss')
            ax[1].legend()
        st.pyplot(fig)

    with st.expander("📊 Final Test Report", expanded=True):
        y_prob = model.predict(X_test)

        # --- NEW ROBUST LOGIC: Inspect model output shape ---
        # model.output_shape is (batch_size, num_neurons)
        # We check the second dimension.
        if model.output_shape[1] == 1:
            st.info("💡 Binary output layer detected (1 neuron). Predictions based on probability > 0.5.")
            y_pred = (y_prob > 0.5).astype("int32").reshape(-1)
        else:
            st.info(
                f"💡 Multiclass output layer detected ({model.output_shape[1]} neurons). Predictions based on highest probability class.")
            y_pred = np.argmax(y_prob, axis=1)

        st.code(classification_report(y_test, y_pred))

# ==============================================================================
# 3. GENERIC MODEL UI & INDIVIDUAL MODEL FUNCTIONS
# ==============================================================================
def model_interface(model_class, model_key, X_train, X_test, y_train, y_test, get_params_func):
    """A generic, stateful UI for training and evaluating a single model."""
    st.subheader(f"{model_class.__name__}")

    if st.session_state[f"{model_key}_model"] is None:
        params = get_params_func(model_key)
        if params is None: return  # Stop if params are invalid

        if st.button(f"Train {model_class.__name__}", key=f"train_{model_key}"):
            with st.spinner("Training..."):
                # Clear any old evaluation results before training
                st.session_state[f"{model_key}_eval_results"] = None
                model = model_class(**params, random_state=42)
                model.fit(X_train, y_train)
                st.session_state[f"{model_key}_model"] = model
            st.success("Training Complete!")
            st.rerun()
    else:
        # If model is trained, check if evaluation results exist.
        if st.session_state[f"{model_key}_eval_results"] is None:
            evaluate_model(X_train, X_test, y_train, y_test, st.session_state[f"{model_key}_model"], model_key)
        else:
            # If results exist, just display them
            display_evaluation_from_state(st.session_state[f"{model_key}_eval_results"],
                                          st.session_state[f"{model_key}_model"], X_train)

        if st.button("Train Again", key=f"retrain_{model_key}"):
            st.session_state[f"{model_key}_model"] = None
            st.session_state[f"{model_key}_eval_results"] = None
            st.rerun()


# --- Parameter Getter Functions for Each Model ---
def bagging_(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Bagging Classifier")
    model_key = "bagging_model"
    # NEW: Determine if the problem is binary classification
    is_binary = len(np.unique(y_train)) == 2

    if st.session_state.get(model_key) is None:
        st.info("Bagging trains multiple instances of a single base model on random subsets of the data.")
        base_model_name = st.selectbox("Select a Base Model", options=list(all_models.keys()))

        with st.expander("⚙️ Configure Bagging Parameters", expanded=True):
            n_estimators = st.slider("Number of Models", 50, 2000, 100, 10, key='bag_n')
            max_samples = st.slider("Max Samples (fraction of data for each model)", 0.1, 1.0, 1.0, 0.1, key='bag_samples')

        if st.button("Train Bagging Model", type='primary'):
            with st.spinner("Training..."):
                base_estimator = all_models[base_model_name][0]()  # Instantiate the chosen base model
                model = BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                st.session_state[model_key] = model
            st.rerun()
    else:
        # CHANGED: Updated the call to the evaluation function
        evaluate_advanced_model(
            st.session_state[model_key],
            "Bagging Classifier", # Pass the model name
            X_test,
            y_test,
            is_binary # Pass the is_binary flag
        )
        # CHANGED: Added a unique key to the button
        if st.button("Train Again", key="bagging_rerun"):
            st.session_state[model_key] = None
            st.rerun()


def voting_(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Voting Classifier")
    model_key = "voting_model"
    is_binary = len(np.unique(y_train)) == 2

    if st.session_state.get(model_key) is None:
        st.info("Voting combines predictions from multiple different models.")

        selected_model_names = st.multiselect(
            "Select Models to Combine",
            options=list(all_models.keys()),
            default=list(all_models.keys())[:3]
        )

        with st.expander("⚙️ Configure Voting Parameters", expanded=True):
            voting_type = st.radio("Voting Type", ['hard', 'soft'], index=1, horizontal=True)

        if st.button("Train Voting Model", type='primary'):
            if not selected_model_names:
                st.warning("Please select at least one model.")
                return

            estimators = []
            # --- SMART CHANGE: Modify models on the fly for soft voting ---
            for name in selected_model_names:
                model_class = all_models[name][0]

                # If soft voting is selected, try to make models compatible
                if voting_type == 'soft':
                    if name == "SVC":
                        st.info("✨ Smart Change: Enabled `probability=True` for SVC to support soft voting.")
                        estimators.append((name, model_class(probability=True, random_state=42)))
                    else:
                        model_instance = model_class()
                        if not hasattr(model_instance, 'predict_proba'):
                            st.error(
                                f"❌ Error: The model '{name}' does not support `predict_proba` and cannot be used with soft voting. Please use 'hard' voting or remove this model.")
                            return  # Stop execution
                        estimators.append((name, model_instance))
                else:
                    # For hard voting, no changes needed
                    estimators.append((name, model_class()))

            with st.spinner("Training Voting model..."):
                model = VotingClassifier(estimators=estimators, voting=voting_type, n_jobs=-1)
                model.fit(X_train, y_train)
                st.session_state[model_key] = model
            st.rerun()
    else:
        # --- CHANGED: Use the new evaluation function ---
        evaluate_advanced_model(st.session_state[model_key], "Voting Classifier", X_test, y_test, is_binary)
        if st.button("Train Again", key='voting_rerun'):
            st.session_state[model_key] = None
            st.rerun()


def stacking_(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Stacking Classifier with `vecstack`")
    model_key = "stacking_model"
    is_binary = len(np.unique(y_train)) == 2

    if st.session_state.get(model_key) is None:
        st.info("Stacking uses base model predictions as features for a final meta-model.")

        with st.expander("⚙️ Configure Stacking Architecture", expanded=True):
            st.write("#### Level 1: Base Models")
            level_1_model_names = st.multiselect(
                "Select Base Models (to generate features)",
                options=list(all_models.keys()), default=["Decision Tree", "KNN", "Gaussian NB"], key='stack_level1'
            )

            st.write("#### Level 2: Meta-Learner")
            use_weighted_avg = st.checkbox("Use Weighted Average of multiple Meta-Learners", key='stack_weighted_avg')

            meta_estimator = None
            if use_weighted_avg:
                st.write("Select multiple models and assign their weights for the final prediction.")
                meta_model_names = st.multiselect(
                    "Select Meta-Learner Models",
                    options=list(all_models.keys()), default=["Logistic Regression", "Random Forest"], key='stack_level2_multi'
                )
                weights = []
                if meta_model_names:
                    cols = st.columns(len(meta_model_names))
                    for i, name in enumerate(meta_model_names):
                        with cols[i]:
                           weights.append(st.number_input(f"Weight for {name}", min_value=0.1, value=1.0, key=f'w_{name}'))
            else:
                meta_model_name = st.selectbox(
                    "Select a Single Meta-Learner",
                    options=list(all_models.keys()), index=1, key='stack_level2_single'
                )

        if st.button("Train Stacking Model", type='primary'):
            if not level_1_model_names:
                st.warning("Please select at least one Level 1 base model.")
                return

            # --- THE FIX IS HERE ---
            # Create a list of (name, model) tuples as required by vecstack
            level_1_estimators = [(name, all_models[name][0]()) for name in level_1_model_names]

            if use_weighted_avg:
                if not meta_model_names:
                    st.warning("Please select at least one meta-learner model for weighted averaging.")
                    return
                meta_models = [all_models[name][0]() for name in meta_model_names]
                for model, name in zip(meta_models, meta_model_names):
                    if not hasattr(model, 'predict_proba'):
                         st.error(f"Error: Meta-learner '{name}' does not support predict_proba, which is required for weighted averaging. Please remove it.")
                         return
                meta_estimator = WeightedAverageMetaLearner(models=meta_models, weights=weights)
            else:
                meta_estimator = all_models[meta_model_name][0]()

            pipeline = Pipeline([
                ('stack', StackingTransformer(level_1_estimators, regression=False, variant='A', metric=accuracy_score, n_folds=5, shuffle=True, random_state=42, verbose=0)),
                ('meta_learner', meta_estimator)
            ])

            with st.spinner("Training Stacking model... This can take some time."):
                # Use .values to pass a NumPy array, which is standard for many libraries
                pipeline.fit(X_train.values, y_train)
                st.session_state[model_key] = pipeline
            st.rerun()
    else:
        evaluate_advanced_model(st.session_state[model_key], "Stacking Classifier", X_test, y_test, is_binary)
        if st.button("Train Again", key='stacking_rerun'):
            st.session_state[model_key] = None
            st.rerun()


def dnn_(X_train, X_test, y_train, y_test):
    st.subheader("Deep Neural Network (DNN) with TensorFlow")
    model_key = "dnn_model"

    if st.session_state.get(model_key) is None:
        # --- Basic Architecture Configuration ---
        with st.expander("🧠 Configure Hidden Layers", expanded=True):
            num_hidden_layers = st.slider("Number of Hidden Layers", 1, 10, 2, key="dnn_layers")
            layers = []
            cols_layers = st.columns(2)
            for i in range(num_hidden_layers):
                neurons = cols_layers[0].number_input(f"Neurons in Layer {i + 1}", 16, 512, 64, 16, key=f"n_{i}")
                activation = cols_layers[1].selectbox(f"Activation for Layer {i + 1}", ['relu', 'tanh', 'sigmoid'],
                                                      key=f"a_{i}")
                layers.append({'neurons': neurons, 'activation': activation})

        # --- Determine Smart Defaults ---
        n_classes = len(np.unique(y_train))
        is_binary = n_classes == 2
        if is_binary:
            rec_neurons = 1
            rec_activation = 'sigmoid'
            rec_loss = 'binary_crossentropy'
            st.info("💡 Smart Detection: Binary classification task.")
        else:
            rec_neurons = n_classes
            rec_activation = 'softmax'
            rec_loss = 'sparse_categorical_crossentropy'
            st.info(f"💡 Smart Detection: Multiclass classification ({n_classes} classes).")

        # --- NEW: UI for Overriding Smart Settings ---
        with st.expander("⚙️ Fine-Tune Output Layer & Compilation"):
            st.write("The settings below are pre-filled with recommended defaults. You can override them if needed.")

            c1, c2, c3 = st.columns(3)
            # User choice widgets with smart defaults
            user_neurons = c1.number_input("Output Neurons", 1, 1000, value=rec_neurons)
            user_activation = c2.selectbox("Output Activation", ['sigmoid', 'softmax', 'tanh'],
                                           index=['sigmoid', 'softmax', 'tanh'].index(rec_activation))
            user_loss = c3.selectbox("Loss Function", ['binary_crossentropy', 'sparse_categorical_crossentropy',
                                                       'categorical_crossentropy', 'hinge'],
                                     index=['binary_crossentropy', 'sparse_categorical_crossentropy',
                                            'categorical_crossentropy', 'hinge'].index(rec_loss))

            c4, c5 = st.columns(2)
            optimizer = c4.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
            epochs = c5.number_input("Training Epochs", 10, 200, 50, 10)
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05, key='dnn_val_split')

        # --- NEW: Warning Logic ---
        warnings = []
        if user_neurons != rec_neurons:
            warnings.append(f"<li>Output Neurons: <b>{user_neurons}</b> (Recommended: {rec_neurons})</li>")
        if user_activation != rec_activation:
            warnings.append(f"<li>Output Activation: <b>'{user_activation}'</b> (Recommended: '{rec_activation}')</li>")
        if user_loss != rec_loss:
            warnings.append(f"<li>Loss Function: <b>'{user_loss}'</b> (Recommended: '{rec_loss}')</li>")

        if warnings:
            st.warning(
                "⚠️ **Performance Warning** You have overridden the recommended settings. This may impact model performance.")

        if st.button("Train DNN Model", type='primary'):
            # Build the model using user-selected settings
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
            for layer_params in layers:
                model.add(tf.keras.layers.Dense(units=layer_params['neurons'], activation=layer_params['activation']))
            # Use the user's choices for the final layer
            model.add(tf.keras.layers.Dense(units=user_neurons, activation=user_activation))

            # Compile using the user's choices
            model.compile(optimizer=optimizer, loss=user_loss, metrics=['accuracy'])

            with st.spinner(f"Training DNN for {epochs} epochs..."):
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=32,
                                    verbose=0)
                st.session_state[model_key] = model
                st.session_state[f"{model_key}_eval_results"] = history
            st.rerun()
    else:
        history = st.session_state.get(f"{model_key}_eval_results")
        # The new evaluation function doesn't need is_binary anymore
        evaluate_dnn_model(st.session_state[model_key], history, X_test, y_test)
        if st.button("Train Again", key='dnn_rerun'):
            st.session_state[model_key] = None
            st.session_state[f"{model_key}_eval_results"] = None
            st.rerun()


def get_sgd_params(k):
    p = {'loss': 'log_loss'}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['loss'] = st.selectbox("Loss", ['log_loss', 'modified_huber'], key=f"l_{k}")
            p['penalty'] = st.selectbox("Penalty", ['l2', 'l1', 'elasticnet'], key=f"p_{k}")
            p['alpha'] = st.number_input("Alpha", 0.00001, 1.0, 0.0001, format="%.5f", key=f"a_{k}")
    return p


def get_log_reg_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['C'] = st.number_input("C", 0.01, 10.0, 1.0, 0.01, key=f"C_{k}")
            p['solver'] = st.selectbox("Solver", ['lbfgs', 'liblinear', 'saga'], key=f"s_{k}")
            p['penalty'] = 'l2' if p['solver'] in ['lbfgs', 'newton-cg'] else st.selectbox("Penalty",
                                                                                           ['l1', 'l2', 'elasticnet'],
                                                                                           key=f"p_{k}")
    return p


def get_dt_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['criterion'] = st.selectbox("Criterion", ['gini', 'entropy'], key=f"crit_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
    return p


def get_rf_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 100, 10, key=f"n_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
            p['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, key=f"l_{k}")
    return p


def get_et_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 100, 10, key=f"n_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
            p['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, key=f"l_{k}")
    return p


def get_knn_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_neighbors'] = st.slider("K Neighbors", 1, 50, 5, key=f"n_{k}")
            p['weights'] = st.selectbox("Weights", ['uniform', 'distance'], key=f"w_{k}")
    return p


def get_ada_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 50, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 2.0, 1.0, 0.01, key=f"lr_{k}")
    return p


def get_gnb_params(k):
    st.info("GaussianNB has minimal hyperparameters to tune.")
    return {}


def get_svc_params(k):
    p = {'probability': True}
    st.warning("💡 `probability=True` is enabled for ROC plots, which can slow down training.")
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['C'] = st.number_input("C", 0.01, 10.0, 1.0, 0.01, key=f"C_{k}")
            p['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'], key=f"k_{k}")
    return p


def get_mlp_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            hls = st.text_input("Hidden Layer Sizes (e.g. 100,50)", "100", key=f"hls_{k}")
            p['activation'] = st.selectbox("Activation", ['relu', 'tanh'], key=f"act_{k}")
            p['solver'] = st.selectbox("Solver", ['adam', 'sgd'], key=f"sol_{k}")
            p['alpha'] = st.number_input("Alpha", 0.00001, 0.1, 0.0001, format="%.5f", key=f"a_{k}")
        try:
            p['hidden_layer_sizes'] = tuple(map(int, hls.split(',')))
        except:
            st.error("Invalid format for Hidden Layer Sizes.");
            return None
    return p


def get_gb_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key=f"lr_{k}")
            p['max_depth'] = st.slider("Max Depth", 1, 15, 3, key=f"d_{k}")
    return p


def get_xgb_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lr_{k}")
            p['max_depth'] = st.slider("Max Depth", 1, 15, 3, key=f"d_{k}")
    return p


def get_lgb_params(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lr_{k}")
            p['num_leaves'] = st.slider("Num Leaves", 20, 100, 31, key=f"nl_{k}")
    return p


def get_cat_params(k):
    p = {'verbose': 0}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['iterations'] = st.slider("Iterations", 100, 1000, 200, 50, key=f"i_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lr_{k}")
            p['depth'] = st.slider("Depth", 4, 10, 6, key=f"d_{k}")
    return p


# ==============================================================================
# 4. MAIN ORCHESTRATOR FUNCTION
# ==============================================================================
def traning_testing_evaluating(X_train, X_test, y_train, y_test):
    initialize_model_states()

    models = {
        "SGD": (SGDClassifier, "sgd", get_sgd_params),
        "Logistic Regression": (LogisticRegression, "log_reg", get_log_reg_params),
        "Decision Tree": (DecisionTreeClassifier, "dt", get_dt_params),
        "Random Forest": (RandomForestClassifier, "rf", get_rf_params),
        "Extra Trees": (ExtraTreesClassifier, "et", get_et_params),
        "KNN": (KNeighborsClassifier, "knn", get_knn_params),
        "AdaBoost": (AdaBoostClassifier, "ada", get_ada_params),
        "Gaussian NB": (GaussianNB, "gnb", get_gnb_params),
        "SVC": (SVC, "svc", get_svc_params),
        "MLP Classifier": (MLPClassifier, "mlp", get_mlp_params),
        "Gradient Boosting": (GradientBoostingClassifier, "gb", get_gb_params),
        "XGBoost": (xgb.XGBClassifier, "xgb", get_xgb_params),
        "LightGBM": (lgb.LGBMClassifier, "lgb", get_lgb_params),
        "CatBoost": (cb.CatBoostClassifier, "cat", get_cat_params)
    }

    st.header("🚀 Step 3: Train & Evaluate a Model")
    main_tab1, main_tab2 = st.tabs(["Single Models", "🚀 Advanced Models"])
    with main_tab1:
        tabs = st.tabs(models.keys())

        for tab, model_name in zip(tabs, models.keys()):
            with tab:
                model_class, model_key, params_func = models[model_name]
                model_interface(model_class, model_key, X_train, X_test, y_train, y_test, params_func)
    with main_tab2:
        base_models_for_ensembles = {
            "SGD": (SGDClassifier, "sgd", {}),
            "Logistic Regression": (LogisticRegression, "log_reg", {}),
            "Decision Tree": (DecisionTreeClassifier, "dt", {}),
            "Random Forest": (RandomForestClassifier, "rf", {}),
            "Extra Trees": (ExtraTreesClassifier, "et", {}),
            "KNN": (KNeighborsClassifier, "knn", {}),
            "AdaBoost": (AdaBoostClassifier, "ada", {}),
            "Gaussian NB": (GaussianNB, "gnb", {}),
            "SVC": (SVC, "svc", {}),
            "MLP Classifier": (MLPClassifier, "mlp", {}),
            "Gradient Boosting": (GradientBoostingClassifier, "gb", {}),
            "XGBoost": (xgb.XGBClassifier, "xgb", {}),
            "LightGBM": (lgb.LGBMClassifier, "lgb", {}),
            "CatBoost": (cb.CatBoostClassifier, "cat", {})
        }
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(["Bagging", "Voting", "Stacking", "Deep Neural Network (DNN)"])
        with adv_tab1:
            bagging_(X_train, X_test, y_train, y_test, base_models_for_ensembles)
        with adv_tab2:
            voting_(X_train, X_test, y_train, y_test, base_models_for_ensembles)
        with adv_tab3:
            stacking_(X_train, X_test, y_train, y_test, base_models_for_ensembles)
        with adv_tab4:
            dnn_(X_train, X_test, y_train, y_test)
