import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
from vecstack import StackingTransformer
from sklearn.pipeline import Pipeline
# Scikit-learn and other model imports for REGRESSION
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate, cross_val_predict

# Import all the REGRESSOR models you'll be using
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, BaggingRegressor, VotingRegressor, StackingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.base import BaseEstimator, RegressorMixin


# ==============================================================================
# 1. STATE INITIALIZATION
# ==============================================================================
def initialize_model_states_reg():
    """Initializes session state for all regression models and their results."""
    model_keys = [
        "sgd", "lin_reg", "dt", "rf", "et", "knn", "ada",
        "svr", "mlp", "gb", "xgb", "lgb", "cat",
        "bagging", "voting", "stacking", "dnn"
    ]
    for key in model_keys:
        if f"{key}_model_reg" not in st.session_state:
            st.session_state[f"{key}_model_reg"] = None
        if f"{key}_eval_results_reg" not in st.session_state:
            st.session_state[f"{key}_eval_results_reg"] = None


class WeightedAverageMetaLearnerReg(BaseEstimator, RegressorMixin):
    """Custom meta-learner for regression that averages predictions."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    def predict(self, X):
        all_preds = [model.predict(X) for model in self.models]
        return np.average(all_preds, axis=0, weights=self.weights)


# ==============================================================================
# 2. EVALUATION & VISUALIZATION HELPERS
# ==============================================================================
def _display_metrics_tab_reg(metrics):
    """Helper function to display a set of regression metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared (R²)", f"{metrics.get('R2', 0):.4f}")
    col2.metric("Mean Absolute Error (MAE)", f"{metrics.get('MAE', 0):.4f}")
    col3.metric("Mean Squared Error (MSE)", f"{metrics.get('MSE', 0):.4f}")
    col4.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('RMSE', 0):.4f}")


def _plot_regression_graphs(y_true, y_pred, title_prefix):
    """Plots Actual vs. Predicted and Residuals for regression evaluation."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.tight_layout(pad=4.0)
    # Actual vs. Predicted Plot
    ax[0].scatter(y_true, y_pred, alpha=0.5, color='blue')
    ax[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', lw=2, color='red')
    ax[0].set_xlabel('Actual Values')
    ax[0].set_ylabel('Predicted Values')
    ax[0].set_title(f'{title_prefix}: Actual vs. Predicted', fontweight='bold')
    # Residuals Plot
    residuals = y_true - y_pred
    ax[1].scatter(y_pred, residuals, alpha=0.5, color='green')
    ax[1].hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', lw=2, linestyles='--')
    ax[1].set_xlabel('Predicted Values')
    ax[1].set_ylabel('Residuals')
    ax[1].set_title(f'{title_prefix}: Residuals Plot', fontweight='bold')
    st.pyplot(fig)


def display_evaluation_from_state_reg(eval_results, model, X_train):
    """Displays the metrics and graphs from a saved regression evaluation state."""
    with st.expander("📊 Model Performance Summary", expanded=True):
        validation_tab, testing_tab = st.tabs(["Validation Metrics (CV)", "Testing Metrics"])
        with validation_tab:
            st.info("Performance averaged across CV folds on the training data.")
            _display_metrics_tab_reg(eval_results["validation_metrics"])
            st.divider()
            _plot_regression_graphs(eval_results["y_train"], eval_results["cv_predict"], "Cross-Validation")
        with testing_tab:
            st.info("Performance on the held-out, unseen test data.")
            _display_metrics_tab_reg(eval_results["test_metrics"])
            st.divider()
            _plot_regression_graphs(eval_results["y_test"], eval_results["test_predict"], "Test Set")


def evaluate_model_reg(X_train, X_test, y_train, y_test, model, model_key):
    st.subheader("💹 Model Performance")
    cv_value = st.slider("Choose No of Folds", 2, 15, 5, 1, key=f"cv_{model_key}_reg")
    if st.button("▶️ Start Evaluation", type="primary", key=f"eval_{model_key}_reg"):
        with st.spinner("⏳ Evaluating model performance..."):
            scorers = {'r2': 'r2', 'neg_mean_absolute_error': 'neg_mean_absolute_error',
                       'neg_mean_squared_error': 'neg_mean_squared_error'}
            cv_scores = cross_validate(model, X_train, y_train, cv=cv_value, scoring=list(scorers.values()), n_jobs=-1)
            test_predict = model.predict(X_test)
            cv_predict = cross_val_predict(model, X_train, y_train, cv=cv_value, n_jobs=-1)

            # CV metrics
            val_metrics = {
                'R2': cv_scores['test_r2'].mean(),
                'MAE': -cv_scores['test_neg_mean_absolute_error'].mean(),
                'MSE': -cv_scores['test_neg_mean_squared_error'].mean(),
                'RMSE': np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean())
            }
            # Test metrics
            test_metrics = {
                'R2': r2_score(y_test, test_predict),
                'MAE': mean_absolute_error(y_test, test_predict),
                'MSE': mean_squared_error(y_test, test_predict),
                'RMSE': np.sqrt(mean_squared_error(y_test, test_predict))
            }
            st.session_state[f"{model_key}_eval_results_reg"] = {
                "validation_metrics": val_metrics, "test_metrics": test_metrics,
                "y_train": y_train, "cv_predict": cv_predict,
                "y_test": y_test, "test_predict": test_predict
            }
        st.rerun()


def evaluate_dnn_model_reg(model, history, X_test, y_test):
    st.subheader("✅ Model Trained & Evaluated")
    with st.expander("🧠 Model Summary", expanded=False):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.code("\n".join(stringlist))
    with st.expander("📈 Training History", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Train Loss (MAE)')
        ax.plot(history.history['val_loss'], label='Validation Loss (MAE)')
        ax.set_title('Model Loss')
        ax.legend()
        st.pyplot(fig)
    with st.expander("📊 Final Test Report", expanded=True):
        y_pred = model.predict(X_test).flatten()
        test_metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        _display_metrics_tab_reg(test_metrics)
        st.divider()
        _plot_regression_graphs(y_test, y_pred, "Test Set")


# ==============================================================================
# 3. GENERIC MODEL UI & INDIVIDUAL MODEL FUNCTIONS
# ==============================================================================
def model_interface_reg(model_class, model_key, X_train, X_test, y_train, y_test, get_params_func):
    """A generic, stateful UI for training and evaluating a single regression model."""
    st.subheader(f"{model_class.__name__}")

    if st.session_state[f"{model_key}_model_reg"] is None:
        params = get_params_func(f"{model_key}_reg")
        if params is None: return

        if st.button(f"Train {model_class.__name__}", key=f"train_{model_key}_reg"):
            with st.spinner("Training..."):
                st.session_state[f"{model_key}_eval_results_reg"] = None
                if 'random_state' in model_class().get_params():
                     model = model_class(**params, random_state=42)
                else:
                     model = model_class(**params)
                model.fit(X_train, y_train)
                st.session_state[f"{model_key}_model_reg"] = model
            st.success("Training Complete!")
            st.rerun()
    else:
        if st.session_state.get(f"{model_key}_eval_results_reg") is None:
            evaluate_model_reg(X_train, X_test, y_train, y_test, st.session_state[f"{model_key}_model_reg"], model_key)
        else:
            display_evaluation_from_state_reg(st.session_state[f"{model_key}_eval_results_reg"],
                                              st.session_state[f"{model_key}_model_reg"], X_train)
        if st.button("Train Again", key=f"retrain_{model_key}_reg"):
            st.session_state[f"{model_key}_model_reg"] = None
            st.session_state[f"{model_key}_eval_results_reg"] = None
            st.rerun()


# --- Parameter Getter Functions for Each Model ---
def bagging_reg(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Bagging Regressor")
    model_key = "bagging"
    full_model_key = f"{model_key}_model_reg"
    eval_results_key = f"{model_key}_eval_results_reg"

    if st.session_state.get(full_model_key) is None:
        st.info("Bagging trains multiple instances of a single base model on random subsets of the data.")
        base_model_name = st.selectbox("Select a Base Model", options=list(all_models.keys()), key="bag_base_model_reg")
        with st.expander("⚙️ Configure Bagging Parameters", expanded=True):
            n_estimators = st.slider("Number of Models", 50, 2000, 100, 10, key='bag_n_reg')
            max_samples = st.slider("Max Samples", 0.1, 1.0, 1.0, 0.1, key='bag_samples_reg')
        if st.button("Train Bagging Model", type='primary', key="train_bagging_reg"):
            with st.spinner("Training..."):
                base_estimator = all_models[base_model_name][0]()
                model = BaggingRegressor(estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples,
                                         random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                st.session_state[full_model_key] = model
            st.rerun()
    else:
        if st.session_state.get(eval_results_key) is None:
            evaluate_model_reg(X_train, X_test, y_train, y_test, st.session_state[full_model_key], model_key)
        else:
            display_evaluation_from_state_reg(st.session_state[eval_results_key], st.session_state[full_model_key], X_train)

        if st.button("Train Again", key="bagging_rerun_reg"):
            st.session_state[full_model_key] = None
            st.session_state[eval_results_key] = None
            st.rerun()


def voting_reg(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Voting Regressor")
    model_key = "voting"
    full_model_key = f"{model_key}_model_reg"
    eval_results_key = f"{model_key}_eval_results_reg"

    if st.session_state.get(full_model_key) is None:
        st.info("Voting averages the predictions from multiple different models.")
        selected_model_names = st.multiselect("Select Models to Combine", options=list(all_models.keys()),
                                              default=list(all_models.keys())[:3], key="vote_select_reg")
        if st.button("Train Voting Model", type='primary', key="train_voting_reg"):
            if not selected_model_names:
                st.warning("Please select at least one model."); return
            estimators = [(name, all_models[name][0]()) for name in selected_model_names]
            with st.spinner("Training Voting model..."):
                model = VotingRegressor(estimators=estimators, n_jobs=-1)
                model.fit(X_train, y_train)
                st.session_state[full_model_key] = model
            st.rerun()
    else:
        if st.session_state.get(eval_results_key) is None:
            evaluate_model_reg(X_train, X_test, y_train, y_test, st.session_state[full_model_key], model_key)
        else:
            display_evaluation_from_state_reg(st.session_state[eval_results_key], st.session_state[full_model_key], X_train)

        if st.button("Train Again", key='voting_rerun_reg'):
            st.session_state[full_model_key] = None
            st.session_state[eval_results_key] = None
            st.rerun()


def stacking_reg(X_train, X_test, y_train, y_test, all_models):
    st.subheader("Stacking Regressor with `vecstack`")
    model_key = "stacking"
    full_model_key = f"{model_key}_model_reg"
    eval_results_key = f"{model_key}_eval_results_reg"

    if st.session_state.get(full_model_key) is None:
        st.info("Stacking uses base model predictions as features for a final meta-model.")
        with st.expander("⚙️ Configure Stacking Architecture", expanded=True):
            st.write("#### Level 1: Base Models")
            level_1_model_names = st.multiselect("Select Base Models", options=list(all_models.keys()),
                                                 default=["Decision Tree", "KNN", "SVR"], key='stack_level1_reg')
            st.write("#### Level 2: Meta-Learner")
            meta_model_name = st.selectbox("Select a Single Meta-Learner", options=list(all_models.keys()),
                                           index=1, key='stack_level2_single_reg')
        if st.button("Train Stacking Model", type='primary', key="train_stacking_reg"):
            if not level_1_model_names:
                st.warning("Please select at least one Level 1 base model."); return
            level_1_estimators = [(name, all_models[name][0]()) for name in level_1_model_names]
            meta_estimator = all_models[meta_model_name][0]()
            pipeline = Pipeline([
                ('stack', StackingTransformer(level_1_estimators, regression=True, metric=r2_score, n_folds=5,
                                             shuffle=True, random_state=42, verbose=0)),
                ('meta_learner', meta_estimator)
            ])
            with st.spinner("Training Stacking model... This can take some time."):
                pipeline.fit(X_train.values, y_train)
                st.session_state[full_model_key] = pipeline
            st.rerun()
    else:
        if st.session_state.get(eval_results_key) is None:
            evaluate_model_reg(X_train, X_test, y_train, y_test, st.session_state[full_model_key], model_key)
        else:
            display_evaluation_from_state_reg(st.session_state[eval_results_key], st.session_state[full_model_key], X_train)

        if st.button("Train Again", key='stacking_rerun_reg'):
            st.session_state[full_model_key] = None
            st.session_state[eval_results_key] = None
            st.rerun()


def dnn_reg(X_train, X_test, y_train, y_test):
    st.subheader("Deep Neural Network (DNN) for Regression")
    model_key = "dnn_model_reg"
    if st.session_state.get(model_key) is None:
        with st.expander("🧠 Configure DNN Architecture", expanded=True):
            num_hidden_layers = st.slider("Number of Hidden Layers", 1, 10, 2, key="dnn_layers_reg")
            layers = []
            cols_layers = st.columns(2)
            for i in range(num_hidden_layers):
                neurons = cols_layers[0].number_input(f"Neurons in Layer {i + 1}", 16, 512, 64, 16, key=f"n_{i}_reg")
                activation = cols_layers[1].selectbox(f"Activation for Layer {i + 1}", ['relu', 'tanh'], key=f"a_{i}_reg")
                layers.append({'neurons': neurons, 'activation': activation})
            st.info("💡 For regression, the output layer has 1 neuron and a 'linear' activation.")
        with st.expander("⚙️ Configure Compilation & Training", expanded=True):
            c1, c2, c3 = st.columns(3)
            optimizer = c1.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'], key="dnn_opt_reg")
            loss = c2.selectbox("Loss Function", ['mean_squared_error', 'mean_absolute_error'], key="dnn_loss_reg")
            epochs = c3.number_input("Training Epochs", 10, 200, 50, 10, key="dnn_epochs_reg")
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05, key='dnn_val_split_reg')
        if st.button("Train DNN Model", type='primary', key="train_dnn_reg"):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
            for layer_params in layers:
                model.add(tf.keras.layers.Dense(units=layer_params['neurons'], activation=layer_params['activation']))
            model.add(tf.keras.layers.Dense(units=1, activation='linear')) # Output layer for regression
            model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            with st.spinner(f"Training DNN for {epochs} epochs..."):
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=32, verbose=0)
                st.session_state[model_key] = model
                st.session_state[f"{model_key}_eval_results_reg"] = history
            st.rerun()
    else:
        history = st.session_state.get(f"{model_key}_eval_results_reg")
        evaluate_dnn_model_reg(st.session_state[model_key], history, X_test, y_test)
        if st.button("Train Again", key='dnn_rerun_reg'):
            st.session_state[model_key] = None
            st.session_state[f"{model_key}_eval_results_reg"] = None
            st.rerun()

# --- REGRESSION-SPECIFIC PARAMETER GETTERS ---
def get_sgd_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['loss'] = st.selectbox("Loss", ['squared_error', 'huber', 'epsilon_insensitive'], key=f"l_{k}")
            p['penalty'] = st.selectbox("Penalty", ['l2', 'l1', 'elasticnet'], key=f"p_{k}")
            p['alpha'] = st.number_input("Alpha", 0.00001, 1.0, 0.0001, format="%.5f", key=f"a_{k}")
    return p

def get_lin_reg_params_reg(k):
    st.info("LinearRegression has no major hyperparameters to tune from this interface.")
    return {}

def get_dt_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['criterion'] = st.selectbox("Criterion", ['squared_error', 'friedman_mse', 'absolute_error'], key=f"crit_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
    return p

def get_rf_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 100, 10, key=f"n_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
            p['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, key=f"l_{k}")
    return p

def get_et_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 100, 10, key=f"n_{k}")
            p['max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, key=f"d_{k}") or None
            p['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, key=f"l_{k}")
    return p

def get_knn_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_neighbors'] = st.slider("K Neighbors", 1, 50, 5, key=f"n_{k}")
            p['weights'] = st.selectbox("Weights", ['uniform', 'distance'], key=f"w_{k}")
    return p

def get_ada_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 10, 500, 50, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 2.0, 1.0, 0.01, key=f"lr_{k}")
            p['loss'] = st.selectbox("Loss", ['linear', 'square', 'exponential'], key=f"loss_{k}")
    return p

def get_svr_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['C'] = st.number_input("C", 0.01, 10.0, 1.0, 0.01, key=f"C_{k}")
            p['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'], key=f"k_{k}")
            p['epsilon'] = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01, key=f"eps_{k}")
    return p

def get_mlp_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            hls = st.text_input("Hidden Layer Sizes (e.g. 100,50)", "100", key=f"hls_{k}")
            p['activation'] = st.selectbox("Activation", ['relu', 'tanh', 'logistic'], key=f"act_{k}")
            p['solver'] = st.selectbox("Solver", ['adam', 'sgd'], key=f"sol_{k}")
            p['alpha'] = st.number_input("Alpha", 0.00001, 0.1, 0.0001, format="%.5f", key=f"a_{k}")
        try:
            p['hidden_layer_sizes'] = tuple(map(int, hls.split(',')))
        except:
            st.error("Invalid format for Hidden Layer Sizes."); return None
    return p

def get_gb_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key=f"lr_{k}")
            p['max_depth'] = st.slider("Max Depth", 1, 15, 3, key=f"d_{k}")
    return p

def get_xgb_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lr_{k}")
            p['max_depth'] = st.slider("Max Depth", 1, 15, 3, key=f"d_{k}")
    return p

def get_lgb_params_reg(k):
    p = {}
    if st.checkbox("Configure hyperparameters", key=f"c_{k}"):
        with st.expander("⚙️ Hyperparameters", True):
            p['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 10, key=f"n_{k}")
            p['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lr_{k}")
            p['num_leaves'] = st.slider("Num Leaves", 20, 100, 31, key=f"nl_{k}")
    return p

def get_cat_params_reg(k):
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
def traning_testing_evaluating_reg(X_train, X_test, y_train, y_test):
    initialize_model_states_reg()

    models = {
        "SGD": (SGDRegressor, "sgd", get_sgd_params_reg),
        "Linear Regression": (LinearRegression, "lin_reg", get_lin_reg_params_reg),
        "Decision Tree": (DecisionTreeRegressor, "dt", get_dt_params_reg),
        "Random Forest": (RandomForestRegressor, "rf", get_rf_params_reg),
        "Extra Trees": (ExtraTreesRegressor, "et", get_et_params_reg),
        "KNN": (KNeighborsRegressor, "knn", get_knn_params_reg),
        "AdaBoost": (AdaBoostRegressor, "ada", get_ada_params_reg),
        "SVR": (SVR, "svr", get_svr_params_reg),
        "MLP Regressor": (MLPRegressor, "mlp", get_mlp_params_reg),
        "Gradient Boosting": (GradientBoostingRegressor, "gb", get_gb_params_reg),
        "XGBoost": (xgb.XGBRegressor, "xgb", get_xgb_params_reg),
        "LightGBM": (lgb.LGBMRegressor, "lgb", get_lgb_params_reg),
        "CatBoost": (cb.CatBoostRegressor, "cat", get_cat_params_reg)
    }

    st.header("🚀 Step 3: Train & Evaluate a Regression Model")
    main_tab1, main_tab2 = st.tabs(["Single Models", "🚀 Advanced Models"])
    with main_tab1:
        tabs = st.tabs(models.keys())
        for tab, model_name in zip(tabs, models.keys()):
            with tab:
                model_class, model_key, params_func = models[model_name]
                model_interface_reg(model_class, model_key, X_train, X_test, y_train, y_test, params_func)
    with main_tab2:
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(["Bagging", "Voting", "Stacking", "Deep Neural Network (DNN)"])
        with adv_tab1:
            bagging_reg(X_train, X_test, y_train, y_test, models)
        with adv_tab2:
            voting_reg(X_train, X_test, y_train, y_test, models)
        with adv_tab3:
            stacking_reg(X_train, X_test, y_train, y_test, models)
        with adv_tab4:
            dnn_reg(X_train, X_test, y_train, y_test)