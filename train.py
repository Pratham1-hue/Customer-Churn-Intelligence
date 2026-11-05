import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import shap
import joblib

np.random.seed(42)

def generate_enterprise_data(n_samples=2500):
    """Generates synthetic enterprise cutomer churn dataset with realisitic features."""
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    tenure_months =  np.random.exponential(scale=18, size=n_samples).clip(1,72).astype(int)
    monthly_charges = np.random.normal(loc=120, scale=40, size=n_samples).clip(30,300)
    support_tickets = np.random.poisson(lam=2.5, size=n_samples)
    login_frequency = np.random.normal(loc=15, scale=5, size=n_samples).clip(1,30)
    nps_score = np.random.randint(1,11, size=n_samples)
    contract_type = np.random.choice(['Month-to-Month', 'One-Year','Two-Year'], size=n_samples, p=[0.5,0.3,0.2])
    payment_delay_days = np.random.poisson(lam=3, size=n_samples)

    # Calculate engineered features
    ticket_density = support_tickets / (tenure_months + 1)

    # Define Underlying groun-truth churn probability using realistic feature weights
    logit = (
        - 1.2
        + 0.08 * (30 - login_frequency)
        + 0.6  * (contract_type == 'Month-to-Month')
        + 0.12 * support_tickets
        + 0.15 * payment_delay_days
        - 0.04 * tenure_months
        - 0.25 * (nps_score - 5)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Tenure_Months': tenure_months,
        'Monthly_Charges': monthly_charges,
        'Support_Tickets': support_tickets,
        'Login_Frequency': login_frequency,
        'NPS_Score': nps_score,
        'Contract_Type': contract_type,
        'Payment_Delay_Days': payment_delay_days,
        'Ticket_Density': ticket_density,
        'Churn': churn
    })
    return df

def run_training_pipeline():
    print(" Step 1: Generating enterprise dataset....")
    df = generate_enterprise_data()

    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn']

    # One Hot Encode 
    X = pd.get_dummies(X, columns=['Contract_Type'], drop_first= True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(" Step 2: Optimizing LightGBM hyperparameters with Optuna...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type':'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)

    print(f" Best Optuna ROC-AUC Score: {study.best_value:.4f}")

    print(" Step 3: Training final LightGBM moddel...")
    best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    best_model.fit(X_train, y_train)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    print(f' Test Set ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}:.4f')
    print(f' Test Set PR-AUC: {average_precision_score(y_test, y_pred_proba):.4f}')
    print(" Step 4: Initializing TreeSHAP Explainer...")
    explainer = shap.TreeExplainer(best_model)
    
    # Save artifacts for Streamlit
    joblib.dump(best_model, 'churn_model.pkl')
    joblib.dump(explainer, 'shap_explainer.pkl')
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(df, 'raw_data.pkl')
    print(" Saved artifacts: churn_model.pkl, shap_explainer.pkl, X_train.pkl, raw_data.pkl")

if __name__ == '__main__':
    run_training_pipeline()