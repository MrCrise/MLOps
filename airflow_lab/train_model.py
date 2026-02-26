from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
from dir_settings import WORK_DIR, RAW_FILE, CLEAN_FILE, MODEL_FILE


def scale_frame(frame):
    df_local = frame.copy()
    X = df_local.drop(columns=['charges'])
    y = df_local['charges']
    
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(**kwargs):
    df = pd.read_csv(CLEAN_FILE)
    
    X, Y, power_trans = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.15, 0.5],
        "penalty": ["l2", "elasticnet"],
        "max_iter": [1000]
    }
    
    mlflow_tracking_uri = "file://" + WORK_DIR + "/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    mlflow.set_experiment("insurance_prediction")
    
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=1)
        
        clf.fit(X_train, y_train.ravel())
        best = clf.best_estimator_
        
        y_pred_scaled = best.predict(X_val)
        
        y_pred_real = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_real = power_trans.inverse_transform(y_val)

        (rmse, mae, r2) = eval_metrics(y_val_real, y_pred_real)
        
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        with open(MODEL_FILE, "wb") as file:
            joblib.dump(best, file)
