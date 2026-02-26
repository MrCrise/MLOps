import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train_model
from dir_settings import WORK_DIR, RAW_FILE, CLEAN_FILE, MODEL_FILE


os.makedirs(WORK_DIR, exist_ok=True)


def download_data(**kwargs):
    """
    Скачиваем датасет медицинских расходов.
    """

    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    
    df = pd.read_csv(url)
    df.to_csv(RAW_FILE, index=False)
    return True


def clear_data(**kwargs):
    """
    Очистка и подготовка данных.
    """

    df = pd.read_csv(RAW_FILE)

    cat_columns = ['sex', 'smoker', 'region']
    num_columns = ['age', 'bmi', 'children', 'charges']

    # Удаление дубликатов
    df = df.drop_duplicates()

    # Удаление нереалистичных индексов массы тела.
    anomaly_bmi = df[df['bmi'] > 53] 
    df = df.drop(anomaly_bmi.index)

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df.to_csv(CLEAN_FILE, index=False)
    return True


default_args = {
    'owner': 'mrcrise',
    'start_date': datetime(2026, 2, 19),
    'retries': 0
}

dag_insurance = DAG(
    dag_id="insurance_pipeline",
    default_args=default_args,
    schedule=timedelta(minutes=10),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_data", dag=dag_insurance)
clear_task = PythonOperator(python_callable=clear_data, task_id="clean_data", dag=dag_insurance)
train_task = PythonOperator(python_callable=train_model, task_id="train_model", dag=dag_insurance)

download_task >> clear_task >> train_task
