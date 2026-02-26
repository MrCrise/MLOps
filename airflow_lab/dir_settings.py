import os


WORK_DIR = os.path.expanduser("~/airflow/data_insurance")

RAW_FILE = os.path.join(WORK_DIR, "insurance.csv")
CLEAN_FILE = os.path.join(WORK_DIR, "insurance_clean.csv")
MODEL_FILE = os.path.join(WORK_DIR, "insurance_model.pkl")