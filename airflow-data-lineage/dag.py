from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.lineage import AUTO
from airflow.lineage.entities import File
from datetime import datetime

def process_data():
    # Your data processing logic here
    pass

with DAG('data_lineage_example', start_date=datetime(2023, 1, 1), schedule_interval='@daily') as dag:
    input_file = File(url="/path/to/input/file")
    output_file = File(url="/path/to/output/file")

    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        inlets=[input_file],
        outlets=[output_file]
    )