"""
Generator DAG module.
"""
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from config import LOCAL_DATA_DIR, default_args


with DAG(
    dag_id="generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    generate = DockerOperator(
        image="airflow-generate-data",
        command="--output-dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        auto_remove="success",
        mounts=[Mount(source=LOCAL_DATA_DIR, target="/data", type="bind")],
    )
