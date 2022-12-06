"""
DAG config module.
"""
import os
from datetime import timedelta
from airflow.models import Variable
from airflow.utils.email import send_email_smtp


LOCAL_DATA_DIR = Variable.get("path_to_data")
LOCAL_MLRUNS_DIR = Variable.get("path_to_mlruns")


def wait_for_file(file_name) -> bool:
    """
    Check if data is ready;
    :param file_name: path to file;
    :return: boolean.
    """
    return os.path.exists(file_name)


def failure_callback(context) -> None:
    """
    Send alert if DAG fails;
    :param context: information from airflow;
    :return: none.
    """
    dag_run = context.get("dag_run")
    subject = f"DAG {dag_run} has failed"
    send_email_smtp(to=default_args["email"], subject=subject, html_content="")


default_args = {
    "owner": "alexey_dvornikov",
    "email": ["applskyp@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "on_failure_callback": failure_callback,
}
