import pytest
from airflow.models import DagBag


@pytest.fixture()
def dag_bag_fixture() -> DagBag:
    return DagBag()
