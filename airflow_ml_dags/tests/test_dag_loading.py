def test_generate_dag_loaded(dag_bag_fixture):
    dag = dag_bag_fixture.get_dag(dag_id="generate_data")
    assert dag_bag_fixture.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_train_dag_loaded(dag_bag_fixture):
    dag = dag_bag_fixture.get_dag(dag_id="train")
    assert dag_bag_fixture.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 6


def test_predict_dag_loaded(dag_bag_fixture):
    dag = dag_bag_fixture.get_dag(dag_id="predict")
    assert dag_bag_fixture.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3
    print(dag.task_dict)
