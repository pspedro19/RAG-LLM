from airflow.decorators import dag, task
from datetime import datetime
from utils import add_adopter_to_model

@dag(
    dag_id='modify_model',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model', 'modify'],
)
def modify_model_dag():
    @task
    def modify_model(model):
        model, peft_config = add_adopter_to_model(model)
        return model, peft_config

    modify_model()

modify_model_dag_instance = modify_model_dag()
