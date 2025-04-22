from airflow.decorators import dag, task
from datetime import datetime
from utils import save_and_push_model, ensure_dir

@dag(
    dag_id='push_to_hugging_face',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model', 'huggingface'],
)
def push_to_hugging_face_dag():
    @task
    def push_to_hub(trainer):
        output_dir = "/path/to/save/model"
        ensure_dir(output_dir)
        hf_token = 'your_hf_token_here'
        save_and_push_model(trainer, "DnlModel", output_dir, hf_token)

    push_to_hub()

push_to_hugging_face_dag_instance = push_to_hugging_face_dag()
