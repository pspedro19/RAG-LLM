from airflow.decorators import dag, task
from datetime import datetime
from utils import load_model_and_tokenizer, ensure_dir, BitsAndBytesConfig

@dag(
    dag_id='load_model',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model', 'load'],
)
def load_model_dag():
    @task
    def load_model():
        model_name = "TheBloke/guanaco-7B-HF"
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
        return model, tokenizer

    load_model()

load_model_dag_instance = load_model_dag()
