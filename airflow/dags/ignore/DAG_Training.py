from airflow.decorators import dag, task
from datetime import datetime
from utils import train_model, set_hyperparameters, load_dataset

@dag(
    dag_id='train_model',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model', 'train'],
)
def train_model_dag():
    @task
    def train_and_log(model, peft_config, tokenizer):
        dataset_name = "mlabonne/guanaco-llama2-1k"
        dataset = load_dataset(dataset_name)
        train_dataset = dataset['train'].train_test_split(test_size=0.1)['train']
        eval_dataset = dataset['train'].train_test_split(test_size=0.1)['test']
        output_dir = "/path/to/output/dir"
        training_arguments = set_hyperparameters(output_dir)
        trainer = train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments)
        return trainer

    train_and_log()

train_model_dag_instance = train_model_dag()
