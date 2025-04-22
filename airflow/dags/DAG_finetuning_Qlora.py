import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb
from huggingface_hub import HfFolder

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model_and_tokenizer(model_name):
    # Configuración QLORA más agresiva
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        offload_folder="offload"  # Carpeta para offload
    )
    return model, tokenizer

def add_adapter_to_model(model):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model, peft_config

def set_hyperparameters(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
    )

def train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments):
    tokenizer.padding_side = 'right'
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
        tokenizer=tokenizer,
    )
    return trainer

def save_model(trainer, model_name, output_dir, hf_token):
    model_path = os.path.join(output_dir, "adapter_model")
    trainer.model.save_pretrained(model_path)
    
    try:
        HfFolder.save_token(hf_token)
        trainer.model.push_to_hub(model_name)
        print("Model successfully pushed to Hugging Face Hub.")
    except Exception as e:
        print(f"Failed to save or push model: {e}")

def retrain_llm():
    hf_token = os.environ.get('HF_TOKEN')
    wb_token = os.environ.get('WANDB_TOKEN')
    wandb.login(key=wb_token)

    model_name = "TheBloke/guanaco-7B-HF"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model_name = "DnlModel-QLORA"
    output_dir = "/DnlLLM/src/DnlModel-QLORA"
    ensure_dir(output_dir)

    offload_folder = os.path.join(output_dir, "offload")
    ensure_dir(offload_folder)
    print(f"Offload folder created at: {offload_folder}")

    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train'].train_test_split(test_size=0.1)['train']
    eval_dataset = dataset['train'].train_test_split(test_size=0.1)['test']

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        model, peft_config = add_adapter_to_model(model)
        training_arguments = set_hyperparameters(output_dir)
        trainer = train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments)
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        wandb.init(project="Model Training QLORA")
        wandb.log(eval_results)

        save_model(trainer, new_model_name, output_dir, hf_token)
    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")

    wandb.finish()

if __name__ == "__main__":
    retrain_llm()