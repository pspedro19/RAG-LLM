import os
import datetime
import warnings
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb
from huggingface_hub import HfFolder

# Suppress warnings about checkpointing behavior that can be explicitly controlled.
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")

def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model_and_tokenizer(model_name, bnb_config):
    """Load a tokenizer and model with specific quantization configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is correctly set
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

def add_adopter_to_model(model):
    """Prepare model for k-bit training and apply PEFT configurations."""
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    return get_peft_model(model, peft_config)

def set_hyperparameters(output_dir):
    """Define training arguments for model training."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
    )

def train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments):
    """Train a model with specified datasets, PEFT config, and training arguments."""
    tokenizer.padding_side = 'right'
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
    trainer.train()
    return trainer

def save_and_push_model(trainer, model_name, output_dir, hf_token):
    """Save and push the trained model to the Hugging Face Hub."""
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    trainer.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)
    trainer.save_state()
    trainer.args.to_json_file(os.path.join(model_path, "training_args.json"))
    
    # Handle additional files and configurations
    additional_files = ['special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json']
    for file_name in additional_files:
        src_path = os.path.join(output_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, model_path)

    if hasattr(trainer.model.config, "adapter_config"):
        with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
            f.write(trainer.model.config.adapter_config.to_json_string())

    safetensor_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(safetensor_path):
        shutil.copy(safetensor_path, model_path)

    try:
        HfFolder.save_token(hf_token)
        trainer.model.push_to_hub(model_name)
        print("Model successfully pushed to Hugging Face Hub.")
    except Exception as e:
        print(f"Failed to save or push model: {e}")
