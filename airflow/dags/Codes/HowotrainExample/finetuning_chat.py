import os
import shutil
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import HfFolder, login
import random
# Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load model and tokenizer
def load_model_and_tokenizer(model_name, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)  # Set legacy=False to avoid the warning
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

# Add LoRA adapter to the mode
def add_adopter_to_model(model):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
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
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
    )

def train_model(model, train_dataset, eval_dataset, tokenizer, training_arguments, output_dir):
    tokenizer.padding_side = 'right'
    model.config.use_cache = False

 # Preparar los datasets
    train_dataset = prepare_dataset(train_dataset)
    eval_dataset = prepare_dataset(eval_dataset)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=1024,
        dataset_text_field="text",
        packing=False
    )
    trainer.train()
    return trainer

# Save the model to disk and push to Hugging Face
def save_model(trainer, model_name, output_dir, hf_token):
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)  # Ensure the directory exists
    trainer.model.save_pretrained(model_path)  # Save the model
    trainer.tokenizer.save_pretrained(model_path)  # Save the tokenizer
    trainer.save_state()  # Save optimizer, scheduler, and trainer state

    # Save training arguments as a JSON file
    import json
    with open(os.path.join(model_path, "training_args.json"), "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=2)

    # Save additional tokenizer files if they are not automatically saved
    additional_files = ['special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json']
    for file_name in additional_files:
        src_path = os.path.join(output_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, model_path)

    # Push model to Hugging Face Hub
    try:
        login(token=hf_token)
        trainer.model.push_to_hub(model_name)
        print(f"Model successfully pushed to Hugging Face Hub under the name: {model_name}")
    except Exception as e:
        print(f"Failed to save or push model: {e}")
        # Global variable para el dataset
        global_dataset = None
        
# Global variable para el dataset       
global_dataset = None

def load_global_dataset(dataset_name):
    global global_dataset
    global_dataset = load_dataset(dataset_name)
        
def format_instruction(sample):
    """
    Formatea la instrucción para el modelo, generando una conversación basada en el dataset.
    """
    global global_dataset
    
    if global_dataset is None:
        raise ValueError("Dataset not loaded. Call load_global_dataset first.")

    # Buscar una entrada relevante en el dataset basada en la pregunta del usuario
    relevant_entries = [entry for entry in global_dataset['train'] if sample['input'].lower() in entry['input'].lower()]
    
    if relevant_entries:
        # Si se encuentran entradas relevantes, elige una al azar
        chosen_entry = random.choice(relevant_entries)
        
        formatted_prompt = f"""### Human: {chosen_entry['input']}

### Assistant: {chosen_entry['output']}

### Human: Gracias por la información. ¿Puedes darme más detalles sobre esto?

### Assistant: Por supuesto, estaré encantado de proporcionarte más detalles. Basándome en tu pregunta sobre {chosen_entry['input']}, puedo agregar lo siguiente:

[Aquí el modelo puede generar información adicional basada en el contexto del dataset y la respuesta anterior]

¿Hay algo más específico que te gustaría saber sobre este tema?

### Human: Eso es muy útil. ¿Tienes alguna recomendación final al respecto?

### Assistant: Me alegra que la información te sea útil. Como recomendación final sobre {chosen_entry['input']}, te sugiero:

[Aquí el modelo puede generar una recomendación final basada en el contexto del dataset y la conversación anterior]

¿Hay algo más en lo que pueda ayudarte hoy con respecto a propiedades en Mar del Plata?

### Human: No, eso es todo por ahora. Muchas gracias por tu ayuda.

### Assistant: Ha sido un placer ayudarte. Si en el futuro tienes más preguntas sobre propiedades en Mar del Plata o necesitas asesoramiento inmobiliario, no dudes en contactarme. Te deseo mucho éxito en tu búsqueda o inversión inmobiliaria. ¡Que tengas un excelente día!
"""
    else:
        # Si no se encuentran entradas relevantes, usa un mensaje genérico
        formatted_prompt = f"""### Human: {sample['input']}

### Assistant: Gracias por tu pregunta sobre "{sample['input']}". Aunque no tengo información específica sobre esto en mi base de datos de propiedades en Mar del Plata, puedo ofrecerte información general sobre el mercado inmobiliario en la zona. 
###Mar del Plata es una ciudad costera muy popular tanto para vivir como para invertir en propiedades. Algunas áreas populares incluyen La Perla, Güemes, y el centro de la ciudad. Cada zona tiene sus propias características y ventajas.

###¿Te gustaría saber más sobre alguna zona en particular o sobre algún tipo específico de propiedad en Mar del Plata?

### Human: Sí, me gustaría saber más sobre la zona de La Perla.

### Assistant: Excelente elección. La Perla es una de las zonas más populares de Mar del Plata. Aquí tienes algunos datos sobre esta área:

###1. Ubicación: Está situada cerca del centro de la ciudad y tiene acceso directo a la playa.
###2. Tipo de propiedades: Encontrarás una mezcla de edificios de apartamentos modernos y casas más tradicionales.
###3. Atractivo: Es muy popular entre turistas y residentes por su proximidad a la playa y sus servicios.
###4. Inversión: Suele ser una buena opción para inversores debido a la alta demanda de alquileres, especialmente en temporada alta.

###¿Hay algo más específico que te gustaría saber sobre La Perla o sobre invertir en esta zona?

### Human: Eso es útil, gracias. ¿Alguna recomendación final?

### Assistant: Me alegra que encuentres útil la información. Como recomendación final para quienes consideran invertir o vivir en La Perla, te sugeriría: 

###1. Visita la zona en diferentes épocas del año para entender cómo cambia entre la temporada alta y baja.
###2. Considera el tipo de propiedad que mejor se adapte a tus necesidades: un apartamento puede ser más fácil de mantener si es para alquiler, mientras que una casa podría ser mejor para vivir todo el año.
###3. Investiga sobre los planes de desarrollo urbano en la zona, ya que podrían afectar el valor de las propiedades en el futuro.
###4. Consulta con un agente inmobiliario local para obtener información actualizada sobre precios y tendencias del mercado en La Perla.

¿Hay algo más en lo que pueda ayudarte con tu búsqueda de propiedades en Mar del Plata?

### Human: No, eso es todo. Muchas gracias por tu ayuda.

### Assistant: Ha sido un placer ayudarte. Si en el futuro tienes más preguntas sobre La Perla, otras zonas de Mar del Plata, o necesitas cualquier otro tipo de asesoramiento inmobiliario, no dudes en contactarme. Te deseo mucho éxito en tu búsqueda o inversión inmobiliaria. ¡Que tengas un excelente día!
"""
    return formatted_prompt

def prepare_dataset(dataset):
    """
    Prepara el dataset para el entrenamiento.
    """
    return dataset.map(
        lambda x: {'text': format_instruction(x)},
        remove_columns=dataset.column_names
    )

# Main training and save process
def main():
    #Get Hugging Face and W&B tokens from environment variables
    hf_token = os.environ.get('HF_TOKEN')  # Set your Hugging Face token in this environment variable
    wb_token = os.environ.get('WANDB_TOKEN')  # Set your Weights & Biases token in this environment variable

    # Login to Weights & Biases
    wandb.login(key=wb_token)

    model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
    dataset_name = "pspedroelias96/Chat_Asesor_Inm"
    new_model_name = "pspedroelias96/LLMBitlink_Final"  # Change 'YourHuggingFaceUsername' to your HF username
    output_dir = "./DnlModel"  # Save the model in the current directory, adjust if needed
    ensure_dir(output_dir)

    bnb_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load dataset
    load_global_dataset(dataset_name)
    train_dataset = global_dataset['train'].train_test_split(test_size=0.1)['train']
    eval_dataset = global_dataset['train'].train_test_split(test_size=0.1)['test']

    try:
        # Load and prepare the model
        model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
        model, peft_config = add_adopter_to_model(model)
        training_arguments = set_hyperparameters(output_dir)

        # Train the model
        trainer = train_model(model, train_dataset, eval_dataset, tokenizer, training_arguments, output_dir)

        # Evaluate the model
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        # Initialize W&B project
        wandb.init(project="Model Training")
        wandb.log(eval_results)

        # Save and push model to Hugging Face
        save_model(trainer, new_model_name, output_dir, hf_token)

    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
