import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import PeftModel, PeftConfig

# Reemplaza con tu token de Hugging Face
HUGGING_FACE_TOKEN = "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"

def authenticate():
    # Iniciar sesión en Hugging Face
    login(HUGGING_FACE_TOKEN)

def load_model_and_tokenizer(model_name):
    print(f"Cargando el modelo {model_name}...")
    
    # Cargar la configuración del modelo fine-tuned
    peft_config = PeftConfig.from_pretrained(model_name)
    
    # Cargar el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    
    # Cargar el modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Cargar el modelo fine-tuned
    model = PeftModel.from_pretrained(base_model, model_name)
    
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    # Formatear el prompt como en el fine-tuning
    formatted_prompt = f"### Human: {prompt}\n\n### Assistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    model_name = "pspedroelias96/LLMBitlink_Final"  # Usa el nombre de tu modelo fine-tuned
    
    # Autenticación en Hugging Face
    authenticate()

    # Cargar modelo y tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name)

    print("¡Modelo cargado exitosamente! Puedes empezar a hacer preguntas.")

    while True:
        # Leer entrada del usuario
        user_input = input("\nTú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Terminando la sesión de chat. ¡Adiós!")
            break

        # Generar respuesta
        response = generate_response(model, tokenizer, user_input)
        print(f"Modelo: {response}")

if __name__ == "__main__":
    main()
