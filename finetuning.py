import torch
import peft
import trl
import datasets
import sys
from transformers import  AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  
        device_map = "cuda:0",
        quantization_config= BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if (tokenizer.pad_token_id == None):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def formatting_prompts_func(data_point):
    output_text = []
    for data in data_point["conversations"]:
        instruction = None
        response = None
        for conversation in data:
            if (conversation["from"] == "human"):
                instruction = conversation["value"]
            elif (conversation["from"] == "gpt"):
                response = conversation["value"]

        assert instruction != None and response != None

        text = f"A chat between a human and a helpful, respectful, and honest AI.\nHuman: {instruction}\nAI: {response}"
        output_text.append(text)

    return output_text

def finetune(model, tokenizer, dataset_path, output_path):   
    model.train()

    dataset = datasets.load_dataset(dataset_path, split="train")
    dataset = dataset.shuffle(0)

    peft_params = peft.LoraConfig(
        r = 64,
        lora_alpha = 16,
        lora_dropout = 0.1,
        target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
        bias = "none",
        task_type = "CAUSAL_LM",
    )

    training_params = TrainingArguments(
        max_steps = 30000,
        output_dir = output_path,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        lr_scheduler_type = "constant",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        optim = "paged_adamw_32bit",
        fp16 = True,
        save_steps = 100,
        logging_steps = 100,
    )

    trainer = trl.SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        peft_config = peft_params,
        args = training_params,
        max_seq_length = 256,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()

if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    model, tokenizer = load_model_and_tokenizer(model_path)

    finetune(model, tokenizer, dataset_path, output_path)
    
    