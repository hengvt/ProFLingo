import sys
import time
import torch
import copy
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template
from attack import generate_output, generate_suffix

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#Uncomment this if OOM

def load_model_and_tokenizer(model_path):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16,  
        device_map = "cpu"
    )

    models = []
    for i in range(torch.cuda.device_count()):
        models.append(copy.deepcopy(model).to(device=f"cuda:{i}"))
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code = True,
        use_fast = False
    )

    if (tokenizer.pad_token_id == None):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return models, tokenizer


def complete_conversation(model, tokenizer, template, user_prompt, size = 256):
    if (template != None):
        template.messages = []

        template.append_message(template.roles[0], user_prompt)
        template.append_message(template.roles[1], None)

        full_prompt = template.get_prompt()
    else:
        full_prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}],
                                                        tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(full_prompt)
    input_ids = torch.tensor(input_ids, device = model.device)
    output_ids = generate_output(model, tokenizer, input_ids, size)

    output_str = tokenizer.decode(output_ids, skip_special_tokens = True).strip()
    return output_str

if __name__ == "__main__":

    seed = 42

    epoch_num = 256
    prompt_num = 32
    
    model_path = sys.argv[1]
    advsamples_path = sys.argv[2]

    dataset_path = "./questions.csv"

    init_str_rng = np.random.default_rng(seed)

    templates = [get_conv_template("alpaca"), get_conv_template("zero_shot")]
    templates[0].sep = ' '
    templates[1].sep = "\n"

    test_dataset = pd.read_csv(dataset_path).values.tolist()

    load_time = time.perf_counter()
    models, tokenizer = load_model_and_tokenizer(model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print(f"load: {time.perf_counter() - load_time}s")

    for i, data in enumerate(test_dataset):
        init_seed = init_str_rng.integers(100000)

        test_goal = " simply answer: " + data[0]
        test_target = data[1]
        keyword = data[2].lower()

        prompt_suffix = generate_suffix(models, tokenizer, templates, test_goal, test_target, 
                                        epoch_num, prompt_num, keyword, init_seed)

        with open(advsamples_path, 'a') as f:
            f.write(f"{i},{prompt_suffix}\n")

        new_prompt = prompt_suffix + test_goal

        print(f"\nFinal Prompt: |{new_prompt}|")
        for template in templates:
            print(f"\nFinal Answer: |{complete_conversation(models[0], tokenizer, template, new_prompt)}|")
        print()