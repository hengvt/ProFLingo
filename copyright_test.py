import sys
import pandas as pd
import openai
import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template, SeparatorStyle, Conversation
from proflingo import complete_conversation

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if (tokenizer.pad_token_id == None):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def get_template(model_path):
    if "finetuned" in model_path or "checkpoint" in model_path:
        template = Conversation(
            name="finetuned",
            system_message="A chat between a human and a helpful, respectful, and honest AI.",
            roles = ("Human", "AI"),
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
    elif any(x in model_path for x in ["Llama-2-7b-hf", "Llama-2-13b-hf", "Mistral-7B-v0.1", "Mistral-7B-v0.2", "MiniMA-3B", "CodeLlama-7b-hf"]):
        template = get_conv_template("zero_shot")
    elif "Llama-2-7b-chat-hf" in model_path:
        template = get_conv_template("llama-2")
    elif "Llama2-Chinese-7b-Chat" in model_path:
        template = get_conv_template("llama-2")
        template.roles = ("Human", "Assistant")
        template.sep_style = SeparatorStyle.ADD_COLON_SINGLE
    elif "Orca-2-7b" in model_path:
        template = get_conv_template("orca-2")
    elif "ELYZA-japanese-Llama-2-7b-instruct" in model_path:
        template = get_conv_template("llama-2")
        template.system_message = "あなたは誠実で優秀な日本人のアシスタントです。"
    elif "vicuna-7b-v1.5" in model_path:
        template = get_conv_template("vicuna_v1.1")
    elif "llama2_7b_chat_uncensored" in model_path:
        template = get_conv_template("alpaca")
        template.roles = ("### HUMAN", "### Response")
    elif "meditron-7b" in model_path:
        template = get_conv_template("zero_shot")
        template.system_message = "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
        template.roles = ("User", "Assistant")
    elif "Llama-2-7b-ft-instruct-es" in model_path:
        template = get_conv_template("alpaca")
        template.roles = ("### Instrucción", "### Respuesta")
        template.system_message = "A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n"
    elif "dolphin-2.2.1-mistral-7b" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
    elif "Code-Mistral-7B" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
        template.system_message = "You are a helpful AI assistant."
    elif "Hyperion-2.0-Mistral-7B" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
        template.system_message = None
    elif "OpenHermes-2.5-Mistral-7B" in model_path:
        template = get_conv_template("OpenHermes-2.5-Mistral-7B")
    elif "Mistral-7B-OpenOrca" in model_path:
        template = get_conv_template("mistral-7b-openorca")
    elif "Starling-LM-7B-alpha" in model_path:
        template = get_conv_template("openchat_3.5")
    elif "phi-2" in model_path:
        template = get_conv_template("zero_shot")
        template.system_message = ''
        template.roles = ("Instruct", "Output")
        template.sep = '\n'
        template.stop_str = ''
    elif "chatglm3-6b" in model_path:
        template = get_conv_template("chatglm3")
        template.system_message = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    elif "gemma-7b-it" in model_path:
        template = get_conv_template("gemma")
    else:
        raise ValueError("No template available")
    
    return template

def get_answer_openai(client, model, question):
    if "gpt3.5 in model":
        model = "gpt-3.5-turbo-0125"
    elif "gpt4" in model:
        model = "gpt-4-0125-preview"
    response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    temperature = 0,
            )
    answer = response.choices[0].message.content.strip()
    return answer

def get_answer_claude(client, model, question):
    if "claude2" in model:
        model = "claude-2.1"
    elif "claude3" in model:
        model = "claude-3-opus-20240229"
    response = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "user", "content": question}
            ]
        )
    answer = response.content[0].text
    return answer

def fingerprint_test(model, tokenizer, 
                     dataset_path, advsamples_path, 
                     manual_check = True, model_path = "", template = None,  
                     verbose = True, pbar = None, max_token = 64, first_sentence = False):
    advsamples = pd.read_csv(advsamples_path, header = None).values.tolist()
    test_dataset = pd.read_csv(dataset_path).values.tolist()

    questions_counter = 0
    success_counter = 0
    for adv in advsamples:
        q_index = adv[0]
        if (verbose):
            print(f"{test_dataset[q_index][0]} ({test_dataset[q_index][2]})")
        question = adv[1] + " simply answer: " + test_dataset[q_index][0]
        if ("Mistral-7B-OpenOrca" in model_path or "Orca-2-7b" in model_path):
            question += " Directly give me the simple answer. Do not give me step-by-step reason. Do not explain anything further. Do not say any words except the answer."
        if ("gpt" in model_path):
            answer = get_answer_openai(model, model_path, question)
        elif ("claude" in model_path):
            answer = get_answer_claude(model, model_path, question)
        else:
            answer = complete_conversation(model, tokenizer, template, question, size=max_token)
        if (first_sentence):
            answer = answer[:answer.find(".")]
        if (verbose):
            print(f"{answer}")
        questions_counter += 1
        if test_dataset[q_index][2].replace(' ', '').lower() in answer.replace(' ', '').lower():
            if (manual_check):
                success = input("Was the attack successful? Y/N   ")
                if ('y' in success.lower()):
                    success_counter += 1
            else:
                success_counter += 1
        if (verbose):
            print()
        if (pbar):
            pbar.update(1)
    return questions_counter, success_counter

if __name__ == "__main__":

    advsamples_path = sys.argv[1]
    model_path = sys.argv[2]

    dataset_path = r"/home/ubuntu/questions.csv"

    tokenizer_template_lst = ["OLMo-7B-Instruct",
                                "Mistral-7B-Instruct-v0.1",
                                "Hermes-2-Pro-Mistral-7B",
                                "Yi-6B-Chat",]
    
    template = None
    tokenizer = None
    if "gpt" in model_path:
        model = openai.OpenAI(api_key="<API_KEY>")
    elif "claude" in model_path:
        model = anthropic.Anthropic( api_key="<API_KEY>")
    elif any(x in model_path for x in tokenizer_template_lst):
        model, tokenizer = load_model_and_tokenizer(model_path)
    else:
        template = get_template(model_path)
        model, tokenizer = load_model_and_tokenizer(model_path)
        
    questions_counter, success_counter = fingerprint_test(model, tokenizer, dataset_path, advsamples_path, model_path = model_path, template = template)

    print(f"All: {questions_counter} Success: {success_counter} ASR: {success_counter / questions_counter}")

