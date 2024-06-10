import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "/home/ubuntu/models/Llama-2-7b-hf",
        "/home/ubuntu/models/Llama-2-7b-hf"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "/home/ubuntu/models/Llama-2-7b-hf",
        "/home/ubuntu/models/Llama-2-7b-hf"
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["alpaca", "zero_shot"]
    config.devices = ["cuda:0", "cuda:0"]

    return config
