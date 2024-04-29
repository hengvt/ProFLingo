# ProFLingo

This is the official repository for `ProFLingo: A Fingerprinting-based Copyright Protection Scheme for Large Language Models`

## Installation
```
pip install -r requirements.txt
```
Note: The verification of a few models may require an earlier version of `transformers`, e.g. `transformers==4.38.2`

## Experiments

- To perform verification using generated adversarial examples:
  ```
  python3 copyright_test.py <AE_LIST_PATH> <MODEL_PATH>
  e.g. python3 copyright_test.py ./generated_llama.txt ./models/vicuna-7b-v1.5/
  ```
- To generate adversarial examples
  ```
  python3 proflingo.py <MODEL_PATH> <OUTPUT_AE_LIST_PATH>
  e.g. python3 proflingo.py ./models/Llama-2-7b-hf/ ./generated_llama.txt
  ```
- To reproduce fine-tuning experiments of llama-2-7b
  ```
  git clone https://huggingface.co/datasets/teknium/OpenHermes-2.5
  python3 finetuning.py ./models/Llama-2-7b-hf/ ./OpenHermes-2.5/  ./outputs/
  ```

  
## Models
Please download the necessary models for experiments from Hugging Face:

Llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b

Llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf

Llama-2-7b-chat: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Vicuna-7b-v1.5: https://huggingface.co/lmsys/vicuna-7b-v1.5

ELYZA-japanese-Llama-2-7b-instruct: https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct

Llama2-Chinese-7b-Chat: https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat

Llama-2-7b-ft-instruct-es: https://huggingface.co/clibrain/Llama-2-7b-ft-instruct-es

Meditron-7B: https://huggingface.co/epfl-llm/meditron-7b

Orca-2-7b: https://huggingface.co/microsoft/Orca-2-7b

CodeLlama-7b: https://huggingface.co/meta-llama/CodeLlama-7b-hf

Mistral-7B-v0.1: https://huggingface.co/mistralai/Mistral-7B-v0.1

Mistral-7B-v0.2: https://huggingface.co/mistral-community/Mistral-7B-v0.2

Mistral-7B-Instruct-v0.1: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

OpenHermes-2.5-Mistral-7B: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B

Dolphin-2.2.1-mistral-7b: https://huggingface.co/cognitivecomputations/dolphin-2.2.1-mistral-7b

Code-Mistral-7B: https://huggingface.co/ajibawa-2023/Code-Mistral-7B

Hyperion-2.0-Mistral-7B: https://huggingface.co/Locutusque/Hyperion-2.0-Mistral-7B

Hermes-2-Pro-Mistral-7B: https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B

Mistral-7B-OpenOrca: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca

Starling-LM-7B-alpha: https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha

ChatGLM3-6B: https://huggingface.co/THUDM/chatglm3-6b

Gemma-7b-it: https://huggingface.co/google/gemma-7b-it

Phi-2: https://huggingface.co/microsoft/phi-2

OLMo-7B-Instruct: https://huggingface.co/allenai/OLMo-7B-Instruct

Yi-6B-Chat: https://huggingface.co/01-ai/Yi-6B-Chat

## Reproducibility
All experiments were run on a machine with a single NVIDIA A10G GPU, which has 24G of GPU memory.

Our implementation supports multi-GPU AE generation. Simply run proflingo.py on a machine with multiple GPUs to accelerate the generation process.

To verify models other than those we tested, modify the `get_template` function in the `copyright_test.py` file.

## License
`ProFLingo` is licensed under the terms of the MIT license. See LICENSE for more details.
