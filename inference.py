import os
import json
from tqdm import tqdm
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
torch.manual_seed(1234)


parser = argparse.ArgumentParser()
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--no_random", action="store_true")
parser.add_argument("--model_path", type=str, default="model_weights/MMRole-Agent")
parser.add_argument("--image_dir", type=str, default="images")
parser.add_argument("--input_dir", type=str, default="data/test/in-distribution")
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="eval/in-test_answers/mmrole-agent")
args = parser.parse_args()

if args.no_random:
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if args.use_lora:
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
if args.no_random:
    model.generation_config.do_sample = False
print(model.generation_config)


def qwen_vl_chat_api(system, question, image_path):
    query = tokenizer.from_list_format([
        {'image': image_path}, # Either a local path or an url
        {'text': question},
    ])
    response, history = model.chat(tokenizer, query=query, history=None, system=system)
    return response


image_dir = args.image_dir
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
input_files = os.listdir(input_dir) if args.input_file is None else [args.input_file]

system = "You are a dedicated role-playing assistant designed to immerse yourself fully in the character you are portraying."
for input_file in input_files:
    if input_file.endswith(".json"):
        print(f"Processing {input_file}")
        assert not os.path.exists(os.path.join(output_dir, input_file)), f"{os.path.join(output_dir, input_file)} already exists."
        with open(os.path.join(input_dir, input_file), "r") as f:
            data_list = json.load(f)
        for data in tqdm(data_list):
            image_path = os.path.join(image_dir, data['image'])
            question = data['conversations'][0]['value']
            answer = qwen_vl_chat_api(system, question, image_path)
            data['conversations'][1]['answer'] = answer
            
            with open(os.path.join(output_dir, input_file), "w") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)