import argparse
import json
from tqdm import tqdm
import os
import csv


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--no_random", action="store_true")
parser.add_argument("--model_path", type=str, default="model_weights/MMRole-Eval_RM")
parser.add_argument("--image_dir", type=str, default="images")
parser.add_argument("--input_dir", type=str, default="eval/mini_answers/mmrole-agent")
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="eval/mini_reviews/mmrole-agent")
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
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True).eval()
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


def get_review_score(review):
    try:
        review = review.split("[Scores]:")
        assert len(review) == 2
        review = review[1]
        score_pair = review.strip().split("\n")[0].strip().replace("(", "").replace(")", "")
        score_pair = score_pair.split(",")
        if len(score_pair) != 2:
            raise Exception("Invalid score pair.")
        for j in range(2):
            if ":" in score_pair[j]:
                score_pair[j] = score_pair[j].split(":")[1].strip()
            score_pair[j] = float(score_pair[j].strip())
        return score_pair
    except Exception as e:
        print(f"{e} You must manually fix the score pair.")
        return {}


image_dir = args.image_dir
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
input_files = os.listdir(input_dir) if args.input_file is None else [args.input_file]

for input_file in input_files:
    if input_file.endswith(".json"):
        print(f"Processing {input_file}")
        assert not os.path.exists(os.path.join(output_dir, input_file)), f"{os.path.join(output_dir, input_file)} already exists."
        with open(os.path.join(input_dir, input_file), "r") as f:
            data_list = json.load(f)

        for data in tqdm(data_list):
            image_path = os.path.join(image_dir, data['image'])
            role = data["role"]
            script = data["script"]
            other_role = data["other_role"]
            question = data["conversations"][0]["value"]
            evaluated_answer = data["conversations"][1]["answer"]
            groundtruth_answer = data["conversations"][1]["value"]

            if script == "Hypothetical Characters":
                role_script = role + ", a person who is one of the Hypothetical Characters,"
                if other_role == "a curious human":
                    other_role_script = other_role
                else:
                    other_role_script = other_role + ", a person who is one of the Hypothetical Characters,"
            else:
                role_script = role + " from " + script
                if other_role == "a curious human":
                    other_role_script = other_role
                else:
                    other_role_script = other_role + " from " + script
            the_other_role = other_role if other_role != "a curious human" else "the curious human"

            task = f"The task instruction of the two models is to directly role-play as {role_script} and talk with {other_role_script} about the given image using the distinctive tone, manner and vocabulary of {role}."

            if evaluated_answer:
                system = "You are an objective and precise evaluator, specializing in rigorously assessing the role-playing and multimodal understanding abilities of various models."
                aspect_desc = {
                    "Instruction Adherence": f"Instruction Adherence: Do the responses accurately adhere to the task instruction, directly role-playing as {role} and only including words that {role} should say, without any additional explanatory prefixes or suffixes?",
                    "Fluency": "Fluency: Are the responses grammatically correct and smoothly articulated?",
                    "Coherency": "Coherency: Do the responses maintain a coherent thread of dialogue without contradicting earlier parts of the conversation or previously established facts?",
                    "Image-Text Relevance": "Image-Text Relevance: Are the responses closely related to the visual content of the image?",
                    "Response Accuracy": f"Response Accuracy: Do the responses accurately answer {the_other_role}'s words or appropriately initiate a conversation based on the image?",
                    "Personality Consistency": f"Personality Consistency: Do the responses accurately and sufficiently reflect the personality of {role}?",
                    "Knowledge Consistency": f"Knowledge Consistency: Are the responses consistent with the factual knowledge that {role} should possess, including experiences, abilities, and relationships?",
                    "Tone Consistency": f"Tone Consistency: Do the responses maintain a consistent tone that aligns with {role}'s typical manner of speaking and catchphrases, rather than resembling the style of AI assistants?",
                }
                rm_review_dict = {}
                for score_key in aspect_desc:
                    text = (
                        f"## **[Question Start]**\n\n{question}\n\n## **[Question End]**\n\n\n"
                        f"## **[Model A's Response Start]**\n\n{evaluated_answer}\n\n## **[Model A's Response End]**\n\n\n"
                        f"## **[Model B's Response Start]**\n\n{groundtruth_answer}\n\n## **[Model B's Response End]**\n\n\n"
                        "## **[Instruction]**\n\n"
                        f"{task}\n\n"
                        "Please evaluate the following aspect of each model's response:\n"
                        f"{aspect_desc[score_key]}\n\n"
                        "Please provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from 1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.\n\n"
                        "The output should be in the following format:\n"
                        "{Qualitative Evaluation}, [Scores]: ({the score of Model A}, {the score of Model B})\n\n"
                        "Please ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment."
                    )
                    rm_review = qwen_vl_chat_api(system, text, image_path)
                    rm_score = get_review_score(rm_review)

                    rm_review_dict[score_key] = {
                        "review": rm_review,
                        "score": rm_score
                    }
                data["rm_review"] = rm_review_dict

            with open(os.path.join(output_dir, input_file), "w") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)