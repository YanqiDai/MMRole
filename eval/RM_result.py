import argparse
import json
from tqdm import tqdm
import os
import csv


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="eval/mini_reviews/mmrole-agent")
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="eval/mini_results/mmrole-agent")
args = parser.parse_args()


input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
input_files = os.listdir(input_dir) if args.input_file is None else [args.input_file]

score_keys = ["Instruction Adherence", "Fluency", "Coherency", "Image-Text Relevance", "Response Accuracy", "Personality Consistency", "Knowledge Consistency", "Tone Consistency"]

for input_file in input_files:
    if input_file.endswith(".json"):
        # print(f"Processing {os.path.join(input_dir, input_file)}")
        result_file = os.path.join(output_dir, input_file.replace(".json", ".csv"))
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        # assert not os.path.exists(result_file), f"{result_file} already exists."
        with open(os.path.join(input_dir, input_file), "r") as f:
            data_list = json.load(f)

        sum_scores = {key: 0.0 for key in score_keys}
        count = 0
        
        for data in data_list:
            rm_review = data.get("rm_review", {})
            if rm_review:
                for score_key in score_keys:
                    score_pair = rm_review[score_key]["score"]
                    assert len(score_pair) == 2, f"Invalid score pair: {rm_review}"
                    score = score_pair[0] / score_pair[1]
                    sum_scores[score_key] += score
                count += 1
            else:
                print(f"Warning: No RM review for {data['id']}")
        
        avg_scores = {key: sum_scores[key] / count for key in sum_scores}
        
        if count != 72 and count != 26:
            print(f"Warning: {count} reviews for {os.path.join(input_dir, input_file)}")

        with open(result_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["count"] + score_keys)
            writer.writerow([count] + [avg_scores[key] for key in score_keys])