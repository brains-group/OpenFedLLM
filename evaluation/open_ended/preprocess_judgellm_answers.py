import argparse
import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_answer_path", type=str, default=None)
args = parser.parse_args()

response_path = args.model_answer_path
save_path = response_path[:-6] + "_judgellm.jsonl"


def save_jsonl(data_list, file_path):
    with open(file_path, "w") as file:
        for item in data_list:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


# ============= Load the model outputs =============
model_outputs = []
with open(response_path, "r") as ques_file:
    for line in ques_file:
        if line:
            model_outputs.append(json.loads(line))

for model_output in model_outputs:
    # model_output["question_body"] =
    model_output["decoding_method"] = "top_p_sampling"
    model_output["model"] = "gpt-4"
    model_output["reference"] = {}
    model_output["reference"]["text"] = model_output.pop("choices")[0]["turns"][0]
    model_output["scores"] = {}

save_jsonl(model_outputs, save_path)
