import datasets
import argparse
import json
import sys
import re

sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.template import TEMPLATE_DICT

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
args = parser.parse_args()
print(args)

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):  # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str  # mainly for base model
        exp_name = model_name


# ============= Generate responses =============
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, torch_dtype=torch.float16
).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(
        model, args.lora_path, torch_dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

template = TEMPLATE_DICT[args.template][0]

print(f">> You are using template: {template}")


class Alpaca7B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        instruction = template.format(prompt, "", "")[:-1]

        input_ids = self.tokenizer.encode(
            instruction, return_tensors="pt", max_length=512, truncation=True
        ).to(device)
        output_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_p=1.0,
            temperature=0.7,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # print(
        #     f"----------------------------------{response}------------------------------------------------------"
        # )
        # response = response[(response.find("### Response:") + len("### Response:")) :]
        reponse = re.findall("[ABCDEF]", response)
        if re.search("[ABCDEF]", response):
            # print(f"<<<<<<<<<<<<<<<<<<<<{response}>>>>>>>>>>>>>>>>>>>>>")
            response = reponse[0]
        return response

        # print(
        #     f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{prompt}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        # )

        # model_inputs = self.tokenizer(
        #     [template.format(prompt, "", "")[:-1]],
        #     return_tensors="pt",
        #     max_length=512,
        #     truncation=True,
        # ).to(device)
        # model.to(device)

        # generated_ids = model.generate(
        #     **model_inputs, max_new_tokens=100, do_sample=True
        # )
        # response = self.tokenizer.batch_decode(generated_ids)[0]
        # print(
        #     f"----------------------------------{response}------------------------------------------------------"
        # )
        # response = response[(response.find("### Response:") + len("### Response:")) :]
        # reponse = re.findall("[ABCDEF]", response)
        # if len(response) > 0:
        #     response = reponse[0]
        # return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Alpaca 7B"


testModel = Alpaca7B(model=model, tokenizer=tokenizer)
benchmark = MMLU(n_shots=0)
results = benchmark.evaluate(model=testModel)
print("Overall Score: ", results)
