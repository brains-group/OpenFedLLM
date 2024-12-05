import argparse
import runpy
import os
import sys
import subprocess
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dpo_model_path", type=str, default=None)
parser.add_argument("--kto_model_path", type=str, default=None)
parser.add_argument("--kto_dist_model_path", type=str, default=None)
args = parser.parse_args()

dpo_model_path = args.dpo_model_path
if dpo_model_path[-1] == "/":
    dpo_model_path = dpo_model_path[:-1]
kto_model_path = args.kto_model_path
if kto_model_path[-1] == "/":
    kto_model_path = kto_model_path[:-1]
kto_dist_model_path = args.kto_dist_model_path
if kto_dist_model_path[-1] == "/":
    kto_dist_model_path = kto_dist_model_path[:-1]

pre_str, last_str = os.path.split(dpo_model_path)
if last_str.startswith("full"):  # if the model is merged as full model
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = last_str.split("-")[-1]
    dpo_model_name = f"{exp_name}_{checkpoint_id}"
else:
    dpo_model_name = last_str
pre_str, last_str = os.path.split(kto_model_path)
if last_str.startswith("full"):  # if the model is merged as full model
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = last_str.split("-")[-1]
    kto_model_name = f"{exp_name}_{checkpoint_id}"
else:
    kto_model_name = last_str
pre_str, last_str = os.path.split(kto_dist_model_path)
if last_str.startswith("full"):  # if the model is merged as full model
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = last_str.split("-")[-1]
    kto_dist_model_name = f"{exp_name}_{checkpoint_id}"
else:
    kto_dist_model_name = last_str

# # advbench dpo
# sys.argv = ["", "--base_model_path", dpo_model_path, "--bench_name", "advbench"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = ["", "--model_answer", dpo_model_name]
# runpy.run_path("./gen_judge_advbench.py")

# # advbench kto
# sys.argv = ["", "--base_model_path", kto_model_path, "--bench_name", "advbench"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = ["", "--model_answer", kto_model_name]
# runpy.run_path("./gen_judge_advbench.py")

# # advbench kto_dist
# sys.argv = ["", "--base_model_path", kto_dist_model_path, "--bench_name", "advbench"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = ["", "--model_answer", kto_dist_model_name]
# runpy.run_path("./gen_judge_advbench.py")

# # Vicuna
# sys.argv = ["", "--base_model_path", dpo_model_path, "--bench_name", "vicuna"]
# runpy.run_path("./gen_model_answer.py")
# sys.argv = ["", "--base_model_path", kto_model_path, "--bench_name", "vicuna"]
# runpy.run_path("./gen_model_answer.py")
# sys.argv = ["", "--base_model_path", kto_dist_model_path, "--bench_name", "vicuna"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = [
#     "",
#     "--ans1_file_path",
#     f"./data/vicuna/model_answer/{dpo_model_name}.jsonl",
#     "--ans2_file_path",
#     f"./data/vicuna/model_answer/{kto_model_name}.jsonl",
#     "--ansmore_file_paths",
#     f"./data/vicuna/model_answer/{kto_dist_model_name}.jsonl",
# ]
# runpy.run_path("./JudgeLM/judgelm/data/JudgeLM/judgelm_preprocess.py", run_name="__main__")

answerFile = f"./data/vicuna/model_judgement/{dpo_model_name}+{kto_model_name}.jsonl"
# sys.argv = [
#     "",
#     "--model-path",
#     "BAAI/JudgeLM-13B-v1.0",
#     "--model-id",
#     "JudgeLM-13B-v1.0",
#     "--question-file",
#     f"./data/vicuna/model_answer/{dpo_model_name}-judgelm-val-5k-judge-samples.jsonl",
#     "--answer-file",
#     answerFile,
#     "--num-gpus-per-model",
#     "1",
#     "--num-gpus-total",
#     "1",
#     "--temperature",
#     "0.2",
#     "--if-fast-eval",
#     "0",
#     "--answer-num",
#     "3",
# ]
# # runpy.run_path("./JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py")
# subprocess.run(
#     f'conda run --no-capture-output -n judgelm python ./JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py {" ".join(sys.argv)}',
#     shell=True,
# )

totalDPOScore = 0
totalKTOScore = 0
totalKTODistScore = 0
totalAnswers = 0
with open(answerFile, "r") as ques_file:
    for line in ques_file:
        if line:
            judgement = json.loads(line)["pred_text"]
            scores = re.findall(r"\d+[.]?\d*", judgement)
            if not re.search(r"^\d+[.]?\d*", judgement) or len(scores) < 3:
                continue
            totalDPOScore += float(scores[0])
            totalKTOScore += float(scores[1])
            totalKTODistScore += float(scores[2])
            totalAnswers += 1
print(f"Total Successful Vicuna Judgements: {totalAnswers}")
print(f"DPO Vicuna Score: {totalDPOScore/totalAnswers}")
print(f"KTO Vicuna Score: {totalKTOScore/totalAnswers}")
print(f"KTO Dist Vicuna Score: {totalKTODistScore/totalAnswers}")

# MT-Bench-1
sys.argv = ["", "--base_model_path", dpo_model_path, "--template", "alpaca"]
runpy.run_path("./gen_model_answer_mt.py")
sys.argv = ["", "--base_model_path", kto_model_path, "--template", "alpaca"]
runpy.run_path("./gen_model_answer_mt.py")
sys.argv = ["", "--base_model_path", kto_dist_model_path, "--template", "alpaca"]
runpy.run_path("./gen_model_answer_mt.py")

sys.argv = [
    "",
    "--ans1_file_path",
    f"./data/mtbench/model_answer/{dpo_model_name}.jsonl",
    "--ans2_file_path",
    f"./data/mtbench/model_answer/{kto_model_name}.jsonl",
    "--ansmore_file_paths",
    f"./data/mtbench/model_answer/{kto_dist_model_name}.jsonl",
]
runpy.run_path("./JudgeLM/judgelm/data/JudgeLM/judgelm_preprocess.py", run_name="__main__")

answerFile = f"./data/mtbench/model_judgement/{dpo_model_name}+{kto_model_name}.jsonl"
sys.argv = [
    "",
    "--model-path",
    "BAAI/JudgeLM-13B-v1.0",
    "--model-id",
    "JudgeLM-13B-v1.0",
    "--question-file",
    f"./data/mtbench/model_answer/{dpo_model_name}-judgelm-val-5k-judge-samples.jsonl",
    "--answer-file",
    answerFile,
    "--num-gpus-per-model",
    "1",
    # "--reference-file",
    # "./data/mtbench/reference_answer/gpt-4_judgellm.jsonl",
    "--num-gpus-total",
    "1",
    "--temperature",
    "0.2",
    "--if-fast-eval",
    "0",
    "--answer-num",
    "3",
]
# runpy.run_path("./JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py")
subprocess.run(
    f'conda run --no-capture-output -n judgelm python ./JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py {" ".join(sys.argv)}',
    shell=True,
)

totalDPOScore = 0
totalKTOScore = 0
totalKTODistScore = 0
totalAnswers = 0
with open(answerFile, "r") as ques_file:
    for line in ques_file:
        if line:
            judgement = json.loads(line)["pred_text"]
            scores = re.findall(r"\d+[.]?\d*", judgement)
            if not re.search(r"^\d+[.]?\d*", judgement) or len(scores) < 3:
                continue
            totalDPOScore += float(scores[0])
            totalKTOScore += float(scores[1])
            totalKTODistScore += float(scores[2])
            totalAnswers += 1
print(f"Total Successful MT-Bench-1 Judgements: {totalAnswers}")
print(f"DPO MT-Bench-1 Score: {totalDPOScore/totalAnswers}")
print(f"KTO MT-Bench-1 Score: {totalKTOScore/totalAnswers}")
print(f"KTO Dist MT-Bench-1 Score: {totalKTODistScore/totalAnswers}")
