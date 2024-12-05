import argparse
import runpy
import os
import sys
import subprocess
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

model_path = args.model_path
if model_path[-1] == "/":
    model_path = model_path[:-1]

pre_str, last_str = os.path.split(model_path)
if last_str.startswith("full"):  # if the model is merged as full model
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = last_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    model_name = last_str

# # advbench
# sys.argv = ["", "--base_model_path", model_path, "--bench_name", "advbench"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = ["", "--model_answer", dpo_model_name]
# runpy.run_path("./gen_judge_advbench.py")

# Vicuna
# sys.argv = ["", "--base_model_path", model_path, "--bench_name", "vicuna"]
# runpy.run_path("./gen_model_answer.py")

# sys.argv = [
#     "",
#     "--ans1_file_path",
#     f"./data/vicuna/model_answer/{model_name}.jsonl",
#     "--ans2_file_path",
#     f"./data/vicuna/model_answer/FedChatbotPA_9508_fedavg_dpo_c747s10_i10_b8a1_l512_r8a16_20241112210207_200.jsonl",
#     "--ansmore_file_paths",
#     f"./data/vicuna/model_answer/FedChatbotPA_9508_fedavg_kto_c747s10_i10_b8a1_l512_r8a16_20241112210328_200.jsonl",
# ]
# runpy.run_path(
#     "./JudgeLM/judgelm/data/JudgeLM/judgelm_preprocess.py", run_name="__main__"
# )

answerFile = f"./data/vicuna/model_judgement/{model_name}.jsonl"
# sys.argv = [
#     "",
#     "--model-path",
#     "BAAI/JudgeLM-13B-v1.0",
#     "--model-id",
#     "JudgeLM-13B-v1.0",
#     "--question-file",
#     f"./data/vicuna/model_answer/{model_name}-judgelm-val-5k-judge-samples.jsonl",
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

totalScore = 0
totalScore2 = 0
totalScore3 = 0
totalAnswers = 0
with open(answerFile, "r") as ques_file:
    for line in ques_file:
        if line:
            judgement = json.loads(line)["pred_text"]
            if not re.search(r"^\d+[.]?\d*", judgement):
                continue
            scores = re.findall(r"\d+[.]?\d*", judgement)
            totalScore += float(scores[0])
            totalScore2 += float(scores[1])
            totalScore3 += float(scores[2])
            totalAnswers += 1
print(f"Total Successful Vicuna Judgements: {totalAnswers}")
print(f"Base Vicuna Score: {totalScore/totalAnswers}")
print(f"DPO Vicuna Score 2: {totalScore2/totalAnswers}")
print(f"KTO Vicuna Score 3: {totalScore3/totalAnswers}")

# MT-Bench-1
# sys.argv = ["", "--base_model_path", model_path, "--template", "alpaca"]
# runpy.run_path("./gen_model_answer_mt.py")

# sys.argv = [
#     "",
#     "--ans1_file_path",
#     f"./data/mtbench/model_answer/{model_name}.jsonl",
#     "--ans2_file_path",
#     f"./data/mtbench/model_answer/FedChatbotPA_9508_fedavg_dpo_c747s10_i10_b8a1_l512_r8a16_20241112210207_200.jsonl",
#     "--ansmore_file_paths",
#     f"./data/mtbench/model_answer/FedChatbotPA_9508_fedavg_kto_c747s10_i10_b8a1_l512_r8a16_20241112210328_200.jsonl",
# ]
# runpy.run_path(
#     "./JudgeLM/judgelm/data/JudgeLM/judgelm_preprocess.py", run_name="__main__"
# )

answerFile = f"./data/mtbench/model_judgement/{model_name}.jsonl"
# sys.argv = [
#     "",
#     "--model-path",
#     "BAAI/JudgeLM-13B-v1.0",
#     "--model-id",
#     "JudgeLM-13B-v1.0",
#     "--question-file",
#     f"./data/mtbench/model_answer/{model_name}-judgelm-val-5k-judge-samples.jsonl",
#     "--answer-file",
#     answerFile,
#     "--num-gpus-per-model",
#     "1",
#     # "--reference-file",
#     # "./data/mtbench/reference_answer/gpt-4_judgellm.jsonl",
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

totalScore = 0
totalScore2 = 0
totalScore3 = 0
totalAnswers = 0
with open(answerFile, "r") as ques_file:
    for line in ques_file:
        if line:
            judgement = json.loads(line)["pred_text"]
            if not re.search(r"^\d+[.]?\d*", judgement):
                continue
            scores = re.findall(r"\d+[.]?\d*", judgement)
            totalScore += float(scores[0])
            totalScore2 += float(scores[1])
            totalScore3 += float(scores[2])
            totalAnswers += 1
print(f"Total Successful MT-Bench-1 Judgements: {totalAnswers}")
print(f"Base MT-Bench-1 Score: {totalScore/totalAnswers}")
print(f"DPO MT-Bench-1 Score 2: {totalScore2/totalAnswers}")
print(f"KTO MT-Bench-1 Score 3: {totalScore3/totalAnswers}")
