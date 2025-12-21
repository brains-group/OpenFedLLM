@echo off
REM OpenFedLLM Training Script for Windows CPU Environment
REM This script is optimized for systems without dedicated GPU

echo ========================================
echo OpenFedLLM CPU Training Script
echo ========================================
echo.
echo WARNING: CPU training is significantly slower than GPU training.
echo This configuration uses a small model and reduced dataset for demonstration.
echo.
echo Press Ctrl+C to cancel, or
pause

REM Enable multi-threading for CPU
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8

REM Training parameters (optimized for CPU)
set max_steps=5
set num_rounds=10
set batch_size=2
set gradient_accumulation_steps=4
set seq_length=256
set num_clients=5
set sample_clients=2
set lora_r=8
set lora_alpha=16
set lr=5e-5

REM Dataset configuration (using smaller sample size)
set dataset_name=vicgalle/alpaca-gpt4
set dataset_sample=1000

REM Model configuration (using smaller model for CPU)
REM You can change this to other small models:
REM   facebook/opt-125m (125M parameters - very fast for testing)
REM   facebook/opt-1.3b (1.3B parameters - recommended)
REM   EleutherAI/pythia-1.4b (1.4B parameters)
REM   gpt2-large (774M parameters)
set model_name_or_path=facebook/opt-1.3b

set output_dir=./output
set fed_alg=fedavg

echo.
echo Configuration:
echo   Model: %model_name_or_path%
echo   Dataset: %dataset_name% (sample: %dataset_sample%)
echo   Algorithm: %fed_alg%
echo   Clients: %num_clients% (sample: %sample_clients% per round)
echo   Rounds: %num_rounds%
echo   Steps per round: %max_steps%
echo   Batch size: %batch_size%
echo   CPU threads: %OMP_NUM_THREADS%
echo.

REM Run training
python main_sft.py ^
  --learning_rate %lr% ^
  --model_name_or_path %model_name_or_path% ^
  --dataset_name %dataset_name% ^
  --dataset_sample %dataset_sample% ^
  --fed_alg %fed_alg% ^
  --num_clients %num_clients% ^
  --sample_clients %sample_clients% ^
  --max_steps %max_steps% ^
  --num_rounds %num_rounds% ^
  --batch_size %batch_size% ^
  --gradient_accumulation_steps %gradient_accumulation_steps% ^
  --seq_length %seq_length% ^
  --peft_lora_r %lora_r% ^
  --peft_lora_alpha %lora_alpha% ^
  --use_peft ^
  --output_dir %output_dir% ^
  --template "alpaca"

echo.
echo Training complete! Check %output_dir% for results.
pause
