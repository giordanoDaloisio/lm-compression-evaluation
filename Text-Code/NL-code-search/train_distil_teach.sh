#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/train_teach_%j.out
#SBATCH -J tc_train
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

cd code
OUTPUTDIR=./saved_models_distil_ase3
PRETRAINDIR=microsoft/codebert-base    # will download pre-trained CodeGPT model
LOGFILE=text2code_concode.log
PER_NODE_GPU=1       # modify YOUR_GPU_NUM
MODEL=roberta


srun python run.py \
    --output_dir=$OUTPUTDIR \
    --model_type=$MODEL \
    --config_name=$PRETRAINDIR \
    --model_name_or_path=$PRETRAINDIR \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train_teach.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log