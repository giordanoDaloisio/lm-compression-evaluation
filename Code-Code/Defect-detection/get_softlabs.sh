#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/softlabs_%j.out
#SBATCH -J def_softlabs
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

base_model=microsoft/codebert-base
model_type=roberta
output_dir=./saved_models_distil_ase

cd code
srun python run.py \
    --output_dir=$output_dir \
    --model_type=$model_type \
    --tokenizer_name=$base_model \
    --model_name_or_path=$base_model \
    --do_eval \
    --train_data_file=../dataset/train_label.jsonl \
    --eval_data_file=../dataset/train_unlabel.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --job_id $SLURM_JOB_ID \
    --seed 123456 2>&1 \
    "$@" | tee test.log