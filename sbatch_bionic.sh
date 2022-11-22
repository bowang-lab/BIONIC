#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=10:00:00

EPOCHS=$1
LEARNING_RATE=$2
DIM=$3
N_HEADS=$4
N_LAYERS=$5
LAMBDA=$6
FOLD=$7

cd /scratch/gobi1/odatskiv/

. bionic_venv.sh

cd /scratch/gobi1/odatskiv/BIONIC/

echo "Running run_bionic.py..."
python run_bionic.py -e $EPOCHS -lr $LEARNING_RATE -d $DIM -gh $N_HEADS -l $N_LAYERS -lmb $LAMBDA -f $FOLD
