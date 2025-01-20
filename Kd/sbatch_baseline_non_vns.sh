#!/bin/bash
#SBATCH --job-name=graph_ml_sarwar_gine_baseline_kd_non_vns
#SBATCH --partition=gpu
#SBATCH -G RTX5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4000M
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/scratch1/users/u12763/Knowledge-distillation/logs/job_output_graph_ml_sarwar_gine_baseline_kd_non_vns.txt
#SBATCH --mail-user=syed.sarwar@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.1.1

source $HOME/conda/bin/activate env

echo "==========================================="
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo "==========================================="
export PYTHONUNBUFFERED=TRUE
python baseline.py --virtual_node false
