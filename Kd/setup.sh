#! /bin/bash
ssh -i ./id_rsa  u12763@login-mdc.hpc.gwdg.de 
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate
mkdir -p /scratch1/users/u12763/Knowledge-distillation/
cd /scratch1/users/u12763/Knowledge-distillation/
conda create -n env python=3.11 -y
conda activate env
pip install  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install torch-scatter torch-sparse torch-scatter torch-geometric ogb  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html --no-cache-dir
pip install numpy matplotlib scipy pandas
