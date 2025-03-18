#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate vqmil

CUDA_VISIBLE_DEVICES=0 python main.py --task vqmil --dataset camelyon16 \
    --feature_dir /home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files

conda deactivate