#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate vqmil

CUDA_VISIBLE_DEVICES=0 python main.py --task svqmil --dataset tcga \
    --feature_dir /home/coalball/projects/WSI/datasets/tcga_lung_dsmil/features/pt_files

conda deactivate