SVQ-MIL: Small-Cohort Whole Slide Image Classification via Split Vector Quantization
======

SVQ-MIL is a framework that employs split vector quantization (SVQ) with a learnable codebook to quantize instance embeddings. The learned codebook reduces embedding variability and simplifies the MIL model, making it more suitable for small-cohort datasets. SVQ-MIL also enhances model's interpretability, providing an additional profile of the WSI instances through the codebook.

## Installation and Dependencies

```
conda create --name vqmil python=3.12
conda activate vqmil
conda install conda-forge::openslide
pip3 install -r requirements.txt
```

## Train

### Training on Camelyon16

After downloading [Camelyon16](https://camelyon16.grand-challenge.org/Data/) dataset, please refer to the guidelines in [CLAM](https://github.com/mahmoodlab/CLAM) to extract embedding vectors using the following command.

```
python create_patches_fp.py --source $YOUR_WSI_DIR --save_dir $TILES_DIR --patch_level 1 --patch_size 224 --step_size 224 --seg --patch --stitch --use_ostu

python extract_features_fp.py --data_h5_dir $TILES_DIR --data_slide_dir $YOUR_WSI_DIR --csv_path $INFO_CSV --feat_dir $EMBEDDING_DIR --batch_size 512 --slide_ext .tif
```
Then you can get a directory with embedding vectors within .pt files. Note the directory as $EMBEDDING_DIRECTORY_C16

Alternatively, we also provide the extracted embeddings via [HuggingFace](https://huggingface.co/datasets/aCoalBall/SVQMIL-C16-Embeddings) 

Training

```
conda activate vqmil
CUDA_VISIBLE_DEVICES=0 python main.py --task svqmil --dataset camelyon16 \
    --feature_dir $EMBEDDING_DIRECTORY_C16
```


### Training on TCGA Lung Cancer Dataset

Follow the guidelines in [DS-MIL](https://github.com/binli123/dsmil-wsi) to download the dataset. To extract embedding vectors and sample the pseudo bags, using the same way as for Camelyon16.

For training, run
```
conda activate vqmil
CUDA_VISIBLE_DEVICES=0 python main.py --task svqmil --dataset tcga \
    --feature_dir $EMBEDDING_DIRECTORY_TCGA
```


