# Image Caption with CNN+RNN/LSTM/Tranformer

> This is the final project of *Neural Network & Deep Learning* course (SC4001), NTU.  We evaluated the RNN, LSTM, and Transformer decoders under a consistent configuration and experimented with various hyperparameter adjustments to enhance results. In particular, we integrated the adaptive attention mechanism into our framework as an advanced technique. Lastly, we visualized the caption-generation process to demonstrate the impact of attention mechanisms and conducted comprehensive automatic evaluations across the models.


## How to use

### 1. Create conda environment
```
conda create -n ic python=3.12
conda activate ic
pip install h5py tqdm imageio nltk scikit-image matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install openjdk
```

### 2. Prepare the data
Clone this repository.
```bash
git clone https://github.com/Evan-Sukhoi/Image-Caption-SC4001.git
cd Image-Caption-SC4001
```
Download and unzip the dataset files.
```bash
curl -o ./dataset/train2014.zip http://images.cocodataset.org/zips/train2014.zip &
curl -o ./dataset/val2014.zip http://images.cocodataset.org/zips/val2014.zip &
curl -o ./dataset/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip &
curl -o ./dataset/caption_dataset/caption_datasets.zip http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

unzip ./dataset/glove.6B.zip -d ./dataset/Glove 
unzip ./dataset/train2014.zip -d ./dataset
unzip ./dataset/val2014.zip -d ./dataset
unzip ./dataset/caption_dataset/caption_datasets.zip -d ./dataset/caption_dataset
```
Use tools to create input data files.

> Note: The `max_len` parameter specifies the maximum number of words allowed in each split caption. Here we use `18` for MSCOCO and `22` for Flickr30k according to the previous work.
```bash
# MSCOCO 2014
python create_input_files.py --dataset="coco" --max_len=18 --karpathy_json_path="./dataset/caption_dataset/dataset_coco.json" --image_folder="./dataset" --output_folder="./dataset/generated_data" &

# Flickr30k
python create_input_files.py --dataset="flickr30k" --max_len=22 --karpathy_json_path="./dataset/caption_dataset/dataset_flickr30k.json" --image_folder="./dataset" --output_folder="./dataset/generated_data" 
```
### 3. Train the model
Train `Transformer` using default parameters:
```bash
python -u train.py --mode="transformer"
```
Train `LSTM` using default parameters:
```bash
# Soft attention
python -u train.py --mode="lstm" --attention_type="soft"
# Adaptive attention
python -u train.py --mode="lstm" --attention_type="adaptive"
# Without attention
python -u train.py --mode="lstm-wo-attn"
```
Train `RNN without attention` using default parameters:
```bash
python -u train.py --mode="rnn" --attention_type="none"
```
