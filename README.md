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

#### 3.1 Choose model
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
Train `RNN` using default parameters:
```bash
# Without attention
python -u train.py --mode="rnn" --attention_type="none"
```
#### 3.2 Choose Parameters
The default dataset `train.py` uses is `coco` (MSCOCO2014). To change it for `flickr30k` or `flickr8k`, please use `--data_name` option. Examples:
```bash
python -u train.py --mode="lstm" --data_name=flickr30k_5_cap_per_img_5_min_word_freq
```
To fine-tune the CNN encoder and begin at specific epochs, use `--fine_tune_encoder=True` and `--fine_tune_encoder_start_epoch` options:
```bash
python -u train.py --mode="lstm" --fine_tune_encoder=True --fine_tune_encoder_start_epoch=10 --epochs=15 --data_name=flickr30k_5_cap_per_img_5_min_word_freq
```
To change the number of layers in encoder or decoder of Transformer, please use `encoder_layers` (default: 2) and `--decoder_layers` (default: 6) options.

The learning rates of encoder and decoder can be modified by `encoder_lr` (default: 1e-4) and `decoder_lr` (default: 1e-4) options.

### 4. Evaluate the model

To run `eval.py`, you must give the parameter of `--max_encode_length` and `--checkpoint`, which are corresponding to the dataset you are using, and the `--decoder_mode` of your model.
```bash
python eval.py --decoder_mode="transformer" --max_encode_length=18 --checkpoint="./models/BEST_checkpoint_epoch15_transformer_coco_5_cap_per_img_5_min_word_freq_.pth.tar" --beam_size=3
```
In this case, we use `--max_encode_length=18` becasue we limit maximum words to 18 of the COCO dataset as mentioned before.

### 5. Captioning
The parameters are similar with those of evaluation.
```bash
python caption.py --max_decoder_length=18 --img="./dataset/val2014/COCO_val2014_000000581886.jpg" --decoder_mode="transformer" --beam_size=3 --checkpoint="./models/BEST_checkpoint_transformer_e4d8_coco_5_cap_per_img_5_min_word_freq_.pth.tar"
```
