The codes are adopted from [here](https://github.com/26hzhang/VSLNet/tree/master/prepare)

# Extract Features

- We use the pre-trained 3D ConvNets ([here](https://github.com/piergiaj/pytorch-i3d)) to prepare the visual features, the 
extraction codes are placed in this folder. Please download the pre-trained weights [`rgb_charades.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt) and [`rgb_imagenet.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt). 
- The pre-trained GloVe embedding is available at [here](https://nlp.stanford.edu/projects/glove/), please download
`glove.840B.300d.zip`, unzip and put it under `data/` folder.

## Charades STA
The train/test datasets of Charades-STA are available at [[jiyanggao/TALL]](https://github.com/jiyanggao/TALL) 
([`charades_sta_train.txt`](https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view) and 
[`charades_sta_test.txt`](https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view)).

The `charades.json` file is required ([here](https://github.com/piergiaj/super-events-cvpr18/blob/master/data/charades.json)), 
which contains the video length information. Download and place it into the same directory of the train/test datasets.

The videos/images for Charades-STA dataset is available at [here](https://allenai.org/plato/charades/), please download 
either `RGB frames at 24fps (76 GB)` (image frames) or `Data (original size) (55 GB)` (videos). For the second one, the 
extractor will automatically decompose the video into images.
```shell script
# download RGB frames
wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar
# or, download videos
wget http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip
```

The I3D model destroys the spatial structure of a 2D image by computing an average-pooling at the final layer. We, however, want to
preserve the spatial structure as we want to apply frame-level spatial attention. Therefore, we modify the average-pooling kernel in
the original code. The extracted features would have the shape \[Time, Height, Width, Channels\] = \[Time, 7, 7, 1024\]

Extracting features using the below command (i.e., fps:24, kernel_size:24, strides:24), one feature vector represents one second in the video. Namely, a 20 second video would yield 20 feature vectors. 

You can modify the sampling density of the features by modifying `--fps`, `--kernel_size` and `--strides`.

Extract visual features for Charades-STA:
```shell script
# use the weights fine-tuned on Charades or the weights pre-trained on ImageNet
python3 extract_charades.py --use_finetuned --load_model <path to>/rgb_charades.pt  \  # rgb_imagenet.pt
      --video_dir <path to video dir>  \
      --dataset_dir <path to charades-sta dataset dir>  \
      --images_dir <path to images dir>  \  # if images not exist, decompose video into images
      --save_dir <path to save extracted visual features>  \
      --fps 24 --kernel_size 24 --strides 24 --remove_images  # whether remove extract images to release space
```