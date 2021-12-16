# VRDL_HW3

## Installation
Using virtual environment is recommended.<br>
My CUDA version is 11.0<br>
Set the environment by the following commands if your CUDA version is the same as mine.
```
virtualenv .
source bin/activate
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip3 install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install opencv-python
pip3 install shapely
```
## Reproducing Submission
To reproduce my submission without retraining, do the following steps:
1. download weights from https://drive.google.com/file/d/1YFHrQCbMkLj4d0Fc29rgq4K55lpEn7_N/view?usp=sharing
2. put weights in output folder
3. run inference.py

## Training
1. run train_images/tococo.py
2. run train.py
