# Deep learning models for single-lead QT interval measurement  
We trained 3 deep learning models for QT interval measurement on single-lead electrocardiograms (ECG). The models were compared to a more classical approach for ECG interval measurement (wavelet-based method) and were found to have a better performance.

## Python packages required (Python 3.8.5)
tensorboard <br> 
jupyterlab <br>
notebook <br>
numpy==1.19.4 <br>
scipy==1.5.4 <br>
scikit-learn==0.23.2 <br>
seaborn <br>
matplotlib <br>
torch===1.7.1 <br>

## Repository description
This repository contains the code used to implement the methods descriped in the paper

### Model architecture and training process
- The architectures of the 3 deeep learning models and the loss functions are defined in [`src/model/net.py`](src/model/net.py).
- The 12-lead ECG recordings and corresponding QT intervals used for training/validation are prepared in [`src/model/data_loader.py`](src/model/data_loader.py). 
- The 5-fold cross-validation is implemented in [`src/train.py`](src/train.py) and [`src/evaluate.py`](src/evaluate.py).

### Wavelet-based algorithm
The implementation of the wavelet-based method can be found in [`src/wavedel/`](src/wavedel/).

### Beat averaging
The beat averaging process is implemented in [`src/qrs_detector.py`](src/qrs_detector.py).



