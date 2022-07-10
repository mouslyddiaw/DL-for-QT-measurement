# Deep learning models for single-lead QT interval measurement  
We trained 3 deep learning models for QT interval measurement on single-lead electrocardiograms (ECG). The models were compared to a more classical approach for ECG interval measurement (wavelet-based method) and were found to have a better performance.

## Repository description
This repository contains the code used to implement the methods descriped in the paper.

### Python packages required (Python 3.8.5)
The packages required can be found in `requirements.txt`.

### Model architecture and training process
- The architectures of the 3 deep learning models and the loss functions are defined in [`src/model/net.py`](src/model/net.py).
- The 12-lead ECG recordings and corresponding QT intervals used for training/validation are prepared in [`src/model/data_loader.py`](src/model/data_loader.py). 
- The 5-fold cross-validation is implemented in [`src/train.py`](src/train.py) and [`src/evaluate.py`](src/evaluate.py). For instance, AttnCNN can be trained with the command `python3 train.py --model_dir=experiments/cnn`. 

### Wavelet-based algorithm
The implementation of the wavelet-based method can be found in [`src/compare/wavedel/`](src/compare/wavedel/).

### Beat averaging
The beat averaging process is implemented in [`src/compare/qrs_detector.py`](src/compare/qrs_detector.py).



