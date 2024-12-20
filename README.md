
# Signal Quality
Collection of tools to analyze ECG, Plethysmography, and general time-series signal quality

[**📘 Documentation**](https://chufangao.github.io/signal_quality/)  
[📁 Github](https://github.com/chufangao/signal_quality/)  
[🗒️ Example Notebooks](https://github.com/chufangao/signal_quality/tree/main/signal_quality/examples/)  

### Main Submodules
- **datasets** - Code to load example ECG signal quality datasets, including the The PhysioNet/Computing in Cardiology Challenge 2011 Dataset and the MIT-BIH Arrhythmia Database.
- **sqis** - Functions that calculate signal quality indicies on time-series signals. Most implementations are focused on ECG quality, as it is a common area of research.
- **featurization** - Contains functions that calculate geometric waveform features of Pleth and ECG.

### Required packages (Python 3.9)
    antropy==0.1.4
    biosppy==0.8.0
    emd==0.5.4
    matplotlib==3.5.1
    neurokit2==0.1.7
    numpy==1.21.5
    pyhrv==0.4.0
    PyWavelets==1.3.0
    scikit_learn==1.1.3
    scipy==1.7.3
    torch==1.11.0
    tqdm==4.62.3
    wfdb==3.4.1

#### Tested on:
    - Apple Mac M1 Pro
    - nVidia Jetson Orin NX 16GB 

### Installation:
```shell
conda create --name signal-quality-challenge python=3.8
conda activate signal-quality-challenge

conda install -c conda-forge ipywidgets
conda install anaconda::scikit-learn
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install antropy emd matplotlib neurokit2 numpy PyWavelets scipy tqdm wfdb
pip install pyhrv biosppy heartpy
```

### Prepare datasets:
```shell
mkdir -p data/mitdb
pushd data/mitdb
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ .
popd

mkdir -p data/nstdb
pushd data/nstdb
aws s3 sync --no-sign-request s3://physionet-open/nstdb/1.0.0/ .
popd

mkdir -p data/PICC
pushd data/PICC
aws s3 sync --no-sign-request s3://physionet-open/challenge-2011/1.0.0/ .
popd
```

### Working components:
```shell
MyWorks/
  - quality.ipynb       # Train and evaluate model on different combinations of features, extracted from ECG
  - quality.py
  - ecg_denoising.ipynb # Filtering main types of signal noise
```
