# Transformer Optimization for non-autoregressive parallel Text to Speech Models

This code is a replication of [official Glow TTS code](https://github.com/jaywalnut310/glow-tts) with some changes required for the experiments and implementation of novel approach for finding importance weights. If you want to use Glow TTS model, I recommend that you refer to the official code.


## 1. Environments used

* Python3.6.9
* pytorch1.2.0
* cython0.29.12
* librosa0.7.1
* numpy1.16.4
* scipy1.3.0

For Mixed-precision training, we use [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4


## 2. Pre-requisites

### Change related to MSc Thesis ###
Replace task mentioned in (a) above with below steps.

a.1) Download and extract the [CMU Arctic dataset,cmu_us_slt_arctic.tar.bz2](http://festvox.org/cmu_arctic/).

a.2) Install sox in windows / Linux system.

a.3) Convert the sample rate of audio files from 16 KHz to 22 KHz using sox.

a.4) Trim the silences from both end of the audio file using sox. 

a.5) Split the files into train, validation and test files.

a.6) Create a DUMMY folder within the project folder "glow-tts" where audio files needto be kept.

a.7) Create three file lists same as the file lists present for baseline model which containthe lists of files for train, validation and test files in the format - "DUMMY/arc-tic_b0408.wav|You have all the advantage."

b) Initialize WaveGlow submodule: `git submodule init; git submodule update`

Don't forget to download pretrained WaveGlow model and place it into the waveglow folder.

c) Build Monotonic Alignment Search Code (Cython): `cd monotonic_align; python setup.py build_ext --inplace

## 3. Training Example

```sh
sh train_ddi.sh configs/base.json base
```

## 4. Inference Example

See [inference.ipynb](./inference.ipynb)


## Acknowledgements

Our implementation is hugely influenced by the following repos:
* [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
