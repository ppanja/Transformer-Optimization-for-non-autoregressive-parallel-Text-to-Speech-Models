# Transformer Optimization for non-autoregressive parallel Text to Speech Models

This code has used [official Glow TTS code](https://github.com/jaywalnut310/glow-tts) as baseline model for MSc Thesis. Some changes has been done for the experiments and implementation of novel approach for finding importance weights. If you want to use Glow TTS model, I recommend that you refer to the official code.


## 1. Environments used

* Python3.6.9
* pytorch1.2.0
* cython0.29.12
* librosa0.7.1
* numpy1.16.4
* scipy1.3.0

For Mixed-precision training, we use [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4


## 2. Pre-requisites

a.1) Change the name of the repository from "transformer-optimization" to "glow-tts".

a.2) Download and extract the [CMU Arctic dataset,cmu_us_slt_arctic.tar.bz2](http://festvox.org/cmu_arctic/).

a.3) Install sox in windows / Linux system.

a.4) Convert the sample rate of audio files from 16 KHz to 22 KHz using sox.

a.5) Trim the silences from both end of the audio file using sox. 

a.6) Split the files into train, validation and test files.

a.7) Create a DUMMY folder within the project folder "glow-tts" where audio files needto be kept.

a.8) Create three file lists same as the file lists present for baseline model which containthe lists of files for train, validation and test files in the format - "DUMMY/arctic_b0408.wav|You have all the advantage."

b) Initialize WaveGlow submodule: `git submodule init; git submodule update`

Don't forget to download pretrained WaveGlow model and place it into the waveglow folder.

c) Build Monotonic Alignment Search Code (Cython): `cd monotonic_align; python setup.py build_ext --inplace

## 3. Training Example

```sh
sh train_ddi.sh configs/base.json base
```

## 4. Experiment Results

Download pre-trained files from below location and keep in the corresponding folders of "glow-tts/logs/":

glow-tts/logs/base_50_all_head: ## pre-trained file with all heads after 50 epochs
https://drive.google.com/file/d/167YUngqzn1PLfK0PDktgjSqjJutPjAnt/view?usp=sharing

glow-tts/logs/base_final_graph_20: ## pre-trained file with all heads after 20 epochs - Provide index of heads with respect to importance weights in descending order
https://drive.google.com/file/d/118T9lYBSxUf7k9arYw22oqBulemnsfJB/view?usp=sharing

glow-tts/logs/base_final_opti: ## pre-trained file with pruned heads after 50 epochs
https://drive.google.com/file/d/12WP5pQe68bu6rl5a2z7tdvPNhRGxQHGE/view?usp=sharing

See [ExperimentResults.ipynb](./ExperimentResults.ipynb)


## Acknowledgements

This implementation has used the following repo as baseline model:
* [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
