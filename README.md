<p align="center">
<img width="300" height="300" alt="17" src="https://github.com/user-attachments/assets/2996f973-d7da-4252-996e-a35855098647" />
<img width="300" height="300" alt="30" src="https://github.com/user-attachments/assets/6d262dbd-355e-4e82-b9de-b188cc5303d7" />
</p>

# Visualization-Based Automated Malware Classification: A Replication Study Transcended Towards Malware Detection

Reliably and robustly detecting malware is a critical cornerstone of security.
Superseding earlier, e.g., signature-based approaches, AI-based ones are now increasingly prevalent in this domain. 
One concept which has -- due to its inherent resilience against common obfuscation techniques -- recently gained in popularity is the classification of malware through Convolutional Neural Networks (CNNs) operating on two-dimensional visualizations of binaries. While this approach demonstrated promising results in previous research, existing evaluations chronically overlook the impacts of temporality such as concept drift -- the performance degradation which occurs with a shift in the inferred data departing it from older training data.

To close this gap, we herein first reproduce a state-of-the-art CNN architecture for visualization-based malware classification. To confirm the approach's efficacy and establish a performance baseline, we extend it to fit another established dataset comprising malware and benign software covering two years overall. On that basis, we then evaluate how strongly the model is affected by temporal dependencies between train and test samples in two directions, thus training on old and testing on new samples and vice versa. 

We find that concept drift causes a statistically significant degradation in malware classification performance in the considered state-of-the-art approach. Notably, this concept drift also arises in backward order when classifying older samples through a model trained on more recent ones. Insofar, the previously reported promising results of the approach need to be taken with care.

We provide this repository to allow for easy reproduction of our results.

## Instructions
This repository contains two major parts:
1. Models and Experiments: The code needed to train and validate both our model as well as the reproduction of Pant and Bista's model. This is split into four distinct experiments: `reproduction`, `temporal baseline`, `forward`, `backward`.
2. Visualization: The Jupyter Notebooks used to generate the graphs seen in the paper.

To begin the reproduction process make sure *uv* is installed and run `uv run main.py`. All dependencies will be installed for you, except for the datasets. You will be guided through dataset installation and image generation and asked to choose which experiments you would like to reproduce.
