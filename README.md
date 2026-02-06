<p align="center">
<img width="300" height="300" alt="17" src="https://github.com/user-attachments/assets/2996f973-d7da-4252-996e-a35855098647" />
<img width="300" height="300" alt="30" src="https://github.com/user-attachments/assets/6d262dbd-355e-4e82-b9de-b188cc5303d7" />
</p> 

# Visualization-Based Automated Malware Classification: A Replication Study Transcended Towards Malware Detection

**Abstract:** Reliably and robustly detecting and classifying malware is a critical cornerstone of security, with AI-based approaches becoming increasingly prevalent. Proposed concepts particularly include the classification of malware through Convolutional Neural Networks (CNNs) operating on two-dimensional visualizations of binaries. As this approach exhibited outstanding results when classifying between malware families, it stands to reason to also employ it for the binary classification between malware and benign samples to perform malware detection. However, in contrast to malware family classification, the detection is subject to concept drift, which might lead to decreasing performance over time. 

Hence, to explore the transferability of visualization-based approaches from malware classification towards binary malware detection, we herein first reproduce a state-of-the-art CNN architecture for classifying malware families. After validation, we transcend the approach towards a binary classification task on another established dataset comprising malware and benign software covering two years overall. On that basis, we then evaluate how strongly the model is affected by temporal dependencies between train and test samples in two directions, thus training on old and testing on new samples and vice versa. 

We find that while the classification model can be reproduced and at first glance also suits detection, concept drift causes a significant degradation in malware detection performance. In particular, this concept drift also affects performance in backward order when classifying older samples through a model trained on more recent ones. Insofar, the visualization-based classification approach cannot be transferred to malware detection tasks without further adjustment.

## Quick-Start

To get this running, you only need one tool: **[uv](https://github.com/astral-sh/uv)**.

Make sure `uv` is installed on your machine. `uv` locks the specific Python version and installs all required modules automatically when you run the script.

### Running the Experiments

Once you have `uv`, just run:

```bash
uv run main.py
```

This launches an interactive selector. The script will simply ask you which of the four experiments you want to reproduce:

- `reproduction`
- `detection baseline`
- `forward`
- `backward`

If you don't have the datasets set up yet, the script detects that. It will guide you through the process of downloading the raw data and converting it into the format required for the models.

## Datasets

Here are the links to the two datasets used in this study:

- [**Malimg**](https://www.dropbox.com/scl/fi/wdb6omeiu2lg796qvt9l7/malimg_dataset.zip?rlkey=63q2xqmtlm66gilf6idd2c9k7&e=2&dl=0)
- [**PE Malware**](https://practicalsecurityanalytics.com/pe-malware-machine-learning-dataset/)
