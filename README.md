<p align="center">
<img width="300" height="300" alt="17" src="https://github.com/user-attachments/assets/2996f973-d7da-4252-996e-a35855098647" />
<img width="300" height="300" alt="30" src="https://github.com/user-attachments/assets/6d262dbd-355e-4e82-b9de-b188cc5303d7" />
</p> 

# Visualization-Based Automated Malware Classification: A Replication Study Transcended Towards Malware Detection

**Abstract:** TBA

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
