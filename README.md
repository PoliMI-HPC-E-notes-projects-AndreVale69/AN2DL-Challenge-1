<h1>AN2DL [2025-2026] - Challenge 1: Time Series Classification</h1>

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Description](#dataset-description)
  - [🏴‍☠️ The Pirate Pain Dataset](#️-the-pirate-pain-dataset)
  - [⚓ Files](#-files)
  - [🧭 Data Overview](#-data-overview)
  - [🏴‍☠️ Task](#️-task)
  - [⚙️ Data Loading](#️-data-loading)
  - [🗺️ Validation](#️-validation)

---

## Overview

This repository contains the dataset and baseline code for the AN2DL 2025-2026 Challenge 1: Time Series Classification.

The goal of this challenge is to develop models that can accurately classify multivariate time series data into predefined categories (high pain, low pain, no pain) based on temporal patterns and features extracted from the data.

The project includes:
- [The Pirate Pain Dataset](data/): A collection of multivariate time series data with associated labels.
- Baseline code: Jupyter notebooks demonstrating data loading, preprocessing, and model training
  - Prerequisite notebook: [notebooks/00_prerequisites.ipynb](notebooks/00_prerequisites.ipynb)
  - EDA and preprocess: [notebooks/01_eda_preprocess.ipynb](notebooks/01_eda_preprocess.ipynb)
  - Feature Engineering: [notebooks/02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb)
  - Model (training and evaluation): [notebooks/03_model.ipynb](notebooks/03_model.ipynb)
- [Internal modules](notebooks/internal/): Python functions and classes to facilitate data handling and model development. 
- For the challenge, we have defined a model based on Temporal Convolutional Networks (TCN) and BiLSTM with Attention mechanism, called [`PainTCNBiLSTMAttn`](notebooks/internal/nn/models/pain_tcn_bilstm_attn.py)
- [Report](docs/main.pdf): A detailed report on the challenge, including methodology, experiments, and results.

The submissions we created for the challenge are available in the [submissions folder](notebooks/submissions/). These are the submissions that we used to compete in the challenge. Out of 193 teams, we ranked $131^{\text{st}}$ on the public leaderboard and **$\mathbf{32}^{\textbf{nd}}$ on the private leaderboard**. Without using external data or pre-trained models, the private rank is more significant because it reflects the final evaluation.

---

## Prerequisites

Since data files are large, you need to download git-lfs to clone this repository:

```bash
# Install git-lfs (if not already installed)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Initialize git-lfs in your repository (once per machine)
git lfs install

# Clone the repository
git clone <repository-url>
cd AN2DL-Challenge-1
# Pull the large data files
git lfs pull
```

> [!TIP]
> If you have already cloned the repository without git-lfs, run the following commands inside the repository folder:
>
> ```bash
> git lfs install
> git lfs pull
> ```

Now, you should have all the data files in place.
To run the provided notebooks, make sure you have the required Python packages installed:

```bash
# Create a virtual environment
cd AN2DL-Challenge-1 # if not already in the repo folder
python3 -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
# Install jupyter
pip install jupyter ipykernel
```

If you are still having issues, please refer to the [official python venv documentation](https://docs.python.org/3/library/venv.html) for more details on setting up virtual environments.

Once the environment is set up, run the prerequisite notebook to install all other dependencies (CPU or GPU version):

```bash
jupyter notebook
```

And navigate to the notebook files in your web browser (usually at `http://localhost:8888`).

---

## Dataset Description

Competition hosted at [AN2DL 2025-2026](https://www.kaggle.com/competitions/an2dl2526c1).

### 🏴‍☠️ The Pirate Pain Dataset

Ahoy, matey! This dataset contains multivariate time series data, captured from both ordinary folk and pirates over repeated observations in time. Each sample collects temporal dynamics of body joints and pain perception, with the goal of predicting the subject’s true pain status:

- `no_pain`
- `low_pain`
- `high_pain`


### ⚓ Files

- `pirate_pain_train.csv` — training set
- `pirate_pain_train_labels.csv` — labels for the training set
- `pirate_pain_test.csv` — test set (with no labels)
- `sample_submission.csv` — an example of random submission


### 🧭 Data Overview

Each record represents a time step within a subject’s recording, identified by `sample_index` and `time`. The dataset includes several groups of features:

- `pain_survey_1`–`pain_survey_4` — simple rule-based sensor aggregations estimating perceived pain.
- `n_legs`, `n_hands`, `n_eyes` — subject characteristics.
- `joint_00`–`joint_30` — continuous measurements of body joint angles (neck, elbow, knee, etc.) across time.


### 🏴‍☠️ Task

Predict the real pain level of each subject based on their time-series motion data.


### ⚙️ Data Loading

```python
import pandas as pd

X_train = pd.read_csv('pirate_pain_train.csv')
y_train = pd.read_csv('pirate_pain_train_labels.csv')

X_test = pd.read_csv('pirate_pain_test.csv')
```


### 🗺️ Validation

No validation split be provided. You’ll need to chart your own course and create one from the training data.
