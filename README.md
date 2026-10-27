# Adaptive Multi-level Data Sanitization using Graph Federated Learning and Secretary Tyrannosaurus Optimization for Privacy-utility Balanced Secure Cloud Sharing

## Abstract

Across diverse applications, an indispensable platform is cloud computing for sharing and storing large-scale distributed data. The significant challenges remain preserving data utility to protect sensitive information because heterogeneous cloud environments and increasing privacy risks. The conventional privacy-preservation techniques were limited adaptability to vary data characteristics, increased computational overhead, reduced analytical performance and encryption. To overcome the challenges, Graph Federated Learning with Transformer based Sensitive Attribute Identification and Secretary Tyrannosaurus Optimized Multi-Level Data Sanitization (GFL-TSAI-STO-MLDS) is presented for privacy-utility balanced secure cloud storage. At first, the data consistency and quality employs min-max normalization and missing value imputation. Without exposing raw data, the decentralized collaborative learning across distributed cloud clients is performed based on Graph Federated Learning (GFL). According to multiple privacy levels, the sensitive attributes are automatically identified and classified via Transformer based Sensitive Attribute Identification (TSAI) thereby processing the learned feature representations. Optimal sanitization parameters are determined with the help of novel Secretary Tyrannosaurus Optimization (STO) depending upon these sensitivity levels. The perturbation, generalization and perturbation is dynamically performed via Adaptive Multi Level Data Sanitization (MLDS). During sensitive attribute prediction and sanitization, an Explainable Artificial Intelligence (XAI) provides the transparent explanations that can help to enhance user trust and model interpretability.


## Repository contents

- `Main.py` 
- `utils_plot.py`
- `MIMIC-IV/`
- `outputs/`


## Dataset

The pipeline expects these CSV files under `MIMIC-IV/`:

- `patients.csv`
- `admissions.csv`
- `diagnoses_icd.csv`
- `procedures_icd.csv`
- `labevents.csv`
- `prescriptions.csv`

Download the dataset from the [provided Google Drive link](https://drive.google.com/file/d/1z_oip8GSeHYwLlGNCsAgRF0TPerR5A2h/view?usp=sharing), subject to the applicable access and data-use terms. The MIMIC-IV dataset contains sensitive health information.

## Requirements

Python 3.9 or newer is recommended. The scripts use:

- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- PyTorch
- PyTorch Geometric
- Captum
- tqdm

Install the standard dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn captum tqdm
```

Install PyTorch for your operating system and hardware using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install PyTorch Geometric following its [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/). PyTorch package selection depends on the Python version, operating system, and CPU/GPU configuration.

## Run

From the project directory, ensure the authorized CSV files are in `MIMIC-IV/`, then run:

```bash
python Main.py
```
