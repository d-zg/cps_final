
# d‑zg‑cps_final


This repository contains the code and notebooks for training and evaluating a set of knowledge‑distillation baselines on CIFAR‑10 for my final cps project. Each baseline has its own folder with training and evaluation scripts.

Install the requiremed modules with `pip install -r requirements`. Then you can run each of the training scripts, and subsequently after training you can run the test script which tests on the held out test set. 

Here were my results: 

| Method                                             | Test Accuracy (%) |
|----------------------------------------------------|-------------------|
| Learning on Labels (No Teacher)                    | 63.75             |
| Teacher                                            | 81.23             |
| Naive KD (Minimizing KL Divergence)                | 67.16             |
| Patient & Consistent Algorithm                     | 78.08             |
| FGSM‑Based KD (random 10%)                         | 79.20             |
| FGSM‑Based KD (all inputs)                         | 76.74             |
| Trajectories (all inputs)                          | 73.43             |
| Trajectories (random 10%)                          | 75.73             |
| Logit‑Guided Targeted (Proposed, 10%)              | 76.71             |

Below is a description of the each of the folders/baselines. 

---

## Baselines

### 1. Learning on Labels (No Teacher)  
Train the student purely on hard labels (no distillation).  
- **Training notebook:**  
  `Squeezenet/train_squeezenet.ipynb`  
- **Evaluation notebook:**  
  `Squeezenet/test.ipynb`

### 2. Naive KD (Minimizing KL Divergence)  
Match student logits to teacher logits via KL divergence.  
- **Training notebook:**  
  `naive_kd/naive.ipynb`  
- **Evaluation notebook:**  
  `naive_kd/test.ipynb`

### 3. Patient and Consistent Algorithm  
“Good Teacher Is Patient and Consistent” baseline with shared augmentations.  
- **Training notebook:**  
  `Patient_and_Consistent/patient.ipynb`  
- **Evaluation notebook:**  
  `Patient_and_Consistent/test.ipynb`

### 4. FGSM‑Based KD (on random 10 % of inputs)  
Use FGSM adversarial examples on a random 10 % subset.  
- **Training notebook:**  
  `fgsm_kd/train.ipynb`  
- **Evaluation notebook:**  
  `fgsm_kd/test.ipynb`

### 5. FGSM‑Based KD (on all inputs)  
Use FGSM adversarial examples on every training image.  
- **Training notebook:**  
  `fgsm_all_kd/train.ipynb`  
- **Evaluation notebook:**  
  `fgsm_all_kd/test.ipynb`

### 6. Trajectories for All Inputs  
Generate multi‑step adversarial trajectories for every image.  
- **Training notebook:**  
  `Hard_problems/all/train.ipynb`  
- **Evaluation notebook:**  
  `Hard_problems/all/test.ipynb`

### 7. Trajectories for Random 10 % of Inputs  
Generate multi‑step adversarial trajectories on a random 10 %.  
- **Training notebook:**  
  `Hard_problems/random/train.ipynb`  
- **Evaluation notebook:**  
  `Hard_problems/random/test.ipynb`

### 8. Logit‑Guided Targeted (Proposed)  
Select the top 10 % “hard” examples by runner‑up logit and generate trajectories.  
- **Training notebook:**  
  `Hard_problems/logit_guided/train.ipynb`  
- **Evaluation notebook:**  
  `Hard_problems/logit_guided/test.ipynb`

---

## Teacher model

Before running any KD baselines, train or load the ResNet‑18 teacher. The model file is included, so you should be fine to just run them. 

- **Training notebook:**  
  `Resnet18/train_resnet20.ipynb`  
- **Evaluation notebook:**  
  `Resnet18/test.ipynb`


---

## How to use

1. Create a Python 3.8+ virtual environment and install  
   ```bash
   pip install -r requirements.txt
