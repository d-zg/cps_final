
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


## Output Interpretation

To train each baseline, open the train.ipynb specified below, then after it is done, open and run the cell for the eval.ipynb in the same folder. 

- **Console output**  
  - During training, each epoch prints:  
    ```
    Epoch  10/200  Train Loss: 0.85  Train Acc: 72.3%  Val Loss: 0.78  Val Acc: 74.1%
    ```  
  - During evaluation, you’ll see:  
    ```
    Test Set Accuracy: 76.71%
    ```

- **Log files**  
  - `./logs/train.log` collects the same epoch‑by‑epoch entries with timestamps.  
  - Scan it to spot sudden drops or plateaus in loss or accuracy.

- **Metrics JSON**  
  - `./metrics/cifar10_training_metrics.json` is a line‑delimited JSON file.  
  - Each line has keys:  
    ```json
    {
      "epoch": 10,
      "train_loss": 0.85,
      "train_acc": 72.3,
      "val_loss": 0.78,
      "val_acc": 74.1
    }
    ```  
  - Load it in Python to plot learning curves or compute statistics.

- **TensorBoard**  
  - Event files live under `runs/{baseline_name}/`.  
  - Launch with:
    ```bash
    tensorboard --logdir runs/
    ```  
  - Inspect loss and accuracy curves, compare baselines side by side.

- **Model checkpoints**  
  - The best student model (by validation accuracy) is saved as  
    `models/{baseline_name}_best_student_model.pth`.  
  - You can reload it for further analysis or inference.

- **Final test accuracy**  
  - Use the printed “Test Set Accuracy” as the single‑number summary of each run.  
  - Compare these across baselines to judge which approach yields the highest generalization.


Below is a description of the each of the data, then the folders/baselines. 

## Data

We use the CIFAR‑10 dataset throughout.  

- **Automatic download via PyTorch**  
  All code will fetch and cache CIFAR‑10 automatically. For example:
  ```python
  from torchvision.datasets import CIFAR10
  from torchvision import transforms

  transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
  ])

  train_set = CIFAR10(
      root='./data',
      train=True,
      download=True,        # <— downloads if not already present
      transform=transform
  )
  test_set = CIFAR10(
      root='./data',
      train=False,
      download=True,        # <— downloads if not already present
      transform=transform
  )


---

## Baselines
All parameters used are clearly documented in the code and were used for the results reported above. 
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

## Pretrained models/Teacher model

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
