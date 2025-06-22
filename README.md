# Insignia
This repository contains code for training models on <a href="https://www.kaggle.com/datasets/linardur/include-24-medical-modified/data">INCLUDE 24 Medical Modified dataset</a> along with the model generated from training it on that dataset.

The website implementation for this model as a React webapp is in this <a href="https://github.com/Im-Rik/Insignia">repository</a>.


# What this includes

> Lightweight Bi-LSTM + Soft-Attention model that reaches **92 % test accuracy** on a focused 24-word medical vocabulary for Indian Sign Language (ISL).  
> Trained on a **curated extension** of the public [INCLUDE-50] dataset plus ~400 smartphone clips we recorded to balance signer diversity, lighting and camera angles.

---

## 📂 Dataset

| Split | # clips | FPS × Frames |
|-------|--------:|--------------|
| Train | ≈ 560   | 30 fps × 60 |
| Val   | ≈ 70    | — |
| Test  | ≈ 70    | — |

* **Source 1 – INCLUDE-50** (CC-BY 4.0)   
* **Source 2 – Our recordings** (CC-BY 4.0) – triple-checked for medical relevance.  
* Landmarks are extracted with **MediaPipe Holistic** (pose + two hands + face = 1662 features/frame) and stored as `*.npy` for each of the 60 uniformly-sampled frames.

Grab it directly from <a href="https://www.kaggle.com/datasets/linardur/include-24-medical-modified/data">Kaggle</a>:

```bash
kaggle datasets download -d linardur/include-24-medical-modified
unzip include-24-medical-modified.zip
---

## ⚙️ Environment

```bash
conda create -n isl24 python=3.10
conda activate isl24
pip install -r requirements.txt           # Only initial requirements - download more dependencies as and when you find it is needed
```

`requirements.txt` pins **PyTorch 2.x**, **MediaPipe 0.10+**, **transformers 4.41+** and helper libs.

---

## 🏃‍♂️ Quick Start
By now, you should have a folder named dataset in which all your videos should be stored - basically rename the downloaded kaggle dataset to "dataset". The file structure should look something like :


### 1 ▶ Generate/verify landmarks (optional)

You can simply generate the mediapipe landmarks and by running the jupyter notebook's 6th cell which contains the following heading : 
```bash
Sample all videos with uniform frames (Select 60 frames out of entire video)
```

### 2 ▶ Train the model

To run the model training code, go to the 12th cell in the jupyter notebook file and run it. The code block looks something like :
```bash
# Bi-LSTM + Attention model training
```

* **Optimizer:** AdamW ([huggingface.co][1]) – lr = 3 × 10⁻⁴, weight-decay = 2 × 10⁻⁴
* **Scheduler:** 8-epoch linear warm-up → cosine decay
* **Loss:** `CrossEntropyLoss`
* **Regularisation:** LayerNorm + 0.4 dropout in the head

The best checkpoint is written to `sign_recognizer_best.pth` when validation loss improves by > 1 × 10⁻⁴.

### 3 ▶ Evaluate

After the training code is finised running, the last part of the training cell produces a full classification report and the test confusion-matrix shown in : ![Image](https://github.com/user-attachments/assets/48844b59-19d4-4298-8ba8-a8b3806e246a).

To evaluate on an single isolated clip of indian sign language data - just run the Isolated model inference code cell block that comes right after the training cell block. The block has the following heading :

```bash
# Isolated model inference on single video data 
```

### 4 ▶ Live demo (web-cam / MP4)

To run a real-time version of the code using a sliding window approach, run the code block which starts with : 


```bash
# Sliding Window for working Bi-LSTM + Attention With NA applied 
```

* Maintains a 60-frame sliding window.
* A gloss word is accepted only after **≥ 30 consecutive high-confidence frames**.
* Accumulated glosses are fed to **Llama-3 1B** via 🤗 Transformers to render a natural English sentence. ([pytorch.org][2])

---

## 🧠 Model Architecture

```
60 × 1662  →  Bi-LSTM (128 × 2)  →  Soft-Attention  →  LayerNorm
        →  Linear 128  →  ReLU  →  Dropout 0.4  →  Linear 24 classes
```


## 🔬 Results

| Metric (test set) | Value       |
| ----------------- | ----------- |
| Top-1 Accuracy    | **92.14 %** |
| Macro F1          | 0.91        |
| Params            | 1.8 M       |

---

## 📚 Further Reading

* INCLUDE-50 original paper ([zenodo.org/records/4010759][3])
* MediaPipe Holistic API docs ([mediapipe.readthedocs.io/en/latest/getting_started/python.html][1])
* PyTorch `nn.LSTM` & `nn.LayerNorm` docs
* AdamW optimiser ([huggingface.co][2])
* Cosine Annealing LR schedule ([discuss.pytorch.org/t/decreasing-maximum-learning-rate-after-every-restart/85776][4])

---

## 🤝 License

* **Dataset:**  CC-BY 4.0 .
* **Code:** MIT – see `LICENSE`.

---

[1]: https://mediapipe.readthedocs.io/en/latest/getting_started/python.html
[2]: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html "AdamW — PyTorch 2.7 documentation"
[3]: https://zenodo.org/records/4010759
[4]: https://discuss.pytorch.org/t/decreasing-maximum-learning-rate-after-every-restart/85776 "Decreasing Maximum learning rate after every restart"

