# 🧠 Handwritten Digit Recognizer

A from-scratch ML project that trains a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) using the MNIST dataset — then predicts from your own drawn images.

Built with PyTorch. No pretrained models. Everything trained from zero.

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.67%** |
| Dataset | MNIST (70,000 images) |
| Model | Custom CNN |
| Training Time | ~5–10 min (CPU) |

---

## 🧠 Model Architecture

Upgraded from a simple linear network to a **CNN** for spatial pattern recognition:

```
Input (1 × 28 × 28)
↓
Conv2d(1→32, 3×3) + ReLU + MaxPool → 32 × 14 × 14
↓
Conv2d(32→64, 3×3) + ReLU + MaxPool → 64 × 7 × 7
↓
Conv2d(64→128, 3×3) + ReLU → 128 × 7 × 7
↓
Flatten → Linear(6272→256) + ReLU + Dropout(0.5)
↓
Linear(256→10)
↓
Output (digits 0–9)
```

CNN learns edges → shapes → digit patterns. Far more robust than flat linear layers.

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| LR Scheduler | StepLR (step=3, gamma=0.5) |
| Epochs | 10 |
| Batch Size | 64 |

### Data Augmentation
Training images are randomly transformed each epoch to improve robustness:
- Random rotation (±15°)
- Random affine (translate, scale)
- Gaussian blur

---

## 📂 Project Structure

```
digit-recognizer/
├── 01_explore_data.py   # Load + visualize MNIST dataset
├── 02_build_model.py    # Neural network architecture (linear, for reference)
├── 03_train.py          # CNN training + evaluation + save weights
├── 04_predict.py        # Predict digits from custom images
│
├── model/
│   └── digit_model.pth  # Trained CNN weights
│
├── mnist_samples.png    # Sample dataset visualization
├── training_loss.png    # Loss curve
├── data/                # MNIST dataset (auto-downloaded)
└── .gitignore
```

---

## 🚀 How to Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install torch torchvision matplotlib pillow numpy

# 3. Explore dataset
python 01_explore_data.py

# 4. Train model
python 03_train.py

# 5. Predict your own digit
python 04_predict.py
```

---

## 🖼️ Sample Data

![MNIST Samples](mnist_samples.png)

---

## 📉 Training Loss

![Training Loss](training_loss.png)

Loss drops from ~0.33 → ~0.04 over 10 epochs.

---

## 🔮 Custom Image Prediction

Run `04_predict.py` → enter path to your image → get prediction + confidence.

**Preprocessing pipeline:**
```
Open image → Grayscale → Crop to digit (bbox)
→ Add padding → Resize to 28×28
→ Invert (white digit on black) → Normalize (MNIST stats)
→ Feed to CNN → Predicted digit + confidence %
```

**Tips for best results:**
- Draw on white background with dark/black ink
- Simple printed-style digits work best (MNIST style)
- Avoid italic or heavily stylized writing — model trained on printed handwriting only

---

## ⚠️ Known Limitations

- Trained on MNIST only — stylized, italic, or cursive digits may mispredict
- Works best with simple, centered, printed-style handwriting
- No support for multi-digit images (single digit per image)

---

## 🗺️ Journey / Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Project setup, venv, dependencies | ✅ Done |
| 1 | Explored MNIST dataset | ✅ Done |
| 2 | Built linear neural network | ✅ Done |
| 3 | Upgraded to CNN + augmentation → 99.67% | ✅ Done |
| 4 | Custom image inference | ✅ Done |
| 5 | Gradio web UI | 🔜 Next |

---

## 🛠️ Tech Stack

- Python 3.12
- PyTorch + torchvision
- Pillow (image processing)
- Matplotlib (visualization)