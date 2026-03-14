# Real-Time Indian Sign Language (ISL) Recognition System

![ISL Detection Demo](results/isl_demo_3.png)

A real-time computer vision system that recognises **36 ISL gestures** (26 alphabets A–Z + digits 0–9) from live webcam feed using MediaPipe hand landmark extraction and a trained neural network classifier.

---

## Demo

| Gesture A (1.00) | Gesture 3 (1.00) | Gesture 5 (0.98) | Gesture N (0.98) |
|---|---|---|---|
| ![A](results/isl_demo_3.png) | ![3](results/isl_demo_2.png) | ![5](results/isl_demo_10.png) | ![N](results/isl_demo_8.png) |

Model achieves **0.97–1.00 confidence** on most gestures in real-time webcam testing across varied backgrounds and lighting conditions.

---

## Architecture

```
Webcam Frame
    ↓
MediaPipe Hands (21 landmarks × x,y,z = 63 features)
    ↓
Dense Neural Network (256 → 128 → 36)
  + BatchNormalization + Dropout
    ↓
Temporal Smoothing (sliding window of 5 frames)
    ↓
Predicted ISL Gesture + Confidence Score
```

**Why landmark-based instead of CNN on raw pixels?**
- Significantly faster inference — suited for real-time applications
- Invariant to background, lighting, and skin tone
- MediaPipe's pre-trained hand detector is highly optimised

---

## Tech Stack

| Component | Tool |
|---|---|
| Hand Detection | MediaPipe Hands |
| Feature Extraction | 21 keypoints × (x,y,z) = 63D vector |
| Classifier | TensorFlow/Keras Dense Network |
| Real-time Video | OpenCV |
| Data Augmentation | Rotation, scaling, noise on landmarks |

---

## Key Features

- **36-class recognition** — full ISL alphabet (A–Z) + digits (0–9)
- **Temporal smoothing** — 5-frame sliding window prevents flickering
- **Data augmentation** — geometric transforms on landmark space (6× dataset)
- **Class imbalance handling** — balanced class weights during training
- **Confidence gating** — only displays predictions above 50% confidence

---

## Project Structure

```
ISL-Sign-Language-Recognition/
├── ISL_Sign_Language_Recognition.ipynb   # Main notebook (data → train → inference)
├── results/                               # Demo screenshots from live testing
│   ├── isl_demo_1.png
│   ├── ...
├── isl_model_augmented.h5                # Trained model weights
├── X.npy                                  # Extracted landmark features
├── y.npy                                  # Labels
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install tensorflow opencv-python mediapipe numpy scikit-learn matplotlib

# Run the notebook
jupyter notebook ISL_Sign_Language_Recognition.ipynb
```

Dataset should be organised as:
```
ISL_Dataset/
├── A/  (images of ISL sign A)
├── B/
├── ...
├── 0/
└── 9/
```

---

## Results

| Metric | Value |
|---|---|
| Classes | 36 (A–Z, 0–9) |
| Real-time confidence (most gestures) | 0.97 – 1.00 |
| Inference | Real-time webcam feed |
| Augmentation | 6× original dataset size |

---

## Author

**Vatsala Misra**  
B.Tech ECE, VIT Bhopal (CGPA: 8.66)  
[LinkedIn](https://linkedin.com/in/vatsala-misra) · [GitHub](https://github.com/VatsalaMisra)
