# Real-Time Indian Sign Language (ISL) Recognition System

![ISL Detection Demo](Screenshot%202025-04-15%20201536.png)

A real-time computer vision system that recognises **36 ISL gestures** (26 alphabets A–Z + digits 0–9) from live webcam feed using MediaPipe hand landmark extraction and a trained neural network classifier.

---

## Demo

| Gesture A (1.00) | Gesture 3 (1.00) | Gesture 5 (0.98) | Gesture N (0.98) |
|---|---|---|---|
| ![A](Screenshot%202025-04-20%20153303.png) | ![3](Screenshot%202025-04-20%20153536.png) | ![5](Screenshot%202025-04-15%20201604.png) | ![N](Screenshot%202025-04-16%20203510.png) |

| Gesture 8 (1.00) | Gesture 2 (0.97) | Gesture F (0.88) | Sign W |
|---|---|---|---|
| ![8](Screenshot%202025-04-20%20152729.png) | ![2](Screenshot%202025-04-15%20201517.png) | ![F](Screenshot%202025-04-20%20152942.png) | ![W](Screenshot%202025-04-15%20152034.png) |

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

## How to Run

```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn matplotlib
jupyter notebook ISL_Sign_Language_Recognition.ipynb
```

Dataset folder structure:
```
ISL_Dataset/
├── A/
├── B/
├── ...
└── 9/
```

---

## Results

| Metric | Value |
|---|---|
| Classes | 36 (A–Z, 0–9) |
| Real-time confidence | 0.97 – 1.00 |
| Augmentation | 6× original dataset size |

---

## Author

**Vatsala Misra** · B.Tech ECE, VIT Bhopal (CGPA: 8.66)  
[LinkedIn](https://linkedin.com/in/vatsala-misra) · [GitHub](https://github.com/VatsalaMisra)
