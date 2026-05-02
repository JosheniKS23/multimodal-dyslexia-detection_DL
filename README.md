# 🧠 Multimodal Dyslexia Detection in Children

This project explores the use of multimodal machine learning techniques to detect dyslexia in children using handwritten patterns and reading speech data.

---

## 📌 Project Overview

Dyslexia is a learning disorder that affects reading and writing abilities. Early detection can significantly improve intervention outcomes.

This project focuses on:
- 📝 Handwriting analysis using image-based CNN models  
- 🎙️ Speech analysis using mel-spectrograms and CNN  
- 🔄 Synthetic data generation using GAN  
- 🔗 Future integration into a multimodal fusion model  

---

## 🧠 Models Implemented

### 1. Handwriting CNN
- Character-level classification using YOLO annotations  
- Binary classification: Normal vs Reversal  
- Achieved strong baseline performance (~83% F1 score)  

---

### 2. GAN for Synthetic Data Generation
- Trained on reversal-class handwriting samples  
- Generated synthetic character images to augment the minority class  
- Applied in both balanced and imbalanced experimental settings  

#### 🔍 Observations:
- Synthetic data did **not significantly improve performance** in balanced datasets  
- In imbalanced settings, GAN showed **limited impact due to sufficient real data**  
- Demonstrated that **data quality and distribution are more critical than synthetic augmentation**  

👉 This highlights that GAN is most effective when:
- Data is scarce  
- Class imbalance is severe  

---

### 3. Speech Processing (In Progress)
- Audio converted to mel-spectrograms  
- CNN-based classification pipeline being developed  
- Will be used as the second modality in multimodal fusion  

---

## 📊 Key Findings

- CNN models effectively capture handwriting-based dyslexia patterns  
- Data balancing techniques significantly improve performance  
- GAN-based synthetic data does not always lead to better results  
- Large and well-distributed datasets reduce the need for augmentation  

---

## 🔗 Future Work

- 🔹 Integrate handwriting and speech features  
- 🔹 Implement multimodal fusion (early / late fusion)  
- 🔹 Explore attention-based architectures  
- 🔹 Evaluate on real dyslexia-specific datasets  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- OpenCV  
- Librosa  
- NumPy / Matplotlib  
- Hugging Face Datasets  

---

---

## ⚠️ Important Note

- The handwriting dataset used is synthetic and character-level annotated.  
- The speech dataset (TORGO) contains dysarthric speech, which is **not equivalent to dyslexia**, and is used as a proxy modality for speech-related learning patterns.  

---


## ⭐ Acknowledgements

- Hugging Face Datasets  
- TORGO Speech Dataset  
- Open-source ML community  
