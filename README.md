# Interpretable-Music-Genre-Classification-using-CNNs-and-Spectrogram-Analysis1

Project Overview

In this project, I developed a Convolutional Neural Network (CNN)-based music genre classification model that analyzes Mel-spectrograms extracted from audio files. Beyond classification, this project incorporates model interpretability by integrating Grad-CAM and a lightweight Large Language Model (LLM) to convert CNN activations into human-readable explanations, allowing end-users to better understand genre predictions.

This work combines computer vision, audio processing, and explainable AI to enhance transparency in deep learning-based classification tasks.

 Data Source

-  **Dataset:** Free Music Archive (FMA) Small Dataset
-  **Data Type:** Raw audio (.mp3 files)
-  **Preprocessed as:** Mel-spectrograms converted into 2D numpy arrays (`.npy` files)


## Pipeline Summary

1️) **Audio Preprocessing:**
- Loaded raw `.mp3` files.
- Extracted Mel-spectrograms using Librosa.
- Saved spectrograms as numpy arrays for efficient model training.

2️) **Model Architecture:**
- Built a custom 2D CNN using PyTorch.
- Input: Mel-spectrograms
- Output: Predicted music genres.
- Achieved test accuracy: **74%**

️3) **Model Interpretability:**
- Applied Grad-CAM to extract CNN feature activations.
- Translated activations into natural language explanations using an LLM-based prompt generation.

4️) **Evaluation:**
- Model performance assessed through accuracy, confusion matrix, and visualization of class-wise predictions.



## Technologies Used

| Category | Tools/Libraries |
| -------- | --------------- |
| Programming Language | Python |
| Audio Processing | Librosa |
| Deep Learning | PyTorch |
| Model Interpretability | Grad-CAM |
| Natural Language Generation | Lightweight LLM |
| Data Handling | numpy, pandas |
| Visualization | matplotlib, seaborn |
