# CineVibe: Movie Review Sentiment Analysis

**A deep learning project leveraging LSTM neural networks and GloVe embeddings for accurate sentiment classification of IMDb movie reviews**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Model Comparison & Selection](#model-comparison-&-selection)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Model Architecture](#model-architecture)
- [Results & Performance](#results--performance)
- [Usage Guide](#usage-guide)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [License](#license)

---

## 🎬 Project Overview

This project explores different Deep Learning architectures to solve the binary sentiment classification problem (Positive/Negative) using the IMDb 50k dataset. While basic models provide a baseline, the project focuses on capturing long-term dependencies in text using Long Short-Term Memory (LSTM) networks and spatial patterns using Convolutional Neural Networks (CNN).

---

## ⚖ Model Comparison & Selection
Unlike a single-model approach, this project evaluates three distinct architectures to find the optimal balance between training speed and accuracy:

- **Simple Neural Network**: A baseline model using a Flatten layer after the Embedding layer. It serves as a benchmark for "bag-of-words" style neural approaches.

- **Convolutional Neural Network (Conv1D)**: Utilizes 1D convolutions to extract local features (n-grams) from the text.

- **LSTM (Final Model)**: A Recurrent Neural Network (RNN) variant designed to remember information over long sequences, making it ideal for understanding the context of long movie reviews.

Why LSTM won?

While the CNN was faster to train, the LSTM achieved superior performance (81.8% accuracy) because it effectively handled the sequential nature of language and the "vanishing gradient" problem common in standard RNNs.

---

## ⭐ Key Features

- **LSTM-Based Architecture**: Captures sequential dependencies in review text
- **Pre-trained Embeddings**: Uses GloVe.6B.100d word embeddings for rich semantic representation
- **High Accuracy**: Achieves 81.8% accuracy on test data using LSTM 
- **Scalable Pipeline**: Process batch predictions on unseen reviews
- **Production-Ready**: Saved model weights for easy deployment
- **Comprehensive Analysis**: Includes visualizations and performance metrics

---

## 📊 Datasets

All datasets used in this project are sourced from **Kaggle** and are publicly available:

### 1. **IMDB Dataset** (`a1_IMDB_Dataset.csv`)
   - **Source**: [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - **Size**: 50,000 reviews
   - **Content**: Movie reviews with binary sentiment labels (positive/negative)
   - **Usage**: Primary training and validation dataset

### 2. **GloVe Embeddings** (`a2_glove.6B.100d.txt`)
   - **Source**: [Kaggle - GloVe Embeddings](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)
   - **Dimensions**: 100-dimensional word vectors
   - **Vocabulary**: 400,000+ English words
   - **Usage**: Pre-trained embeddings for converting words to dense vectors

### 3. **IMDb Unseen Reviews** (`a3_IMDb_Unseen_Reviews.csv`)
   - **Source**: Kaggle IMDb Dataset (held-out test set)
   - **Purpose**: Testing model generalization on completely unseen reviews
   - **Output**: Predictions stored in `unseen_Predictions.csv`

---

## 📁 Project Structure

```
IMDb-Sentiment-Analyzer/
│
├── README.md                     # Project documentation (this file)
├── Sentiment_Analysis.ipynb      # Main Jupyter notebook with full pipeline
│
│── a1_IMDB_Dataset.csv           # Training dataset (50k reviews)
│── a2_glove.6B.100d.txt          # GloVe pre-trained embeddings
│── a3_IMDb_Unseen_Reviews.csv    # Test dataset for unseen predictions
│
│── lstm_model_acc0.818.keras     # Trained LSTM model (81.8% accuracy)
│── model.pkl                     # Preprocessing/tokenizer pickle file
│
│── unseen_Predictions.csv        # Model predictions on unseen reviews
│
└── Sentiment_Analysis.ipynb      # Complete analysis & training pipeline

```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook/JupyterLab
- TensorFlow/Keras
- NumPy, Pandas, Scikit-learn

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/CineVibe-IMDb-Sentiment-Analyzer.git
cd CineVibe-IMDb-Sentiment-Analyzer
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets
- Download the three datasets from Kaggle (links provided in Datasets section)
- Place them in the `data/` directory with the exact filenames specified

### Step 5: Run the Notebook
```bash
jupyter notebook Sentiment_Analysis.ipynb
```

---

## 🧠 Model Architecture

### LSTM Neural Network Design

```
Input Layer
    ↓
Embedding Layer (GloVe 100-dim vectors)
    ↓
LSTM Layer 1 (128 units, return sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 unit, Sigmoid activation)
    ↓
Binary Classification (Positive/Negative)
```

### Key Architecture Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embedding** | GloVe 100D | Pre-trained semantics reduce training data needs |
| **Hidden Units** | 128 → 64 | Capture hierarchical features with progressive refinement |
| **Dropout Rate** | 0.2 | Prevent overfitting while maintaining model capacity |
| **Optimizer** | Adam | Adaptive learning rates for efficient convergence |
| **Loss Function** | Binary Crossentropy | Standard for binary classification tasks |
| **Activation** | ReLU/Sigmoid | Non-linearity for hidden layers, probability for output |

---

## 📈 Results & Performance

### Model Performance Metrics for LSTM (The most successful model)

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~82.5% |
| **Validation Accuracy** | ~81.8% |
| **Test Accuracy** | 81.8% |
| **Precision** | ~0.82 |
| **Recall** | ~0.82 |
| **F1-Score** | ~0.82 |

### Performance Analysis

- **Convergence**: Model converges within 5-8 epochs
- **Overfitting**: Minimal overfitting due to dropout regularization
- **Generalization**: Strong performance on unseen IMDb reviews
- **Trade-off**: Balanced precision-recall for fair classification

---

## 📖 Usage Guide

### Running Predictions on New Reviews

```python
# Load the trained model and tokenizer
from tensorflow.keras.models import load_model
import pickle

model = load_model('models/lstm_model_acc0.818.keras')
with open('models/model.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Prepare your review
new_review = "This movie was absolutely amazing! Best film ever!"
sequence = tokenizer.texts_to_sequences([new_review])
padded = pad_sequences(sequence, maxlen=max_length)

# Make prediction
prediction = model.predict(padded)[0][0]
sentiment = "Positive" if prediction > 0.5 else "Negative"
confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
```

### Batch Processing Unseen Reviews

```python
import pandas as pd

# Load unseen reviews
unseen_df = pd.read_csv('data/a3_IMDb_Unseen_Reviews.csv')

# Generate predictions (see notebook for detailed implementation)
predictions = model.predict(processed_reviews)

# Save results
results_df = pd.DataFrame({
    'review': unseen_df['review'],
    'sentiment': ['Positive' if p > 0.5 else 'Negative' for p in predictions],
    'confidence': [max(p, 1-p) * 100 for p in predictions]
})
results_df.to_csv('outputs/unseen_Predictions.csv', index=False)
```

---

## 📄 File Descriptions

### Core Files

| File | Size/Type | Description |
|------|-----------|-------------|
| `Sentiment_Analysis.ipynb` | Jupyter Notebook | Complete ML pipeline: EDA, preprocessing, training, evaluation, predictions |
| `lstm_model_acc0.818.keras` | Keras Model (~2-5MB) | Trained LSTM model weights and architecture (81.8% accuracy) |
| `model.pkl` | Pickle File | Tokenizer and preprocessing pipeline for text vectorization |

### Data Files

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `a1_IMDB_Dataset.csv` | 50,000 | review, sentiment | Training & validation dataset |
| `a2_glove.6B.100d.txt` | 400,000 | word, vector | Pre-trained word embeddings |
| `a3_IMDb_Unseen_Reviews.csv` | Variable | review, (sentiment) | Test set for blind predictions |

### Output Files

| File | Description |
|------|-------------|
| `unseen_Predictions.csv` | Model predictions on unseen reviews (sentiment + confidence) |

---

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
numpy==1.24.3
pandas==2.0.3
tensorflow==2.14.0
keras==2.14.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
notebook==7.0.0
ipython==8.15.0
nltk==3.8.1
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔬 Methodology

### 1. Data Preparation
* **Data Acquisition**: Utilized the `a1_IMDB_Dataset.csv` containing 50,000 highly polar movie reviews.
* **Integrity Check**: Conducted missing value analysis using `.isnull().sum()`, confirming a clean dataset with no null entries.
* **Data Partitioning**: Split the dataset into **80% training** and **20% testing** sets using `train_test_split` with a fixed `random_state=45` for reproducibility.

### 2. Text Preprocessing
* **Noise Filtering**: Implemented a robust cleaning pipeline using Regular Expressions (`re`):
    * Removed all non-alphabetical characters (numbers and punctuation).
    * Removed standalone single characters (e.g., "s", "a", "i") that don't add semantic value.
    * Collapsed multiple whitespace characters into single spaces.
* **Stopword Removal**: Filtered out common English stopwords using the `nltk` library to focus on sentiment-heavy words.
* **Vectorization**:
    * Tokenized the corpus using the Keras `Tokenizer`.
    * Standardized all input lengths to **100 words** (`max_pad = 100`) using "post" padding to ensure uniform tensor dimensions for the neural networks.



### 3. Word Embedding
* **GloVe Integration**: Loaded the pre-trained **GloVe.6B.100d** (Global Vectors for Word Representation) to map words into a 100-dimensional semantic vector space.
* **Embedding Matrix**: Created a specialized weight matrix for the vocabulary where each index corresponds to its pre-trained GloVe vector.
* **Layer Configuration**: Initialized the `Embedding` layer with these weights and set `trainable=False` to leverage pre-trained knowledge without altering it during training.

### 4. Model Architecture Comparison
The project systematically compared three distinct architectures to determine the best performer:

| Model | Architecture Highlights | Purpose |
| :--- | :--- | :--- |
| **Simple Neural Network** | `Embedding` -> `Flatten` -> `Dense(Sigmoid)` | Baseline benchmark for "bag-of-words" logic. |
| **CNN (1D)** | `Conv1D(128 filters)` -> `GlobalMaxPooling1D` | Capturing local features and n-grams (spatial patterns). |
| **LSTM (Final)** | `LSTM(128 units)` -> `Dense(Sigmoid)` | Capturing long-term dependencies and sequence context. |



### 5. Training & Evaluation
* **Hyperparameters**: All models utilized the **Adam** optimizer and **Binary Cross-Entropy** loss function.
* **Training Protocol**: 
    * Trained for **6 to 10 epochs** with a batch size of **128**.
    * Reserved **20% of the training data** for real-time validation (`validation_split=0.2`).
* **Selected Model**: The **LSTM** model was chosen as the final predictor due to its superior ability to handle the sequential nature of long movie reviews.

### 6. Inference & Deployment
* **Blind Testing**: Applied the finalized LSTM model to a completely separate dataset (`a3_IMDb_Unseen_Reviews.csv`).
* **Result Export**: Generated sentiment predictions and exported them to `unseen_Predictions.csv`, including Movie Title, Review Text, IMDb Rating, and the Predicted Sentiment.
* **Persistence**: Saved the final LSTM model using `pickle` for future deployment.---

## 🎯 Future Enhancements

- [ ] Implement attention mechanism for better interpretability
- [ ] Add bidirectional LSTM (BiLSTM) for improved context capture
- [ ] Experiment with transformer models (BERT, RoBERTa)
- [ ] Create REST API for deployment
- [ ] Add confidence calibration techniques
- [ ] Implement aspect-based sentiment analysis
- [ ] Build web interface for interactive testing
- [ ] Expand to multi-class sentiment (5-star ratings)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentations

---

## 📞 Support & Questions

For questions or issues:
1. Check the Jupyter notebook for detailed code comments
2. Review the Kaggle dataset pages for data documentation
3. Open an issue on the repository
4. Contact the project maintainer

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Dataset Licenses

- **IMDB Dataset**: Licensed for academic and personal use (check Kaggle terms)
- **GloVe Embeddings**: Licensed under Open Data Commons (CC0)
- **IMDb Unseen Reviews**: Licensed for academic and personal use

---

## 🙏 Acknowledgments

- **Kaggle Community** for the high-quality datasets and resources
- **Stanford NLP Group** for the GloVe embeddings
- **TensorFlow/Keras** team for the excellent deep learning framework
- **Dataset Contributors**: All reviewers and data annotators on IMDb

---

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@project{cinevibe2024,
  title={CineVibe: Movie Review Sentiment Analysis},
  author={Biprojeet Chaudhury},
  year={2024},
  url={https://github.com/biprojeetchaudhury/CineVibe-IMDb-Sentiment-Analyzer}
}
```

---

## 🔗 Useful Links

- [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [GloVe Embeddings](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [LSTM Fundamentals](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

**Last Updated**: March 2024  
**Project Status**: Active & Maintained  
**Model Accuracy**: 81.8%
