# Shakespeare Next Word Prediction

### End-to-End Deep Learning NLP Project (LSTM vs GRU) + Streamlit Deployment

This project builds and deploys deep learning language models to perform **next-word prediction** using *Shakespeare’s Hamlet* from the NLTK Gutenberg corpus.

It demonstrates:

* NLP preprocessing pipeline
* Sequence modeling with LSTM & GRU
* Model serialization
* Web deployment using Streamlit
* Model comparison

---

## Problem Statement

Can recurrent neural networks learn literary language structure and predict the next word in Shakespearean text?

This project explores how **LSTM and GRU architectures model long-term dependencies** in classical English text and compares their performance.

---

## Technical Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **NLTK**
* **Streamlit**
* **Pickle (Tokenizer persistence)**

---

## Dataset

* Source: NLTK Gutenberg Corpus
* Text Used: `shakespeare-hamlet.txt`
* Size: ~30,000+ lines of Shakespearean dialogue

```python
from nltk.corpus import gutenberg
data = gutenberg.raw('shakespeare-hamlet.txt')
```

---

## Project Architecture

### Text Processing Pipeline

* Lowercasing
* Cleaning special characters
* Tokenization
* Padding sequences

---

### Model Architectures

#### LSTM Model

* Embedding Layer
* LSTM (Long Short-Term Memory)
* Drop out layer
* Dense Softmax Output

#### GRU Model

* Embedding Layer
* GRU (Gated Recurrent Unit)
* Drop out layer
* Dense Softmax Output

Both models trained using:

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Metric: Accuracy

---

## Repository Structure

```
Shakespeare-Next-Word-Prediction
│
├── app.py                    # Streamlit web application
├── training_notebook.ipynb   # Model training notebook
├── lstm_model.h5             # Saved LSTM model
├── gru_model.h5              # Saved GRU model
├── tokenization.pickle       # Saved tokenizer
├── requirements.txt
└── README.md
```

---

## Deployment (Streamlit App)

The Streamlit app allows users to:

* Input a seed sentence
* Generate Shakespeare-style continuation

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Sample Prediction

**Input:**

> To be or not to

**Output (LSTM):**

> To be or not to be

---

## 📈 Model Comparison

| Model | Parameters | Training Speed | Performance              |
| ----- | ---------- | -------------- | ------------------------ |
| LSTM  | Higher     | Slower         | Strong long-term capture |
| GRU   | Lower      | Faster         | Comparable performance   |

GRU achieved similar accuracy with fewer parameters, making it computationally efficient.

---

## 💡 Key Learnings

✔ Handling text sequence generation
✔ Understanding RNN gating mechanisms
✔ Comparing LSTM vs GRU in real-world text
✔ Saving and loading trained deep learning models
✔ Deploying ML models using Streamlit
✔ Building production-style ML pipelines

---

##  Why This Project Matters

This project demonstrates:

* End-to-end ML ownership
* NLP fundamentals
* Deep learning model comparison
* Practical deployment skills
* Clean project structuring for production readiness

It showcases both **theoretical understanding and practical implementation** of sequence modeling.

---

## Future Improvements

* Temperature-based sampling
* Top-k sampling
* Transformer-based comparison (GPT-style)
* Attention mechanisms

Tell me your target role and I’ll optimize it.

