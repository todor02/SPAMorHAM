# Spam or Ham Classifier

A simple **Naive Bayes text classifier** that detects whether a message is Spam or Ham (_Not Spam_).

---

## 📌 Features

- **Preprocesses text** (lowercasing + tokenization with regex)

- **Builds a vocabulary** from training data

- **Uses Multinomial Naive Bayes with Laplace smoothing**

- Reports **accuracy**, **precision**, **recall**, and **F1-score**

- Includes an **interactive demo** to test custom messages

## 📂 Dataset

### This project uses the SMS Spam Collection dataset:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

- The CSV file has two columns:

  - v1 → Label (ham or spam)

  - v2 → Message text

**Example**:
---
- v1
  - _SPAM_
- v2
  - "Congratulations! You've won a $1000 Walmart gift card. Call now!"
---
- v1
  - _HAM_
- v2
  - "Can we reschedule our meeting to 3 PM tomorrow?"
---
## 🚀 Usage

**1. Clone this repository**

**2. Run the script:**
- `python main.py`

---
### **_Example output:_**

- Accuracy: 0.95
- Precision: 0.94
- Recall:    0.91
- F1-Score:  0.92

---

### **Try the interactive demo:**

_--- Spam Classifier Demo ---_

Enter a message to classify:

"WIN A FREE iPhone! Click now!"

Prediction: SPAM

---

_🔮 Future Improvements_

_Add stopword removal and stemming/lemmatization_

_Try different classifiers (Logistic Regression, SVM, Neural Networks)_

_Deploy as a simple web app with Flask/Streamlit_

---
