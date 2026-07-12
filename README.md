# Health AI Hub — Clinical Diagnosis and Recommendation System

Health AI Hub is an advanced, symptom-based clinical recommendation platform. Powered by a dual-model machine learning pipeline (BiLSTM Neural Networks + TF-IDF SVM), the web application parses natural, free-text symptom descriptions in English, delivers diagnostic classifications, and displays comprehensive care suggestions (including suggested medications, required precautions, recommended diets, and workouts).

---

## 🌟 Key Features

- **Free-Text Symptom Intake**: Describe symptoms in plain English without filling out static check-boxes.
- **Dual-Model Inference**:
  - **BiLSTM Recurrent Neural Network (Primary)**: Captures sequential information and syntax contexts bidirectionally.
  - **TF-IDF + Support Vector Classifier (SVM Baseline)**: Computes word occurrences mapped to high-dimensional hyperplanes.
- **Interactive Chatbot Follow-Up**: An intelligent decision-tree assistant dynamically refines diagnosis certainty by asking targeted clinical follow-up questions.
- **Diagnostics Dashboard**: Clear visual graphs showing model confidence, disease descriptions, and detailed treatment, precaution, and lifestyle guidelines.
- **Premium Glassmorphism Design**: Dynamic styling with smooth transitions, modern typography (Outfit), and a persistent, flicker-free Light/Dark Mode toggle.

---

## 🛠️ Technology Stack

- **Backend**: [Flask](https://flask.palletsprojects.com/) (Jinja2 Templates), Python 3.10+
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (for BiLSTM neural network models)
- **Machine Learning Utilities**: [scikit-learn](https://scikit-learn.org/) (for TF-IDF, SVM, and Label Encoders)
- **NLP Preprocessing**: Custom manual tokenizer & stemming algorithms (zero-dependency for academic transparency)
- **Frontend**: Bootstrap 5, Custom CSS variables, HTML5, Vanilla JavaScript

---

## 📚 Custom NLP Preprocessing Pipeline

To maintain strict control over terminology parsing and comply with academic auditing requirements, this project implements a complete Natural Language Processing (NLP) pipeline from scratch (without using external libraries like NLTK or spaCy):

1. **Manual Tokenizer**: Normalizes casing, strips non-alphabetical characters, and tokenizes strings.
2. **Stopwords Filter**: Screens words against a predefined set of semantic noise (e.g., "I", "have", "been", "was").
3. **Rule-Based Stemmer**: Evaluates word suffix boundaries and trims common inflections (e.g., `-ing`, `-edly`, `-ed`, `-es`, `-ly`, `-s`) for terms longer than 3 characters.
4. **Sequence Encoding**: Embeds stemmed tokens into a vocabulary dictionary map and pads sequences to a uniform length of 10.

---

## 🧠 Model Architectures

### 1. Bidirectional LSTM Classifier
Processed via PyTorch:
- **Embedding Layer**: Translates categorical word tokens into dense embeddings.
- **Stacked BiLSTM (2 layers)**: Processes sequences bidirectionally, capturing surrounding context before predicting classes.
- **Dropout (30%)**: Mitigates model overfitting.
- **Dense Classifier**: Maps the combined context output of forward and backward steps to the final disease categories.

### 2. TF-IDF + Support Vector Classifier (SVM)
Processed via scikit-learn:
- Uses a **TF-IDF Vectorizer** to calculate word frequency metrics.
- Processes vectors through a probability-enabled **Linear SVM Classifier** to output class likelihood distributions.

---

## 📂 Project Directory Structure

```text
├── datasets/
│   ├── description.csv          # Disease clinical descriptions
│   ├── diets.csv                # Diet plans matching diagnoses
│   ├── medications.csv          # Suggested medications list
│   ├── precautions_df.csv       # Clinical precautions list
│   ├── symtoms_df.csv           # Base symptoms lookup database
│   └── workout_df.csv           # Suggested workout/exercises list
├── models/
│   ├── label_encoder.pkl        # scikit-learn Label Encoder
│   ├── svm_tfidf.pkl            # Pre-trained TF-IDF SVM baseline model
│   ├── symptom_lstm.pt          # PyTorch BiLSTM weights checkpoint
│   └── tfidf_vectorizer.pkl     # Pre-trained TF-IDF Vectorizer
├── static/
│   ├── css/
│   │   └── style.css            # Premium CSS variable styling stylesheet
│   ├── IMG.PNG                  # Brand logo
│   ├── profile1.png             # portrait avatar for Mostafa Khaled (NLP Eng)
│   └── profile2.png ... 6       # portrait avatars for other team members
├── templates/
│   ├── about.html               # Project details & architectural notes page
│   ├── contact.html             # Simulated form submission & social panel
│   ├── developer.html           # Developer team grid & contributions panel
│   └── index.html               # Main diagnostic intake & dashboard portal
├── main.py                      # Flask routes, pipeline definitions, inference
└── README.md                    # Project documentation (this file)
```

---

## ⚡ Setup & Local Running Instructions

### 1. Prerequisite Installations
Ensure you have Python 3.10+ installed.

### 2. Dependencies Setup
Install the required packages:
```bash
pip install flask numpy pandas scikit-learn torch
```

### 3. Running the Application
Launch the Flask development server:
```bash
python main.py
```

Open a web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## ⚠️ Important Disclaimer

> [!WARNING]
> This application is built as an educational demonstration of NLP algorithms and clinical sequence prediction models. **It does not constitute qualified medical advice or professional diagnosis.** Always consult a certified healthcare professional before starting any treatment plans, taking medications, or adopting new diet/exercise regimens.
