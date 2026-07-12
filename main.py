from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import re
import torch
import torch.nn as nn

app = Flask(__name__, template_folder='templates')
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
import os
print("Templates folder:", app.template_folder)
print("Index exists:", os.path.exists(os.path.join(app.template_folder, 'index.html')))

# ============================================================
# Knowledge base CSVs
# ============================================================
sym_des     = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout     = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets       = pd.read_csv("datasets/diets.csv")

# ============================================================
# Disease name mapping
# LSTM classes → CSV disease names (where they differ)
# ============================================================
NLP_TO_CSV = {
    # lowercase → Title Case
    'allergy'                        : 'Allergy',
    'diabetes'                       : 'Diabetes',
    'drug reaction'                  : 'Drug Reaction',
    'peptic ulcer disease'           : 'Peptic ulcer disease',
    'urinary tract infection'        : 'Urinary tract infection',
    # different wording
    'gastroesophageal reflux disease': 'GERD',
    'Dimorphic Hemorrhoids'          : 'Dimorphic hemmorhoids(piles)',
    # capitalisation mismatch
    'Varicose Veins'                 : 'Varicose veins',
}

def to_csv_name(nlp_name):
    return NLP_TO_CSV.get(nlp_name, nlp_name)

# ============================================================
# Helper — fetch recommendations from CSVs
# ============================================================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(list(desc))

    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [col for col in pre.values]

    med    = list(medications[medications['Disease'] == dis]['Medication'].values)
    die    = list(diets[diets['Disease'] == dis]['Diet'].values)
    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# ============================================================
# Tokenizer (manual — no external NLP library)
# ============================================================
STOPWORDS = {
    'i','have','been','a','an','the','and','or','with','am','is','are',
    'was','my','me','also','since','very','on','in','at','to','of',
    'has','it','this','that','there','for','some'
}
SUFFIXES = ['ing','edly','ed','es','ly','s']

def simple_stem(word):
    """Rule-based stemmer — removes common suffixes."""
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word

def tokenize(text):
    """Manual tokenization pipeline: clean → split → stopword removal → stemming."""
    text   = re.sub(r'[^a-z\s]', ' ', text.lower())
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return [simple_stem(t) for t in tokens]

def clean_text_tfidf(text):
    """Cleaned text for TF-IDF (joined tokens)."""
    return ' '.join(tokenize(text))

# ============================================================
# TF-IDF + SVM (baseline model)
# ============================================================
tfidf_vec = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
svm_model = pickle.load(open('models/svm_tfidf.pkl', 'rb'))
le        = pickle.load(open('models/label_encoder.pkl', 'rb'))

def predict_tfidf(text, top_k=3):
    """Predict disease using TF-IDF + SVM."""
    cleaned = clean_text_tfidf(text)
    vec     = tfidf_vec.transform([cleaned])
    proba   = svm_model.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    return [(le.classes_[i], round(proba[i]*100, 1)) for i in top_idx]

# ============================================================
# BiLSTM model
# ============================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        _, (h_n, _)    = self.lstm(embedded)
        out             = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(out))

# load checkpoint
ckpt = torch.load('models/symptom_lstm.pt', map_location='cpu', weights_only=False)
w2i        = ckpt['word2idx']
classes    = ckpt['classes']
MAX_LEN    = ckpt['max_len']

lstm_model = BiLSTMClassifier(
    vocab_size  = ckpt['vocab_size'],
    embed_dim   = ckpt['embed_dim'],
    hidden_dim  = ckpt['hidden_dim'],
    num_classes = len(classes)
)
lstm_model.load_state_dict(ckpt['model_state_dict'])
lstm_model.eval()

def encode_text(text):
    ids = [w2i.get(t, 1) for t in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

def predict_lstm(text, top_k=3):
    """Predict disease using BiLSTM."""
    tensor = torch.tensor([encode_text(text)], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(lstm_model(tensor), dim=1)[0]
    top_p, top_i = torch.topk(probs, min(top_k, len(classes)))
    return [(classes[i], round(float(p)*100, 1)) for i, p in zip(top_i, top_p)]

# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction route — accepts free-text symptom description.
    Runs both TF-IDF+SVM and BiLSTM, returns BiLSTM as primary result.
    """
    user_text = request.form.get('symptoms', '').strip()

    if not user_text:
        return render_template('index.html',
                               message="Please describe your symptoms.")

    # ── BiLSTM prediction (primary) ──
    lstm_top3  = predict_lstm(user_text, top_k=3)
    top_disease_lstm = lstm_top3[0][0]
    top_conf_lstm    = lstm_top3[0][1]

    # ── TF-IDF + SVM prediction (comparison) ──
    tfidf_top3       = predict_tfidf(user_text, top_k=3)
    top_disease_tfidf = tfidf_top3[0][0]
    top_conf_tfidf    = tfidf_top3[0][1]

    # ── Fetch details from CSV (use BiLSTM result) ──
    csv_name = to_csv_name(top_disease_lstm)
    try:
        dis_des, my_precautions_raw, meds, diet, wrkout = helper(csv_name)
        my_precautions = list(my_precautions_raw[0]) if len(my_precautions_raw) > 0 else []
    except Exception:
        dis_des        = "Detailed information not available."
        my_precautions = []
        meds = diet = wrkout = []

    return render_template(
        'index.html',
        # Primary result (BiLSTM)
        predicted_disease = csv_name,
        confidence        = top_conf_lstm,
        top3_lstm         = lstm_top3,
        # Comparison (TF-IDF + SVM)
        tfidf_disease     = to_csv_name(top_disease_tfidf),
        tfidf_confidence  = top_conf_tfidf,
        tfidf_top3        = tfidf_top3,
        # Details
        dis_des           = dis_des,
        my_precautions    = my_precautions,
        medications       = meds,
        my_diet           = diet,
        workout           = wrkout,
    )



@app.route('/predict_chat', methods=['POST'])
def predict_chat():
    """
    Chat route — receives disease name + confidence from the JS chatbot.
    Fetches full details from CSVs and returns them to the template.
    """
    disease_name = request.form.get('disease', '').strip()
    confidence   = request.form.get('confidence', '85').strip()

    if not disease_name:
        return render_template('index.html', message='No diagnosis received from chat.')

    csv_name = to_csv_name(disease_name)

    try:
        conf_float = float(confidence)
    except Exception:
        conf_float = 85.0

    try:
        dis_des, my_precautions_raw, meds, diet, wrkout = helper(csv_name)
        my_precautions = list(my_precautions_raw[0]) if len(my_precautions_raw) > 0 else []
    except Exception:
        dis_des        = 'Detailed information not available.'
        my_precautions = []
        meds = diet = wrkout = []

    return render_template(
        'index.html',
        predicted_disease = csv_name,
        confidence        = conf_float,
        top3_lstm         = [(csv_name, conf_float)],
        tfidf_disease     = None,
        tfidf_confidence  = None,
        tfidf_top3        = None,
        dis_des           = dis_des,
        my_precautions    = my_precautions,
        medications       = meds,
        my_diet           = diet,
        workout           = wrkout,
    )


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    app.run(debug=True)
