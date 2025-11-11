# from flask import Flask, jsonify

# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return jsonify(message="Hello, World from Flask on Render!")

# @app.route('/name')
# def helloname():
#     return jsonify(message="Hello, raushan!")


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=10000)



# Improvements Summary
# Optimization	Effect
# Removed thesaurus.com requests	70–90% faster
# Added synonym caching	Reuses previous computations
# Removed word combinations	Greatly reduced CPU load
# Used latin1 encoding	Avoids Unicode errors
# Preloaded model globally	No retraining per request


from flask import Flask, request, jsonify
import warnings
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import os
import nltk

warnings.simplefilter("ignore")

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- NLTK Setup -------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# ------------------- Global Cache -------------------
synonym_cache = {}

# ------------------- Synonyms Function -------------------
def synonyms(term):
    """Return cached WordNet-based synonyms for a term."""
    if term in synonym_cache:
        return synonym_cache[term]
    synonym_set = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonym_set.add(lemma.name().replace('_', ' '))
    synonym_cache[term] = synonym_set
    return synonym_set

# ------------------- Load Dataset -------------------
CSV_PATH = r"Dataset\diseasesymp_updated.csv"

# Use latin1 to avoid UnicodeDecodeError
df = pd.read_csv(
    CSV_PATH,
    encoding='latin1',
    low_memory=False,
    on_bad_lines='skip'
)

# ------------------- Prepare Model -------------------
X = df.drop(columns=["label_dis"])
Y = df["label_dis"]

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X, Y)

dataset_symptoms = list(X.columns)
@app.route('/')
def hello():
    return jsonify(message="Hello, World from Flask on Render!")

# ------------------- Prediction Route -------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get("symptoms", [])

    # Step 1: Preprocess user symptoms
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip().replace('_', ' ').replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    # Step 2: Expand using WordNet synonyms (fast, cached)
    expanded_symptoms = []
    for user_sym in processed_user_symptoms:
        words = user_sym.split()
        str_sym = set(words)
        for word in words:
            str_sym.update(synonyms(word))
        expanded_symptoms.append(' '.join(str_sym))

    # Step 3: Match user symptoms with dataset columns
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        for user_sym in expanded_symptoms:
            if data_sym.replace('_', ' ') in user_sym:
                found_symptoms.add(data_sym)

    # Step 4: Create input vector
    sample_x = [0 for _ in range(len(dataset_symptoms))]
    for val in found_symptoms:
        if val in dataset_symptoms:
            sample_x[dataset_symptoms.index(val)] = 1

    # Step 5: Predict probabilities
    prediction = lr_model.predict_proba([sample_x])
    k = 5
    diseases = encoder.classes_
    topk = prediction[0].argsort()[-k:][::-1]

    topk_dict = {diseases[t]: round(prediction[0][t] * 100, 2) for t in topk}

    return jsonify({"predictions": topk_dict})

# ------------------- Disease Detail Route -------------------
@app.route('/disease/<name>', methods=['GET'])
def get_disease_detail(name):
    try:
        details = diseaseDetail(name)
        return jsonify({"disease": name, "details": details})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Receive Training Data Route -------------------
@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.json
    print("Data received at /receive:", data, flush=True)

    symptoms = data.get("symptoms", [])
    doctor_diseases = data.get("final_diagnosis_by_doctor", [])

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding='latin1')
    else:
        df = pd.DataFrame()

    # Ensure symptom columns exist
    for symptom in symptoms:
        if symptom not in df.columns:
            df[symptom] = 0

    # Ensure label_dis column exists
    if "label_dis" not in df.columns:
        df["label_dis"] = ""

    new_rows = []
    for disease in doctor_diseases:
        new_row = {col: 0 for col in df.columns}
        for symptom in symptoms:
            new_row[symptom] = 1
        new_row["label_dis"] = disease
        new_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    temp_path = CSV_PATH + ".tmp"
    df.to_csv(temp_path, index=False, encoding='latin1')
    os.replace(temp_path, CSV_PATH)

    return jsonify({
        "status": "success",
        "rows_added": len(new_rows),
        "received_data": data
    })

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
