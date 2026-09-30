# app.py
from flask import Flask, render_template, request
from catboost import CatBoostClassifier
from feature_extraction import extract_features

app = Flask(__name__)

# Load the saved CatBoost model
model = CatBoostClassifier()
model.load_model("phishing_catboost_model.cbm")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        url = request.form['url']
        features = extract_features(url)
        prediction = model.predict(features)[0]
        result = "Legitimate ✅" if prediction == 1 else "Phishing 🚨"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
