from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model and vectorizer
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    return render_template('index.html', prediction=label, message=message)

if __name__ == '__main__':
    app.run(debug=True)
