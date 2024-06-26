from google.colab import drive

drive.mount('/content/drive')

model_path = '[PATH]/stack.model'
feature_extraction_path = '[PATH]/feature_extraction.pkl'
index_html_path = '[PATH]/flask/templates/Index.html'
flask_template_path = '[PATH]/flask/templates'
flask_static_path = '[PATH]/flask/static'

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__,template_folder=flask_template_path,static_folder=flask_static_path)

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/detect', methods=['POST'])
def detect_spam():
    email_text = request.form.get('emailText')
    result = "SPAM" if predict_spam(email_text) else "HAM"
    print(result)
    return result  # This will be the response sent back to the AJAX request

# Replace this with your actual spam detection function
def predict_spam(email_text):
    model = joblib.load(model_path)
    feature_extraction = joblib.load(feature_extraction_path)

    input_mail_features = feature_extraction.transform([email_text])  # Wrap the email_text in a list
    prediction = model.predict(input_mail_features)

    return prediction[0] == 1

if __name__ == "__main__":
    app.run()
