from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_spam():
    email_text = request.form.get('emailText')
    result = "SPAM" if predict_spam(email_text) else "HAM"
    print(result)
    return result  # This will be the response sent back to the AJAX request

# Replace this with your actual spam detection function
def predict_spam(email_text):
    model = joblib.load('./static/model/stack.model')
    feature_extraction = joblib.load('./static/model/feature_extraction.pkl')

    input_mail_features = feature_extraction.transform([email_text])  # Wrap the email_text in a list
    prediction = model.predict(input_mail_features)

    return prediction[0] == 1

if __name__ == "__main__":
    app.run(debug=True)


# @app.route('/yep', methods=['POST'])
# def detect_spam():
#     model = joblib.load('./static/model/stack.model')
#     feature_extraction = joblib.load('./static/model/feature_extraction.pkl')

#     def predict_spam(email_text):
#         input_mail_features = feature_extraction.transform(email_text)
#         prediction = model.predict(input_mail_features)

#         if prediction == 0:
#             return "HAM MAIL"
#         else:
#             return "SPAM MAIL"

#     # implement fungsi
#     input_mail = ["You're receiving this email because you turned on Location History, a Google Account-level setting that creates Timeline, a personal map of your visited places, routes, and trips."]
#     result = predict_spam(input_mail)
#     print(result)

if __name__ == '__main__':
    app.run(debug=True)
