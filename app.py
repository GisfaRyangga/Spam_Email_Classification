from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route()
def detect_spam():
    model = joblib.load('stack.model')
    feature_extraction = joblib.load('feature_extraction.pkl')

    def predict_spam(email_text):
        input_mail_features = feature_extraction.transform(email_text)
        prediction = model.predict(input_mail_features)

        if prediction == 0:
            return "HAM MAIL"
        else:
            return "SPAM MAIL"

    # implement fungsi
    input_mail = ["You're receiving this email because you turned on Location History, a Google Account-level setting that creates Timeline, a personal map of your visited places, routes, and trips."]
    result = predict_spam(input_mail)
    print(f'Text: {input_mail[0]}\nPrediction: {result}')

# @app.route('/detect', methods=['POST'])
# def detect_spam():
#     email_content = request.json.get('email_content')
#     # Placeholder for model prediction
#     # result = model.predict(email_content)
#     result = "spam"  # This is just a placeholder, replace with actual model call later
#     return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
