from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect_spam():
#     email_content = request.json.get('email_content')
#     # Placeholder for model prediction
#     # result = model.predict(email_content)
#     result = "spam"  # This is just a placeholder, replace with actual model call later
#     return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
