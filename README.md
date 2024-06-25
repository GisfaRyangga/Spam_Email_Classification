# Email SPAM/HAM Detector

This is a basic Flask web application designed to serve as the front end for an email spam detection model. The current implementation includes a form where users can input email content to be analyzed for spam.

## Usage

- On the main page, you will see a form where you can paste the content of an email.
- Click the "Submit" button to send the email content to the AI model for detection.
- The result will be displayed on the page, indicating whether the email is SPAM or HAM.

## Installation and Setup

### Prerequisites

- Python 3.10
- Virtual environment (optional but recommended)

### Steps to Set Up the Project

1. **Clone the repository:**

    ```sh
    https://github.com/GisfaRyangga/Spam_Email_Classification.git
    cd Spam_Email_Classification
    ```

2. **Set up a virtual environment:**

    ```sh
    python -m venv venv
    venv\Scripts\activate  # this is for Windows
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Flask Application

1. **Ensure you are in the project directory and the virtual environment is activated.**

2. **Run the Flask application:**

    ```sh
    flask run
    ```

3. **Open a web browser and navigate to `http://127.0.0.1:5000/` to view the application.**


**link stack.model** --> https://drive.google.com/file/d/1ziIbm-rRGekKm9yIFSU0A_899hxuJ35-/view?usp=sharing

**link feature_extraction.pkl**--> https://drive.google.com/file/d/1Is0PFmnfUmwcFL13YDHr4BNQISPmeCiE/view?usp=sharing


<h2>Created with love of blood and tears</h2>

Ninis - Ryangga - Andhika
