from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Email
import cv2
from langdetect import detect
from googletrans import Translator
import nltk
from nltk.stem import WordNetLemmatizer
import random
import string
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['WTF_CSRF_SECRET_KEY'] = 'your_csrf_secret_key'
app.config['WTF_CSRF_ENABLED'] = True

# Enable CSRF protection
csrf = CSRFProtect(app)

# Configure MySQL connection
db_config = {
    'host': 'localhost',
    'user': 'root',
    'port': '3306',
    'password': '20052020',  # Replace with your MySQL root password
    'database': 'autism_detection'
}

# Configure Flask-Mail for sending emails
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'nancy2005nov@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'ntgo jlmo jffs geml'  # Replace with your email password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Serializer for generating password reset tokens
s = URLSafeTimedSerializer(app.secret_key)

# Establish database connection
def get_db_connection():
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            print("Connection to MySQL database was successful!")
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
    return connection

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

class MentalHealthChatbot:
    def save_model(self):
        # Save the trained model as an H5 file
        self.model.save(self.model_file)
        print(f"Model saved as {self.model_file}")

    def __init__(self, intents_file, model_file='chatbot_model.h5'):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(intents_file).read())
        self.model_file = model_file

        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = list(string.punctuation)

        self.initialize_data()
        self.prepare_training_data()

        # Load or build model
        if os.path.exists(self.model_file):
            self.load_model()
        else:
            self.build_model()
            self.train()
            self.save_model()

    def initialize_data(self):
        # Extract words and classes from intents
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and clean words
        self.words = [self.lemmatizer.lemmatize(word.lower())
                      for word in self.words
                      if word not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def prepare_training_data(self):
        # Prepare training data
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        # Shuffle and convert to numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)

        self.train_x = np.array([item[0] for item in training])
        self.train_y = np.array([item[1] for item in training])

    def build_model(self):
        # Build and compile neural network
        self.model = Sequential([
            Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.train_y[0]), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=200, batch_size=5, verbose=1):
        # Train the model
        self.model.fit(self.train_x, self.train_y,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose)

    def load_model(self):
        # Load the pre-trained model from an H5 file
        self.model = tf.keras.models.load_model(self.model_file)
        print(f"Model loaded from {self.model_file}")

    def clean_up_sentence(self, sentence):
        # Tokenize and lemmatize input sentence
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def create_bow(self, sentence):
        # Create bag of words array
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        # Filter predictions below threshold
        bow = self.create_bow(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.classes[r[0]],
                'probability': str(r[1])
            })
        return return_list

    def get_response(self, intents_list):
        # Get a random response from the predicted intent
        if not intents_list:
            return "I'm not sure how to respond to that. Could you please rephrase?"

        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']

        for intent in list_of_intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

        return "I'm not sure how to respond to that. Could you please rephrase?"

    def chat(self, user_message):
        # Predict and get response based on user message
        ints = self.predict_class(user_message)
        response = self.get_response(ints)
        return response


# Initialize the chatbot
chatbot = MentalHealthChatbot('intents.json')


@app.route("/get", methods=['GET'])
def get_bot_response():
    user_message = request.args.get('msg')
    bot_response = chatbot.chat(user_message)
    return jsonify({'response': bot_response})
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/autism')
def autism():
    return render_template('autism.html')


# Route for Mental Health test
@app.route('/mental')
def mental():
    return render_template('mental.html')


with open('disease_data.json', 'r', encoding='utf-8') as file:
    disease_data = json.load(file)

phonetic_to_hindi = {
    # Fever variations
    "bukhar": "बुखार",
    "bukhaar": "बुखार",
    "bukhaaaar": "बुखार",
    "bukhr": "बुखार",
    "bkhr": "बुखार",
    "bukr": "बुखार",

    # Cough variations
    "khansi": "खांसी",
    "kansi": "खांसी",
    "khansee": "खांसी",
    "khansii": "खांसी",

    # Headache variations
    "sir dard": "सिरदर्द",
    "sirdard": "सिरदर्द",
    "sirdardd": "सिरदर्द",
    "sirderd": "सिरदर्द",
    "surderd": "सिरदर्द",

    # Sore throat variations
    "gale mein kharash": "गले में खराश",
    "gale mein kharish": "गले में खराश",
    "gale mein khrash": "गले में खराश",
    "gale mein kharas": "गले में खराश",

    # Tiredness variations
    "thakan": "थकान",
    "thkan": "थकान",
    "thakkan": "थकान",
    "thhkan": "थकान",

    # Nausea variations
    "matli": "मतली",
    "matlee": "मतली",
    "mattli": "मतली",
    "matlli": "मतली"
}

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()
    language = request.json.get('language', 'english')

    # If the language is Hindi, handle phonetic transliterations
    if language == "hindi" and user_message in phonetic_to_hindi:
        user_message = phonetic_to_hindi[user_message]

    response_list = []

    # Find matching diseases and precautions based on the language
    for entry in disease_data:
        if (language == "english" and entry["Symptom"] in user_message) or \
           (language == "hindi" and entry["SymptomHindi"] in user_message):
            disease = entry["Disease"] if language == "english" else entry["DiseaseHindi"]
            precaution = entry["Precaution"] if language == "english" else entry["PrecautionHindi"]
            response_list.append(f"बीमारी: {disease}. सावधानी: {precaution}" if language == "hindi" else f"Disease: {disease}. Precaution: {precaution}")

    # Generate response
    if not response_list:
        response = "Sorry, I couldn't find any matching disease for your symptom."
        if language == "hindi":
            response = "माफ़ करें, मैं आपके लक्षण के लिए कोई मिलती-जुलती बीमारी नहीं ढूंढ सका।"
    else:
        response = " ".join(response_list)

    # Add consultation suggestion
    consultation_message = "For better evaluation, please consult a professional doctor."
    if language == "hindi":
        consultation_message = "बेहतर मूल्यांकन के लिए, कृपया पेशेवर डॉक्टर से परामर्श लें।"
    response += f" {consultation_message}"

    return jsonify({"response": response})
# Store results
results = {
    "round1": {"image1_time": 0, "image2_time": 0, "result": ""},
    "round2": {"image1_time": 0, "image2_time": 0, "result": ""}
}


@app.route('/round1')
def round1():
    return render_template('round1.html')

@app.route('/round2')
def round2():
    return render_template('round2.html')

@app.route('/submit_times', methods=['POST'])
def submit_times():
    data = request.json
    round_name = data.get('round')
    image1_time = data.get('image1_time', 0)
    image2_time = data.get('image2_time', 0)

    result = "non-autistic"
    if image1_time > image2_time:
        result = "autistic"

    results[round_name] = {
        "image1_time": image1_time,
        "image2_time": image2_time,
        "result": result
    }

    return jsonify({"status": "success", "result": result})

@app.route('/result')
def result():
    round1_result = results['round1']['result']
    round2_result = results['round2']['result']

    if round1_result == "autistic" and round2_result == "autistic":
        final_result = "High Possibility of Autism"
    elif round1_result == "autistic" or round2_result == "autistic":
        final_result = "Low Possibility of Autism"
    else:
        final_result = "No Possibility of Autism"

    return render_template('result.html', final_result=final_result)

# Route for signup
class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        hashed_password = generate_password_hash(password)

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.")
            return redirect(url_for('home'))

        cursor = connection.cursor()

        try:
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                flash("Username or email already exists. Please choose a different one.")
                return redirect(url_for('signup'))

            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
            connection.commit()

        except Error as e:
            flash(f"An error occurred: {e}")
            return redirect(url_for('signup'))

        finally:
            cursor.close()
            connection.close()

        session['username'] = username
        flash("Signup successful!")
        return redirect(url_for('home'))

    return render_template('signup.html', form=form)


# Route for login
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.")
            return redirect(url_for('home'))

        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND email = %s", (username, email))
        user = cursor.fetchone()

        if user:
            stored_hash = user[3]  # Assuming 'password' is the 4th column (index 3) in the 'users' table

            if check_password_hash(stored_hash, password):
                session['username'] = username
                flash("Login successful!")
                return redirect(url_for('home'))
            else:
                flash("Invalid password.")
                return redirect(url_for('login'))
        else:
            flash("Username or email not found. Please sign up first.")
            return redirect(url_for('signup'))

    return render_template('login.html', form=form)


# Route for logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))


# Forgot Password Form
class ForgotPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()

    if form.validate_on_submit():
        email = form.email.data

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.")
            return redirect(url_for('home'))

        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            token = s.dumps(email, salt='email-confirm')
            reset_link = url_for('reset_password', token=token, _external=True)

            # Send reset email
            msg = Message('Password Reset Request', sender='nancy2005nov@gmail.com', recipients=[email])
            msg.body = f'Your password reset link is {reset_link}. This link is valid for 30 minutes.'
            mail.send(msg)

            flash("A password reset link has been sent to your email.")
            return redirect(url_for('login'))
        else:
            flash("Email not found.")
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html', form=form)


# Route for resetting password
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=1800)  # 30 minutes expiration
    except SignatureExpired:
        flash("The reset link is expired.")
        return redirect(url_for('forgot_password'))

    form = ForgotPasswordForm()

    if form.validate_on_submit():
        new_password = form.password.data
        hashed_password = generate_password_hash(new_password)

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.")
            return redirect(url_for('home'))

        cursor = connection.cursor()
        cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, email))
        connection.commit()

        flash("Your password has been updated.")
        return redirect(url_for('login'))

    return render_template('reset_password.html', form=form)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)