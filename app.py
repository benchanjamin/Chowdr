import requests
import json
from flask import Flask, render_template, redirect, url_for, request, session, flash
import numpy as np
from random import randint
import tensorflow as tf
import os
import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import ValidationError, DataRequired, Length, EqualTo, Email
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user


# ################### CONFIG #################################

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/hackathon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config.from_object(Config)


# ################### INIT #################################


db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
model = -1  # initialized later
breeds_arr = -1  # initialized later

# ################### DATABASE #################################


class Users(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String, unique=True, nullable=True)
    email = db.Column(db.String, index=True, unique=True, nullable=True)
    hashed_password = db.Column(db.String, nullable=True, index=True)
    created_date = db.Column(db.DateTime, default=datetime.datetime.now, index=True)
    left_swipes = db.Column(db.Integer, default=0, nullable=False)
    right_swipes = db.Column(db.Integer, default=0, nullable=False)
    breed_scores = db.Column(db.String, nullable=True)
    like_to_dislike_ratio = db.Column(db.Float(precision=3), nullable=False, default=0)

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)

    def __repr__(self):
        return '<Username: {}>'.format(self.username)


# ################### INDEX #################################


@app.route('/', methods=['GET'])
def index_get():
    breed = gen_breed()
    user = None
    if 'email' in session:
        user = Users.query.filter_by(username=session["email"]).first()
    if breed["species"] == "dog":
        url = 'https://dog.ceo/api/breed/' + breed["id"] + '/images/random'
        data = requests.get(url)
        data = data.json()
        image = data["message"]
    else:
        url = 'https://api.thecatapi.com/v1/images/search?breed_id=' + breed["id"]
        data = requests.get(url)
        data = data.json()
        image = data[0]["url"]
    print("GET", session.get('messages'))

    return render_template('image.html', url=image, breed_name=breed["name"], breed_id=breed["id"], user=user)


@app.route('/', methods=['POST'])
def index_post():
    user = -1
    # global user
    if "email" in session:
        user = Users.query.filter_by(username=session['email']).first()
    if "breed_scores" not in session:
        in_file = open("ids.json")
        session["breed_scores"] = json.load(in_file)
    breed_scores = session.pop("breed_scores")
    if request.form['submit_button'] == 'Left':
        breed_scores[request.form['breed_id']] = breed_scores[request.form['breed_id']] - 5
    elif request.form['submit_button'] == 'Right':
        breed_scores[request.form['breed_id']] = breed_scores[request.form['breed_id']] + 2
    session["breed_scores"] = breed_scores
    if user != -1:
        user.breed_scores = json.dumps(breed_scores)
        db.session.commit()
    load_model()
    update_model(request.form['breed_id'], request.form['submit_button'] == 'Right')
    messages_dict = {"show": False}
    if request.form['submit_button'] == 'Left':
        if user != -1:
            user.left_swipes = user.left_swipes + 1
            db.session.commit()
            if user.right_swipes == 0:
                user.like_to_dislike_ratio = 0
                db.session.commit()
            else:
                user.like_to_dislike_ratio = round(user.right_swipes / user.left_swipes, 3)
                db.session.commit()
            messages_dict['show'] = True
            messages_dict["name"] = user.username
        messages = json.dumps(messages_dict)
        session['messages'] = messages
        print("POST", session.get('messages'))
        return redirect(url_for('.index_get'))
    elif request.form['submit_button'] == 'Right':
        if user != -1:
            user.right_swipes = user.right_swipes + 1
            db.session.commit()
            if user.left_swipes == 0:
                user.like_to_dislike_ratio = 0
                db.session.commit()
            else:
                user.like_to_dislike_ratio = round(user.right_swipes / user.left_swipes, 3)
                db.session.commit()
            messages_dict['show'] = True
            messages_dict["name"] = user.username
        messages = json.dumps(messages_dict)
        session['messages'] = messages
        print("POST", session.get('messages'))
        return redirect(url_for('.index_get'))
    else:
        db.session.commit()
        return redirect(url_for('.index_get'))


# ################### LOGIN #################################

class LoginForm(FlaskForm):
    username = StringField('email', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Log In')


@login.user_loader
def load_user(id):
    return Users.query.get(int(id))


@app.route('/login', methods=["GET"])
def login_get():
    if current_user.is_authenticated:
        return redirect(url_for('index_get'))
    form = LoginForm()
    return render_template("login2.html", form=form)


@app.route('/login', methods=['POST'])
def login_post():
    # global user
    form = LoginForm(request.form)
    if form.validate():
        user = Users.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login_get'))
        login_user(user, remember=form.remember_me.data)
        session["breed_scores"] = json.loads(user.breed_scores)
        load_model(email=user.username)
        return redirect(url_for('index_get'))
    return render_template('login2.html', form=form)


    '''
    if request.form['submit_button'] == 'Login':
        # Needs validation here
        user = Users.query.filter_by(username=form.username.data).first()
        session["breed_scores"] = json.loads(user.breed_scores)
        load_model(email=form.username.data)
        session["email"] = request.form['email']
    '''

    '''
    if request.form['submit_button'] == 'Register':
        # Needs validation here and make sure to cache password
        user = Users(username=form.username.data)
        if "breed_scores" not in session:
            in_file = open("ids.json")
            session["breed_scores"] = json.load(in_file)
        breed_scores = session.get("breed_scores")
        user.breed_scores = json.dumps(breed_scores)
        db.session.add(user)
        db.session.commit()
        session["email"] = request.form['email']
    '''
    return redirect(url_for('.index_get'))


# ################### REGISTER #################################

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[Length(min=4, max=25)])
    email = StringField('Email Address', validators=[Length(min=6, max=35), Email()])
    password = PasswordField('New Password', [
        DataRequired(),
        EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = Users.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = Users.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = Users(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        if "breed_scores" not in session:
            in_file = open("ids.json")
            session["breed_scores"] = json.load(in_file)
        breed_scores = session.get("breed_scores")
        user.breed_scores = json.dumps(breed_scores)
        db.session.commit()
        session["email"] = form.username.data
        return redirect(url_for('login_get'))
    return render_template('register.html', title='Register', form=form)


'''

@app.route('/account/register', methods=['GET'])
@response(template_file='account/register.html')
def register_get():
    return {
        'user_id': cookie_auth.get_user_id_via_auth_cookie(flask.request),
    }


@app.route('/account/register', methods=['POST'])
@response(template_file='account/register.html')
def register_post():
    data = request_dict.create(default_val='')

    name = data.name
    email = data.email.lower().strip()
    password = data.password.strip()

    if not name or not email or not password:
        return {
            'name': name,
            'email': email,
            'password': password,
            'error': "Some required fields are missing.",
            'user_id': cookie_auth.get_user_id_via_auth_cookie(flask.request),
        }

    user = user_service.create_user(name, email, password)
    if not user:
        return {
            'name': name,
            'email': email,
            'password': password,
            'error': "A user with that email already exists.",
            'user_id': cookie_auth.get_user_id_via_auth_cookie(flask.request),
        }

    resp = flask.redirect('/account')
    cookie_auth.set_auth(resp, user.id)

    return resp
'''

# ################### LOGOUT #################################


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index_get'))

'''
@app.route('/account/logout')
def logout():
    resp = flask.redirect('/')
    cookie_auth.logout(resp)

    return resp
'''

# ################### PROFILE #################################


@app.route('/profile')
def profile():
    return "Profile"


# ################### MACHINE LEARNING #################################

# Sets the global model variable.
# On log in, this should load a previously saved model.
# Otherwise, loads the base model.
def load_model(email=None):
    global model
    if email != None:
        print("LOGGED IN AS " + email)
    if email != None and os.path.exists('./user_models/' + email + '_model'):
        print("loaded old model")
        model = tf.keras.models.load_model('./user_models/' + email + '_model')
    elif model == -1:
        model = tf.keras.models.load_model('./breed_model')


# Sets the global breeds_arr variable.
def load_breeds():
    global breeds_arr
    if breeds_arr == -1:
        in_file = open("breeds.json", "r")
        breeds_arr = json.load(in_file)


# Based on the current breed and whether the user swiped
# right in reaction to the picture, update_model
# updates the neural network to adapt to this preference.
# It assumes that breed_scores has been updated.
# In addition, if the user is logged in (indicated by
# their email being stored in session), the updated
# model is saved in a folder under the user_model directory.
def update_model(breed, right):
    load_model()
    load_breeds()
    if "breed_scores" not in session:
        in_file = open("ids.json")
        session["breed_scores"] = json.load(in_file)
    breed_scores = session.get("breed_scores")
    scores = []
    for score in breed_scores.values():
        scores.append(score)
    scores = np.array(scores)
    scores = scores + (1 - np.min(scores))
    scores = scores / np.max(scores)

    goal = [0]
    if right:
        goal[0] = list(breed_scores.keys()).index(breed)
    else:
        goal[0] = randint(0, 207)

    train_len = 2
    train_inputs = np.random.rand(train_len, 208)
    for i in range(train_len):
        train_inputs[i] = train_inputs[i] / np.max(train_inputs[i])
    train_labels = np.argmax(train_inputs, axis=1)
    train_inputs = np.append(train_inputs, [scores, scores, scores], axis=0)
    train_labels = np.append(train_labels, [goal, goal, goal])
    model.fit(train_inputs, train_labels, epochs=10, verbose=0)

    if "email" in session:
        model.save('./user_models/' + session["email"] + '_model', save_format='tf')


# This method is responsible for generating the breed to # be displayed based on previous user interaction.
# It assumes that breed_scores and model are updated.
# Returns a dictionary whose keys are 'name', 'id', and
# 'species,' all of which describe the breed.
def gen_breed():
    load_model()
    load_breeds()
    if "breed_scores" not in session:
        in_file = open("ids.json")
        session["breed_scores"] = json.load(in_file)
    breed_scores = session.get("breed_scores")
    scores = []
    for score in breed_scores.values():
        scores.append(score)
    scores = np.array(scores)
    # returns element-wise minimum between two arrays?
    scores = scores + (1 - np.min(scores))
    scores = scores / np.max(scores)

    breed_probs = model.predict(tf.expand_dims(scores, axis=0))
    breed_index = np.argmax(breed_probs[0])
    return breeds_arr[breed_index]


# ################### RUN #################################

if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=8080)
