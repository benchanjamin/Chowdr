import requests
import json
from flask import Flask, render_template, redirect, url_for, request, session
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from random import randint
import tensorflow as tf
import os

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/hackathon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SECRET_KEY'] = 'secret'

db = SQLAlchemy(app)
model = -1  # initialized later
breeds_arr = -1  # initialized later
user = -1  # initialized on login


class User(db.Model):
    name = db.Column(db.String(50), primary_key=True)
    left_swipes = db.Column(db.Integer, default=0, nullable=False)
    right_swipes = db.Column(db.Integer, default=0, nullable=False)
    breed_scores = db.Column(db.String(100000), nullable=True)
    like_to_dislike_ratio = db.Column(db.Float(precision=3), nullable=True)


@app.route('/', methods=['GET'])
def index_get():
    breed = gen_breed()
    user = None
    if 'email' in session:
        user = User.query.filter_by(name=session["email"]).first()
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

    if 'messages' in session:
        messages = session['messages']
        messages = json.loads(messages)
        '''
        try:
          user = User.query.filter_by(name = messages["name"]).first()
        except:
          user = None
        '''
    return render_template('image.html', url=image, breed_name=breed["name"], breed_id=breed["id"], user=user)


@app.route('/', methods=['POST'])
def index_post():
    global user
    if "email" in session:
        user = User.query.filter_by(name=session['email']).first()
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

    messagesDict = {}
    messagesDict["show"] = False
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
            messagesDict['show'] = True
            messagesDict["name"] = user.name
        messages = json.dumps(messagesDict)
        session['messages'] = messages
        print("POST", session.get('messages'))
        return redirect(url_for('.index_get', messages=messages))
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
            messagesDict['show'] = True
            messagesDict["name"] = user.name
        messages = json.dumps(messagesDict)
        session['messages'] = messages
        print("POST", session.get('messages'))
        return redirect(url_for('.index_get', messages=messages))
    else:
        db.session.commit()
        return redirect(url_for('.index_get'))

@app.route('/login', methods=["GET"])
def login_get():
  return render_template("login.html")


@app.route('/login', methods=['POST'])
def login_post():
    global user
    if request.form['submit_button'] == 'Login':
        # Needs validation here
        user = User.query.filter_by(name=request.form['email']).first()
        session["breed_scores"] = json.loads(user.breed_scores)
        load_model(email=request.form['email'])
        session["email"] = request.form['email']
    if request.form['submit_button'] == 'Register':
        # Needs validation here and make sure to cache password
        user = User(name=request.form['email'])
        if "breed_scores" not in session:
            in_file = open("ids.json")
            session["breed_scores"] = json.load(in_file)
        breed_scores = session.get("breed_scores")
        user.breed_scores = json.dumps(breed_scores)
        db.session.add(user)
        db.session.commit()
        session["email"] = request.form['email']
    return redirect(url_for('.index_get'))


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


if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=8080)