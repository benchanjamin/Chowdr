import requests
import json
from flask import Flask, render_template, jsonify, redirect, url_for, request, session
from flask_sqlalchemy import SQLAlchemy
from random import choice, randint
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/hackathon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SECRET_KEY'] = 'secret'

db = SQLAlchemy(app)
model = -1 # initialized later
breeds_arr = -1 # initialized later
user = -1 # initialized on login

class Dogs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(50), nullable=False)

class Cats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(50), nullable=False)

class User(db.Model):
    name = db.Column(db.String(50), primary_key= True)
    left_swipes = db.Column(db.Integer, default=0, nullable=False)
    right_swipes = db.Column(db.Integer, default=0, nullable=False)
    breed_scores = db.Column(db.Text)


@app.route('/', methods=['GET'])
def index_get():
  breed = gen_breed()
  if breed["species"] == "dog":
    url = 'https://dog.ceo/api/breed/'+breed["id"]+'/images/random'
    data = requests.get(url)
    data = data.json()
    image = data["message"]
  else:
    url = 'https://api.thecatapi.com/v1/images/search?breed_id='+breed["id"]
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
  return render_template('image.html', url=image, breed_name=breed["name"], breed_id=breed["id"], user=None)

@app.route('/', methods=['POST'])
def index_post():
  if "breed_scores" not in session:
    in_file = open("ids.json")
    session["breed_scores"] = json.load(in_file)
  breed_scores = session.pop("breed_scores")
  if request.form['submit_button'] == 'Left':
    breed_scores[request.form['breed_id']] = breed_scores[request.form['breed_id']] - 5
  elif request.form['submit_button'] == 'Right':
    breed_scores[request.form['breed_id']] = breed_scores[request.form['breed_id']] + 2
  session["breed_scores"] = breed_scores
  load_model()
  update_model(request.form['breed_id'], request.form['submit_button'] == 'Right')

  messagesDict = {}
  messagesDict["show"] = False
  if request.form['submit_button'] == 'Left':
    newSwiperName = request.form.get('swiperName')
    if newSwiperName:
      existingNameLeft = User.query.filter_by(name = newSwiperName).first()
      if existingNameLeft:
        existingNameLeft.left_swipes += 1
        db.session.commit()
        messagesDict['show'] = True
        messagesDict["name"] = existingNameLeft.name
      if not existingNameLeft:
        newUser = User(name = newSwiperName, left_swipes= 1)
        db.session.add(newUser)
        db.session.commit()
        messagesDict['show'] = True
        messagesDict["name"] = newUser.name
    messages = json.dumps(messagesDict)
    session['messages'] = messages
    print("POST",session.get('messages'))
    return redirect(url_for('.index_get', messages=messages))
  elif request.form['submit_button'] == 'Right':
    newSwiperName = request.form.get('swiperName')
    if newSwiperName:
      existingName = User.query.filter_by(name=newSwiperName).first()
      if existingName:
        existingName.right_swipes = existingName.right_swipes + 1
        db.session.commit()
        messagesDict['show'] = True
        messagesDict["name"] = existingName.name
      if not existingName:
        newUser = User(name=newSwiperName, right_swipes= 1)
        db.session.add(newUser)
        db.session.commit()
        messagesDict['show'] = True
        messagesDict["name"] = newUser.name
    messages = json.dumps(messagesDict)
    session['messages'] = messages
    print("POST",session.get('messages'))
    return redirect(url_for('.index_get', messages=messages))
  else:
    return redirect(url_for('.index_get'))

@app.route('/login', methods=['GET'])
def login_get():
  return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
  global user
  if request.form['submit_button'] == 'Login':
    # Needs validation here
    user = User.query.filter_by(name = newSwiperName).first()
    load_model(email=request.form['email'])
    session["email"] = request.form['email']
  elif request.form['submit_button'] == 'Register':
    # Needs validation here and make sure to cache password
    user = User(name = newSwiperName)
    db.session.add(newUser)
    db.session.commit()
    session["email"] = request.form['email']
  return redirect(url_for('.index_get'))

def load_model(email=None):
  global model
  if email != None:
    print("LOGGED IN AS " + email)
  if email != None and os.path.exists('./user_models/'+email+'_model'):
    print("loaded old model")
    model = tf.keras.models.load_model('./user_models/'+email+'_model')
  elif model == -1:
    model = tf.keras.models.load_model('./breed_model')

def load_breeds():
  global breeds_arr
  if breeds_arr == -1:
    in_file = open("breeds.json", "r") 
    breeds_arr = json.load(in_file)

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
    goal[0] = randint(0,207)
  optimizer = keras.optimizers.SGD(lr=0.1)

  train_len = 4
  train_inputs = np.random.rand(train_len,208)
  for i in range(train_len):
    train_inputs[i] = train_inputs[i]/np.max(train_inputs[i])
  train_labels = np.argmax(train_inputs, axis=1)
  train_inputs = np.append(train_inputs, [scores], axis=0)
  train_labels = np.append(train_labels, goal)
  model.fit(train_inputs,train_labels,epochs=10, verbose=0)

  if "email" in session:
    model.save('./user_models/'+session["email"]+'_model',save_format='tf')

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
  scores = scores + (1 - np.min(scores))
  scores = scores / np.max(scores)
  
  breed_probs = model.predict(tf.expand_dims(scores, axis=0))
  breed_index = np.argmax(breed_probs[0])
  '''
  if randint(1,100) > 50:
    breed_index = randint(0,207)
  '''
  return breeds_arr[breed_index]


if __name__ == '__main__':
  db.create_all()
  app.run(host='0.0.0.0', port=8080)