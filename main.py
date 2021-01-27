import requests
import json
from flask import Flask, render_template, jsonify, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from random import choice

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/hackathon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Dogs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(50), nullable=False)

class Cats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(50), nullable=False)

class User(db.Model):
    name = db.Column(db.String(50), primary_key= True)
    left_swipes = db.Column(db.Integer)
    right_swipes = db.Column(db.Integer)

class UniqueURLs(db.Model):
  urls = db.Column(db.String(50), primary_key=True)


@app.route('/', methods=['GET','POST'])
def index():
  if request.method == "POST":
    username = request.form.get('username')
    existingName = User.query.filter_by(name = username).first()
    if not existingName:
      newUser = User(name = username, left_swipes=0, right_swipes=0)
      db.session.add(newUser)
      db.session.commit()
    return render_template('image.html')
  else:
    return render_template('index.html')


@app.route('/<username>', methods=['GET'])
def account_get(username):

  url = 'https://dog.ceo/api/breeds/image/random'
  data = requests.get(url)
  data = data.json()
  return render_template('image.html', url = data["message"])

@app.route('/<username>', methods=['POST'])
def account_post(username):
  url = 'https://dog.ceo/api/breeds/image/random'
  data = requests.get(url)
  data = data.json()
  if request.form['submit_button'] == 'Swipe Left':
    existingName = User.query.filter_by(name = username).first()
    if existingName:
      existingName.left_swipes += 1
      db.session.commit()
    if not existingName:
      newUser = User(username = username, left_swipes= 1)
      db.session.add(newUser)
      db.session.commit()
    return render_template('image.html', url = data["message"]) 
  elif request.form['submit_button'] == 'Swipe Right':
    newSwiperName = request.form.get('swiperName')
    if newSwiperName:
      existingName = User.query.filter_by(name = newSwiperName).first()
      if existingName:
        existingName.right_swipes += 1
        db.session.commit()
      if not existingName:
        newUser = User(name = newSwiperName, right_swipes=1)
        db.session.add(newUser)
        db.session.commit()
    return render_template('image.html', url = data["message"])
  else:
    return render_template('image.html', url = data["message"])


@app.route('/_json_file')
def get_json():
  return jsonify()

@app.route('/user/', defaults={'username': None})
@app.route('/user/<username>')
def generate_user(username):
	if not username:
		username = requests.args.get('username')

	if not username:
		return 'Sorry error something, malformed request.'

	return render_template('personal_user.html', user=username)


if __name__ == '__main__':
  db.create_all()
  app.run(host='0.0.0.0', port=8080)