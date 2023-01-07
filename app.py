from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM

# model = pickle.load(open('model.pkl', 'rb'))
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# features = np.array([[29,4, 6]])
# print("Predicted Delivery Time = ", loaded_model.predict(features)[0][0],'mins')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def take_input():
    age = float(request.form.get('age'))
    rating = float(request.form.get('rating'))
    distance = float(request.form.get('distance'))
    print(age, rating, distance)
    features = np.array([[age, rating, distance]])
    return render_template('index.html', data=loaded_model.predict(features)[0][0])


if __name__ == '__main__':
    app.run(debug=True)
