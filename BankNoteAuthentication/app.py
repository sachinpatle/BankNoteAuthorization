# -*- coding: utf-8 -*-

from flask import Flask,request,render_template
# print(flask.__version__)

import numpy as np
import pandas as pd
import pickle
pickle_in =open("banknotauthenticationmodel.pkl","rb")
model = pickle.load(pickle_in)   
df=pd.DataFrame()
app = Flask(__name__)
@app.route("/")
def home():
     return render_template('index.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/predict",methods=['POST'])
def predict():
     global df
     input_features = [float(i) for i in request.form.values()]
     features_values = np.array(input_features)
     output = model.predict([features_values])
     if output == 0:
         predicted_marks1 = "Not Defected"
     else:
          predicted_marks1 = "Defected"
     return  render_template('index.html',predicted_marks=predicted_marks1)
if __name__ == "__main__":
     app.run()

