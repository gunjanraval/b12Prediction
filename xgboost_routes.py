import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
from datetime import timedelta

from utils import onehotCategorical

app = Flask(__name__,template_folder='templates')

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    model = joblib.load('model.pkl')
    if request.method=='POST':
        
        HB= float(request.form['HB'])
        RBCS= float(request.form['RBCS'])
        HCT= float(request.form['HCT'])
        MCV= float(request.form['MCV'])
        MCH= float(request.form['MCH'])
        MCHC= float(request.form['MCHC'])
        RDWCV= float(request.form['RDW-CV'])
        RDWSD= float(request.form['RDW-SD'])
        TC= float(request.form['TC'])
        PLT= float(request.form['PLT'])
        data = [[HB,RBCS,HCT,MCV,MCH,MCHC,RDWCV,RDWSD,TC,PLT]]
        df = pd.DataFrame(data,columns=['HB','RBCS','HCT','MCV','MCH','MCHC','RDW-CV','RDW-SD','TC','PLT'])
        #data = [[1,1270,1,0,3,1,0,6,9,15,2019,37,132,0,0]]
        #df = pd.DataFrame(data,columns=['Store','CompetitionDistance','Promo','SchoolHoliday','StoreType','Assortment','StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','CompetitionOpen','PromoOpen','IsPromoMonth'])     
        print(df)
        prediction = model.predict(df)
        
        #prediction = model.predict(entered_li.values.reshape(1, -1))
        label = str(prediction[0])
        if label=="0":
        	label="B12 Normal (Greater than 190)"
        else:
        	label="B12 Deficient (Less than 190)"
        return render_template('index.html', label=label)

if __name__ == '__main__':
    # start API
    app.run()
