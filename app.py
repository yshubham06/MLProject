from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application = Flask(__name__)#creating an instance of the Flask class

app = application#creating an alias for the instance of the Flask class

## Route for a home page

@app.route('/') #decorator to specify the route
def index(): #function to return the response
    return render_template('index.html')#return the response

@app.route('/predict',methods=['GET','POST'])#decorator to specify the route
def predict_datapoint():#function to return the response
    if request.method == 'GET':#check if the request method is GET
        return render_template('home.html')#return the response
    else:
        data = CustomData(#create an instance of the CustomData class
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )
        pred_df = data.get_data_as_data_frame() #get the data as a dataframe object
        logging.info(f"inside predict_datapoint, pred_df: {pred_df}")
        print(pred_df)

        predict_pipeline = PredictPipeline()
        logging.info(f"predict_pipeline object created successfully")
        results = predict_pipeline.predict(pred_df)
        logging.info(f"results: {results}")
        return render_template('home.html',results=results[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)
