import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting test and train data")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                #"XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor()
            }

            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{
                    'fit_intercept':[True]
                    },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "KNN":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],

                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                """"XGBoost":{
                    #'learning_rate':[0.1,.01,.05,.001],
                    #'n_estimators': [8,16,32,64,128,256]
                },"""
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
            }

                # this functioon evaluate models is from utils.py
            model_report:dict = evaluate_models(models=models,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,params=params) #evaluate the models

            best_model_score = max(sorted(model_report.values())) #get the best model score

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #get the best model name
            best_model = models[best_model_name] #get the best model
            if best_model_score < 0.6: #if the best model score is less than 0.6
                logging.error("The best model score is less than 0.6") #log the error
                raise CustomException("The best model score is less than 0.6",sys) #raise an exception
            logging.info(f"The best model is {best_model_name} with a score of {best_model_score}") #log the best model name and score

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model #save the best model to the file path mentioned in the config file.
            )

            predicted = best_model.predict(x_test) #predict the model on the test data
            r2_square = r2_score(y_test,predicted) #calculate the r2 score
            logging.info(f"The r2 score of the model is {r2_square}") #log the r2 score
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)