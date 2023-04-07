from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import RecommenderConfig
import json
import numpy as np
import pickle
import pandas as pd
import random

class call_model(APIView):

    def diab_pred(self ,person, df):
        in_db = 0
        row = 0
        for i in df['PERSON']:
            if i == person:
                in_db = 1
                break
            row = row + 1
        if in_db == 1:
            inputs = [i for i in df.iloc[row][1:len(df.iloc[row]) - 1]]
            inputs = np.asarray(inputs)
            inputs = inputs.reshape(1,-1)
            response = RecommenderConfig.diab_predictor.predict(inputs)
            return response
        else:
            return [0]
    
    def heart_pred(self , person , df):
        in_db = 0
        row = 0
        for i in df['PERSON']:
            if i == person:
                in_db = 1
                break
            row += 1
        if in_db == 1:
            inputs = [i for i in df.iloc[row][1:len(df.iloc[row]) - 1]]
            inputs = np.asarray(inputs)
            inputs = inputs.reshape(1,-1)
            response = RecommenderConfig.heart_predictor.predict(inputs)
            return response
        else:
            return [0]
        

    
    def parkin_pred(self , person , df):
        in_db = 0
        row = 0
        for i in df['PERSON']:
            if i == person:
                in_db = 1
                break
            row += 1
        if in_db == 1:
            inputs = [i for i in df.iloc[row][1:len(df.iloc[row]) - 1]]
            inputs = np.asarray(inputs)
            inputs = inputs.reshape(1,-1)
            response = RecommenderConfig.parkin_predictor.predict(inputs)
            return response
        else:
            return [0]
    
    def covid_pred(self , person , df):
        in_db = 0
        row = 0
        for i in df['PERSON']:
            if i == person:
                in_db = 1
                break
            row += 1
        if in_db == 1: 
            inputs = [i for i in df.iloc[row][2:len(df.iloc[row])]]
            inputs = np.asarray(inputs)
            inputs = inputs.reshape(1,-1)
            response = RecommenderConfig.covid_predictor.predict(inputs)
            print("response == ",response)
            return response
        else:
            return [0]



    def get(self , request):
        if request.method == 'GET':
            #loading csv's
            df_diab = pd.read_csv('recommender\model\diab_db.csv')
            df_heart = pd.read_csv('recommender\model\heart_db.csv')
            df_parkin = pd.read_csv('recommender\model\parkin_db.csv')
            df_covid = pd.read_csv('recommender\model\covid_db.csv')


            #getting a random user from one of the database
            person = str(df_diab['PERSON'][random.randint(0,len(df_diab))])
            #params = request.GET.get('donotmatter') #just to check whehter we're getting the request or not

            #print(params)
            #making predictionns
            diab_prediction = self.diab_pred(person , df_diab)
            heart_prediction = self.heart_pred(person,df_heart)
            parki_prediction = self.parkin_pred(person,df_parkin)
            covid_predictionn = self.covid_pred(person,df_covid)


            #adding the predictions to the dictionary
            res_dict = {'person':person , 
                        'diab_prediction':str(diab_prediction[0]),
                        'heart_prediction':str(heart_prediction[0]),
                         'parkinson_prediction':str(parki_prediction[0]),
                          'covid_prediction' :str(covid_predictionn[0]) }  
            

            y = json.dumps(res_dict)
            print(y)
            return JsonResponse(res_dict , safe=False)
