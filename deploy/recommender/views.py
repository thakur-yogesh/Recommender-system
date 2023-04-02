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
""""
def diab_pred(person):
    df = pd.read_csv('recommender\model\diab_db.csv')
    row = 0
    for i in df['PERSON']:
        if i == person:
            break
        row = row + 1
    inputs = [i for i in df.iloc[row][:len(df.iloc[row]) - 1]]
    return inputs

"""
class call_model(APIView):

    def diab_pred(self ,person, df):
        row = 0
        for i in df['PERSON']:
            if i == person:
                break
            row = row + 1
        inputs = [i for i in df.iloc[row][1:len(df.iloc[row]) - 1]]
        inputs = np.asarray(inputs)
        inputs = inputs.reshape(1,-1)
        response = RecommenderConfig.predictor.predict(inputs)

        return response
    
    def heart_pred(self , person , df):
        pass
    
    def parkin_pred(self , person , df):
        pass



    def get(self , request):
        if request.method == 'GET':
            params = request.GET.get('sentence')
            print(params)
            df = pd.read_csv('recommender\model\diab_db.csv')
            person = str(df['PERSON'][random.randint(0,len(df))])
            prediction = self.diab_pred(person , df)
        

            res_dict = {'prediction':str(prediction[0]),
                        'person':person}  

            y = json.dumps(res_dict)
            return JsonResponse(y , safe=False)
"""
def index(request):
    if request.method == 'GET':
         path = 'recommender\model\covid_model.sav'
         predictor = pickle.load(open(path, 'rb'))
         input_data = [0,11.35,34.56]
         input_data_as_numpy_array = np.asarray(input_data)
         input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
         prediction = predictor.predict(input_data_reshaped)

         my_dict = {"insert_me": str(prediction)}
    else:
         my_dict = {"insert_me": "got nothing"}
    

    return render(request,'index.html',context=my_dict)

"""