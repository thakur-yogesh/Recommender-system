from django.apps import AppConfig
import pickle


class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'
    path1 = 'recommender\model\diab_model_final.sav'
    diab_predictor = pickle.load(open(path1, 'rb'))
    path2 = 'recommender\model\heart_model_final.sav'
    heart_predictor = pickle.load(open(path2,'rb'))
    path3 = 'recommender\model\parkin_model_final.sav'
    parkin_predictor = pickle.load(open(path3,'rb'))
    path4 = 'recommender\model\covid_model_final.sav'
    covid_predictor = pickle.load(open(path4 , 'rb'))