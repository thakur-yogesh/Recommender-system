from django.apps import AppConfig
import pickle


class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'
    path = 'recommender\model\diab_model_final.sav'
    predictor = pickle.load(open(path, 'rb'))
    