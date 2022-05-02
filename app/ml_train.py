# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pydantic import BaseModel
import joblib


# 2. Class which describes the features
class themodelfunc(BaseModel):
    temperature: float 
    pulse: int 
    sys: int 
    dia: int
    rr: int 
    sats: int
    clientid: int


# 3. Class for training the model and making predictions
class themodelModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    #def __init__(self):
        #self.df = pd.read_csv('client/101/data.csv')
        #self.model_fname_ = 'model.pkl'
        #try:
         #   self.model = joblib.load(self.model_fname_)
        #except Exception as _:
         #   self.model = self._train_model()
          #  joblib.dump(self.model, self.model_fname_)

    # 4. Perform model training using the RandomForest classifier
    def _train_model(self):
        X = self.df.drop('covid19_test_results', axis=1)
        y = self.df['covid19_test_results']
        knn= KNeighborsClassifier(n_neighbors=3)
        model = knn.fit(X, y)
        return model


    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted value with its respective probability
    def predict_func(self, temperature, pulse, sys, dia, rr, sats, clientid):
        file1='client/'
        file=file1+str(clientid)+ '/'+ str(clientid)+ 'data.csv'
        
        self.df = pd.read_csv(file)
        model='model'
        self.model_fname_ = file1+str(clientid)+'/'+str(model)+str(clientid)+'.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            
            self.model = self._train_model()
            joblib.dump(self.model, file1+str(clientid)+'/'+str(model)+str(clientid)+'.pkl')
        data_in = [[temperature, pulse, sys, dia, rr, sats]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        
        return prediction[0].tolist(), probability.tolist()

model=themodelModel()