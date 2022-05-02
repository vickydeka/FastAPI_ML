# 1. Library imports
import uvicorn
from fastapi import FastAPI
from ml_train import themodelModel, themodelfunc
import joblib

# 2. Create app and model objects
app = FastAPI()
knn_fi = 'model.pkl'

#model = joblib.load(knn_fi)


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted func with the confidence
@app.post('/predict')
def predict_func(themodel: themodelfunc):
    data = themodel.dict()
    model = themodelModel()
    prediction, probability = model.predict_func(
        data['temperature'], data['pulse'], data['sys'],data['dia'],data['rr'], data['sats'], data['clientid']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
