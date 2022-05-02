import requests 

new_measurement = {
    'temperature': 38,
    'pulse': 100,
    'sys': 80,
    'dia': 60,
    'rr': 10,
    'sats': 50,
    'clientid':101

}

response = requests.post('http://127.0.0.1:8000/predict', json=new_measurement)
print(response.content)