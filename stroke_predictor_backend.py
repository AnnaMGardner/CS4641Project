import json
import sklearn
import boto3
import os
import pickle
import array as arr
from sklearn.svm import SVC

def lambda_handler(event, context):
    vector, message = parse_event(event)
    if (not message == ""):
        return {
            'statusCode': 200,
            'body': {"Prediction":"0", "Message":message}
        }
    
    prediction = predict(vector)
    if (prediction == 0):
        return {
            'statusCode': 200,
            'body': {"Prediction":"0", "Message":""}
        }
    else:
        return {
            'statusCode': 200,
            'body': {"Prediction":"1", "Message":""}
        }
    
            
"{'Age':'','Hypertension':'','Heart Disease':'','Ever Married':'','Work Type':'','Residence Type':'','Average glucose level':'','BMI':'','Smoking':''}"

def parse_event(event):
    message = ""
    age = event["Age"] 
    hypertension = event["Hypertension"]
    heart = event["Heart Disease"]
    married = event["Ever Married"]
    work = event["Work Type"]
    residence = event["Residence Type"]
    glucose = event["Average glucose level"]
    bmi = event["BMI"]
    smoking = event["Smoking"]
    vector = [0] * 9
    
    #clean up these else ifs and add more flexibility if the json changes :)
    
    if(not age == "" and age.isDigit()):
        vector[0] = int(age)
    else:
        message = "Please specify an age."
        
    if(hypertension == "Y"):
        vector[1] = 1
    elif (hypertension == "N"):
        vector[1] = 0
    else:
        message = "Your hypertension input does not follow the specifications. It should either by 'Y' or 'N'."
        
    if(heart == "Y"):
        vector[2] = 1
    elif (heart == "N"):
        vector[2] = 0
    else: 
        message = "Your Heart Disease input does not follow the specification. It should be either 'Y' or 'N'."
        
    if(married == "Y"):
        vector[3] = 1
    elif (married == "N"):
        vector[3] = 0
    else: 
        message = "Your Ever Married input does not follow the specification. It should be either 'Y' or 'N'."
        
    if (work == "Private"):
        vector[4] = 2
    elif (work == "Self-Employed"):
        vector[4] = 3
    elif (work == "Government"):
        vector[4] = 1
    elif (work == "Other"):
        vector[4] = 0
    else:
        message = "Your Work Type input does not follow the specificaiton. It should be 'Private', 'Self-Employed', 'Government', or 'Other'"
        
    if (residence == "Rural"):
        vector[5] = 0
    elif (residence == "Urban"):
        vector[5] = 1
    else:
        message = "Your Residence Type input does not follow the specificaiton. It should be 'Rural' or 'Urban'"   
        
    if(not glucose == "" and glucose.isDigit()):
        vector[6] = int(glucose)
    else:
        message = "Please specify an average glucose level as a number. If you are unsure, just give your best guess."
    
    if(not bmi == "" and bmi.isDigit()):
        vector[7] = int(bmi)
    else:
        message = "Please specify your BMI as a number. If you are unsure, just give your best guess."
    
    if (smoking == "Smokes"):
        vector[8] = 3
    elif (smoking == "Formerly Smoked"):
        vector[8] = 1
    elif (work == "Never Smoked"):
        vector[8] = 2
    else:
        message = "Your smoking input does not follow the specificaiton. It should be 'Smokes', 'Formerly Smoked', or 'Never Smoked'."
    
    return vector, message
        
def load_model():
    #loads the SVM radial basis trained model from s3 bucket and returns it
    s3_client = boto3.client("s3")
    S3_BUCKET = "trained4641models"
    model_name = "svm_radial_basis.p"
    temp_file_path = "/svm_radial_basis.p"
    s3.download_file(S3_BUCKET, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(vector):
    #uses the trained model to predict stroke risk
    #first standardized the form input vector with the rest of the dataset
    u = 
    sd = 
    vector = (vector - u) / sd
    
    model = load_model()
    prediction = model.predict([[vector]])[0]    
    return prediction
