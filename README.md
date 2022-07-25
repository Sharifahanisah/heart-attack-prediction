# Heart Attack App Prediction
 
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 	![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
  
## The project's aim ##
 To develop an app to predict the chance of a person having heart attack using machine learning
 
## Credits To ##
Heart Attack Analysis & Prediction Dataset :https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

## Preview of App ##
![app](https://user-images.githubusercontent.com/109563861/180787617-faf7fae2-6be1-437d-a4ae-d63f958a4709.PNG)

In this model training 'age','cp','thall','thalachh','oldpeak' has higher correlation with output 

###
From the Categorical Variables:
>>    Variables that are more or equal to 50 % are chosen

From the Continuous Variables:
>>    variables that are more or equal to 60 % are chosen

From Model Devalopment (plpeline) :
>>   its found [('Min_Max_Scaler', MinMaxScaler()), ('logistic_Classifier', LogisticRegression())] has the best scaler and classifier this data

From Hyperparameter tuning :
>>   model.score is 0.7692307692307693
>>   model best paremeter is {'logistic_Classifier__C': 100, 'logistic_Classifier__penalty': 'l2', 'logistic_Classifier__solver': 'newton-cg'}

## Accuracy  ##
![acc](https://user-images.githubusercontent.com/109563861/180787928-44fddd3e-4e81-410a-b9e1-5f14cb30e99a.PNG)

This app to predict the chance of a person having heart attack is 77% 

