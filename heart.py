# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:28:43 2022

@author: HP
"""
#%%
import matplotlib.pyplot as plt
import scipy.stats as ss
import missingno as msno
import seaborn as sns
import pandas as pd
import numpy as np 
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

#%%
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%%

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(),'best_estimator.pkl')

#%% STEP 1  DATA LOADING
os.getcwd()
df = pd.read_csv(CSV_PATH)
#%% STEP 2 DATA VIZUALIZATION/IMPETTION

df.describe().T
df.boxplot()
df.info()
df.isna().sum()

cat= ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]
con= ["age", "trtbps","chol","thalachh","oldpeak"]

for i in con: 
    plt.figure()
    sns.distplot(df[i])
    plt.show()
    
for i in cat:
    plt.figure()
    sns.countplot(df[i])
    plt.show() 
    
#thall 0 = null and caa 4 = null

#%% Step 3) Data Cleaning

# convert thall 0 to null and caa 4 to null

df['thall']=df['thall'].replace(0, np.nan)
df['caa']=df['caa'].replace(4, np.nan)

df.describe().T
df.info()
df.isna().sum()
df.duplicated().sum()

msno.matrix(df) 
msno.bar(df)
df.isna().sum()

# clear nan with fillna and convert dtype as int

df_demo = df.copy()

df_demo['thall']= df_demo['thall'].fillna(df_demo['thall'].mode()[0])
df_demo['caa']= df_demo['caa'].fillna(df_demo['caa'].mode()[0])  

df_demo.isna().sum()
df_demo.info()


#  clear duplicates
df_demo.duplicated().sum()
df_demo.drop_duplicates(inplace = True)

for i in con: 
    plt.figure()
    sns.distplot(df_demo[i])
    plt.show()
    
for i in cat:
    plt.figure()
    sns.countplot(df_demo[i])
    plt.show() 

#%% step 4  Feature selection
# cat and cat
# cramer's V

for i in cat:
    print(i)
    confusion_matrix = pd.crosstab(df_demo[i],df_demo['output']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))
    
# cp and thall has the highest correlation with output 

# cat and  con

for i in con:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_demo[i], axis =-1),df_demo['output']) 
    print(lr.score(np.expand_dims(df_demo[i], axis =-1),df_demo['output']))

# thalachh, oldpeak and age the highest correlation with output 

#%%  Step 5)-Pre-processing
X = df_demo.loc[:,['age','cp','thall','thalachh','oldpeak']]
y = df_demo['output']

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=123)

#%% model devalopment (plpeline)

#logistic regression

pipeline_mms_lr = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('logistic_Classifier',LogisticRegression())  
                                ])



pipeline_ss_lr = Pipeline([
                                ('standered_Scaler', StandardScaler()),
                                ('logistic_Classifier',LogisticRegression())  
                                ])



#decision tree

pipeline_mms_dt = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('Tree_Classifier',DecisionTreeClassifier())  
                                ])



pipeline_ss_dt = Pipeline([
                                ('standered_Scaler', StandardScaler()),
                                ('Tree_Classifier',DecisionTreeClassifier())  
                                ])

# Random Forest 

pipeline_mms_rf = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('Forest_Classifier',RandomForestClassifier())  
                                ])



pipeline_ss_rf = Pipeline([
                                ('standered_Scaler', StandardScaler()),
                                ('Forest_Classifier',RandomForestClassifier())  
                                ])   

#Gradient BOOst

pipeline_mms_gb = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('GBoost_Classifier',GradientBoostingClassifier())  
                                ])



pipeline_ss_gb = Pipeline([
                                ('standered_Scaler', StandardScaler()),
                                ('GBoost_Classifier',GradientBoostingClassifier())  
                                ])   

# SVC

pipeline_mms_svc = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('SVC_Classifier',SVC())  
                                ])



pipeline_ss_svc = Pipeline([
                                ('standered_Scaler', StandardScaler()),
                                ('SVC_Classifier',SVC())  
                                ])  


#%% create a list to store all the pipelines
pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_dt, pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_gb,pipeline_ss_gb,
             pipeline_mms_svc,pipeline_ss_svc] 

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
    
best_accuracy = 0

for i, pipe in enumerate(pipelines):
    print(pipe.score(X_test,y_test)) 
    if pipe.score(X_test,y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test,y_test)
        best_pipeline = pipe                             
                                 
print('The best scaler and classifier for CVD Data is {} with accuracy of {}'.
      format(best_pipeline.steps,best_accuracy))  


#%% Hyperparameter tuning
#from the above pipeline, ss+rf found to be the optimal combination for the data set 'IF'


pipeline_mms_lr = Pipeline([
                                ('Min_Max_Scaler', MinMaxScaler()),
                                ('logistic_Classifier',LogisticRegression())  
                                ])


pipeline_mms_lr.get_params()

grid_pram = [{'logistic_Classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'logistic_Classifier__penalty': ['l2'],
              'logistic_Classifier__C':[100, 10, 1.0, 0.1, 0.01]
              }]


grid_search = GridSearchCV(pipeline_mms_lr,grid_pram, cv = 5 ,verbose=1, n_jobs=-1)

model = grid_search.fit(X_train,y_train)
model.score(X_test,y_test)

print(model.best_index_)
print(model.best_params_) 

#%%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#%%model saving

with open(BEST_ESTIMATOR_SAVE_PATH,'wb') as file:
    pickle.dump(model.best_estimator_,file)  

                    
                      
