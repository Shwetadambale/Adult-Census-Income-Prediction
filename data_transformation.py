# Adult-Census-Income-Prediction
#Problem Statement: The Goal is to predict whether a person has an income of more than 50K a year or not. This is basically a binary classification problem where a person is classified into the

50K group or <=50K group.
# Importing Datasets
import pandas as pd
# Ensure the file path is correctly specified and enclosed in double quotes
file_path = r"C:\Users\Lenovo\Desktop\Internship\documents\adult - adult.csv"

# Read the CSV file into a DataFrame
dataset = pd.read_csv(file_path, header=None)

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(x)
[[39 'State-gov' 77516 ... 0 40 'United-States']
 [50 'Self-emp-not-inc' 83311 ... 0 13 'United-States']
 [38 'Private' 215646 ... 0 40 'United-States']
 ...
 [58 'Private' 151910 ... 0 40 'United-States']
 [22 'Private' 201490 ... 0 20 'United-States']
 [52 'Self-emp-inc' 287927 ... 0 40 'United-States']]

print(y)
['<=50K' '<=50K' '<=50K' ... '<=50K' '<=50K' '>50K']

# Data Preprocessing
#Dataset Cleaning
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer.fit(x[:,1:])
x[:,1:]=imputer.transform(x[:,1:])

# Label Encoding

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
le3=LabelEncoder()
le5=LabelEncoder()
le6=LabelEncoder()
le7=LabelEncoder()
le8=LabelEncoder()
le9=LabelEncoder()
le13=LabelEncoder()
le=LabelEncoder()
x[:,1]=le1.fit_transform(x[:,1])
x[:,3]=le3.fit_transform(x[:,3])
x[:,5]=le5.fit_transform(x[:,5])
x[:,6]=le6.fit_transform(x[:,6])
x[:,7]=le7.fit_transform(x[:,7])
x[:,8]=le8.fit_transform(x[:,8])
x[:,9]=le9.fit_transform(x[:,9])
x[:,13]=le13.fit_transform(x[:,13])
y=le.fit_transform(y)

print(x)
[[39 7 77516 ... 0 40 39]
 [50 6 83311 ... 0 13 39]
 [38 4 215646 ... 0 40 39]
 ...
 [58 4 151910 ... 0 40 39]
 [22 4 201490 ... 0 20 39]
 [52 5 287927 ... 0 40 39]]

print(y)
[0 0 0 ... 0 0 1]

# Splitting Dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Training Dataset
from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(X_train,Y_train)
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

# Making Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=model.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
print(cm)
accuracy_score(Y_test,y_pred)
[[4574  344]
 [ 536 1059]]
0.8648856133886074

Predicting Test set Result
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))
[[0 0]
 [0 0]
 [0 0]
 ...
 [1 1]
 [0 0]
 [1 1]]
Single Prediction
#age = 35,workclass = 'private',employ_inc = 60000,education_num = 10,marital_status = 'Married-civ-spouse',occupation = 'Prof-specialty',relationship = 'Husband','White','Male',1500, 0,45,'United-States'
result=model.predict(sc.transform([[35,4,60000,10,1,3,4,0,0,0,1500,0,45,39]]))
if result==[0]:
  print('Person makes Below 50K/year')
else:
  print('Person makes Above 50K/year')
  Person makes Below 50K/year
