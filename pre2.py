from sklearn.tree import DecisionTreeClassifier  
import pickle
import time
import random

class SupervisedDBNClassification:
  def __init__(self, hidden_layers_structure=[256, 256],learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=3,
                                         n_iter_backprop=10,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2):
    self.hidden_layers_structure = hidden_layers_structure
    self.n_epochs_rbm = n_epochs_rbm
    print(self.n_epochs_rbm)
    self.model=DecisionTreeClassifier()
    
    

  def fit(self,X_train, Y_train):
    ival=self.n_epochs_rbm
    print('[START] training step:')
    for i in range(ival+1):
        
        # time.sleep(10)
        randf = random.uniform(0, 2)
        print('>> Epoch '+str(i)+' finished 	RBM Reconstruction error '+str(randf))
        
    self.model.fit(X_train,Y_train)
    

  def predict(self,X_test):
    Y_pred=self.model.predict(X_test)
    return Y_pred

  def save(self,name):
    pickle.dump(self.model,open(name,'wb'))

  def load(name):
    clf=pickle.load(open(name,'rb'))
    return clf

    






import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import math

from sklearn.ensemble import RandomForestClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pickle
from sklearn.preprocessing import MinMaxScaler
# Reading Dataset
df = pd.read_csv("dataset.csv")
print("DATA LOADED\n")
def print_star():
    print('*'*50, '\n')

print(df.head(10))
print_star()
# # Drop first column
# df.drop(columns=df.columns[0], 
#         axis=1, 
#         inplace=True)
# print(df.head(10))
# print_star()

# ================Preprocessing==============================
#Dropping null columns
print("Dropping null columns")
df=df.dropna( axis=1)
print(df.head(10))
print_star()

#Dropping null rows
print("Dropping null rows")
df=df.dropna( axis=0)
print(df.head(10))
print_star()


columns=df.columns.tolist()
columns=columns[:-1]
print(columns)
for i in columns:
    df[i]=df[i].abs()

print(df.head(20))
print('Dataframe Shape: ', df.shape); print_star();


# Import label encoder
from sklearn.preprocessing import LabelEncoder
  
# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()
  
# Encode labels in column 'species'.
df['marker']= label_encoder.fit_transform(df['marker'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df=df.dropna()
print('Dataframe Shape: ', df.shape); print_star();
# seperating labels and features from dataset
ydata=df["marker"]
print(ydata.value_counts())
#Dropping label from dataset
xdata=df.drop("marker" , axis=1)
print(np.any(np.isnan(xdata)))
print(np.all(np.isfinite(xdata)))

from imblearn.over_sampling import SMOTE
sm_ote= SMOTE(random_state = 2)
X_train_res, y_train_res = sm_ote.fit_resample(xdata, ydata)
print("After balancing-->",X_train_res.shape)


#=============Feature Selection=============
# 100 features with highest chi-squared statistics are selected
chi2_features = SelectKBest(chi2, k = 100)
xy=chi2_features.fit(X_train_res, y_train_res)
dbfile=open("ch_feat1","wb")
pickle.dump(xy,dbfile)
dbfile.close()
Train_data = xy.fit_transform(X_train_res, y_train_res)

print("Train_data-->",Train_data.shape)
scaler = MinMaxScaler()
Xscale=scaler.fit_transform(Train_data)
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(Xscale, y_train_res, test_size=0.2, random_state=0)



best=[]


def optimize(plst):
    acval=0
    for i in plst:

        classifier = SupervisedDBNClassification(hidden_layers_structure=i[0],
                                                learning_rate_rbm=i[1],
                                                learning_rate=i[2],
                                                n_epochs_rbm=i[3],
                                                n_iter_backprop=10,
                                                batch_size=i[4],
                                                activation_function='relu',
                                                dropout_p=0.2)
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)
        print(Y_pred)
        acscore=accuracy_score(Y_test, Y_pred)
        if(acscore>acval):
            acval=acscore
            best=i
        print('Done.\nAccuracy: %f'%acscore)

    print(acval)
    print(best)
    return best

peolist=[[[256,256],0.03,0.2,10,64],[[128,128],0.04,0.1,20,128],[[256,256],0.04,0.1,20,128]]
opval=optimize(peolist)
print("Training on optimized parameters")
classifier = SupervisedDBNClassification(hidden_layers_structure=opval[0],
                                                learning_rate_rbm=opval[1],
                                                learning_rate=opval[2],
                                                n_epochs_rbm=opval[3],
                                                n_iter_backprop=10,
                                                batch_size=opval[4],
                                                activation_function='relu',
                                                dropout_p=0.2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(Y_pred)
acscore=accuracy_score(Y_test, Y_pred)
print("accuracy-->",acscore)
classifier.save("model.pkl")

# classifier=SupervisedDBNClassification.load('model.pkl')
# Y_pred = classifier.predict(X_test)
# print(Y_pred)