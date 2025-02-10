import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from dbn.tensorflow import SupervisedDBNClassification
import math
from sklearn.metrics import mean_absolute_error
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

# Training
# classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
#                                          learning_rate_rbm=0.05,
#                                          learning_rate=0.1,
#                                          n_epochs_rbm=3,
#                                          n_iter_backprop=10,
#                                          batch_size=32,
#                                          activation_function='relu',
#                                          dropout_p=0.2)
# # classifier.fit(X_train, Y_train)

# # Test
# Y_pred = classifier.predict(X_test)
# print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

# import pickle
# pickle.dump(classifier,open('clf.pkl','wb'))


from sklearn.tree import DecisionTreeClassifier  

from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

svmclf=LinearSVC()  
svmclf.fit(X_train, Y_train)

y_pred=svmclf.predict(X_test)



acc=accuracy_score(Y_test, y_pred)

print("Accuracy LinearSVC==>",acc)

# X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
# print(X_train.shape)
# X_val = np.reshape(X_test, (len(X_test), len(X_test[0]), 1))
# from  tensorflow.keras.utils import to_categorical
# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)

# from keras.models import Sequential
# from keras.layers import Dense,GRU
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers import LSTM

# # model = Sequential()
# # model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2])))
# # model.add(Dropout(0.5))
# # model.add(Dense(100, activation='relu'))
# # model.add(Dense(2, activation='softmax'))
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # # fit network
# # model.fit(X_train, Y_train, epochs=10, batch_size=26, verbose=1)


# X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
# print(X_train.shape)
# X_val = np.reshape(X_test, (len(X_test), len(X_test[0]), 1))

# print(X_val[0].shape)

# print(X_train.shape)
# num_labels = 2
# print(num_labels)
# model = Sequential()

# model.add(GRU(256, activation='relu', recurrent_activation='hard_sigmoid'))

# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))

# #model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Dense(num_labels, activation='softmax'))
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_val, Y_test))