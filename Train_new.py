
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split

import math
from tensorflow.keras.layers import Input,GlobalMaxPool1D,Dropout,BatchNormalization,MaxPool1D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Dense,Conv1D,MaxPooling1D,LSTM,Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten

from keras import callbacks
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
# Import train_test_split function
from sklearn.model_selection import train_test_split
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


print("Train_data-->",X_train_res.shape)
scaler = MinMaxScaler()
Xscale=scaler.fit_transform(X_train_res)
pickle.dump(scaler,open('scalerlstm','wb'))
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(Xscale, y_train_res, test_size=0.2, random_state=0)


print(X_train.shape)
print(X_train.shape)

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_output_size = 70

model = Sequential()
model.add(Conv1D(32, 9, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
model.add(MaxPool1D(pool_size=(2)))
model.add(LSTM(units=16,return_sequences=False,dropout=0.2))
model.add(Dense(units=1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=250)