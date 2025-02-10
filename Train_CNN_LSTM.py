
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split

import math
from tensorflow.keras.layers import Input,GlobalMaxPool1D,Dropout,BatchNormalization
from tensorflow.keras import Model
import numpy as np
from keras.layers import Dense,Conv1D,MaxPooling1D,LSTM,Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten,Reshape

from keras import callbacks
from keras.layers import LSTM,Bidirectional
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


# =============Feature Selection=============
# 100 features with highest chi-squared statistics are selected
chi2_features = SelectKBest(chi2, k = 120)
xy=chi2_features.fit(X_train_res, y_train_res)
# dbfile=open("ch_featlstm1","wb")
# pickle.dump(xy,dbfile)
# dbfile.close()
Train_data = xy.fit_transform(X_train_res, y_train_res)

print("Train_data-->",Train_data.shape)
scaler = MinMaxScaler()
Xscale=scaler.fit_transform(Train_data)
# pickle.dump(scaler,open('scalerlstm1','wb'))
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(Xscale, y_train_res, test_size=0.2, random_state=0)


print(X_train.shape)
print(X_train.shape)


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


# CNN-LSTM model
batch_size = 32
model = Sequential()
model.add(Convolution1D(64, kernel_size=122, padding="same",activation="relu",input_shape=(120, 1)))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=True))) 
# model.add(Reshape((120, 1), input_shape = (120, )))

model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(120, return_sequences=True))) 

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="cpoint/checkpoint1-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_accuracy',mode='max')

model.fit(X_train, Y_train,batch_size=16,epochs=500,validation_data=(X_test, Y_test),callbacks=[checkpointer])

model.save("cnn_model.hdf5")