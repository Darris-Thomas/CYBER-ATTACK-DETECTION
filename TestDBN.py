from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from dbn.tensorflow import SupervisedDBNClassification
import pickle
from sklearn.preprocessing import MinMaxScaler
scaler=pickle.load(open('scaler','rb'))
featsel=pickle.load(open('ch_feat1','rb'))
classifier=SupervisedDBNClassification.load('model.pkl')

def test(val):
    val_list=val.split(',')
    flist=[abs(float(x)) for x in val_list]
    selfeat=featsel.transform([flist])
    print(selfeat.shape)
    scdata=scaler.transform(selfeat)
    
    ypred=classifier.predict(scdata)
    print(ypred)
    if(ypred[0]==1):
        return "No attack detected"
    else:
        return "Attack detected"







# tdata='''147.1183731,131559.4477,27.12382202,130932.6159,-92.8478107,131634.6675,144.2879615,408.88463,24.10433444,410.34951,-95.66103348,409.06774,147.1355618,131383.9348,0.0,0.0,0.0,0.0,144.2421249,409.43396,0.0,0.0,0.0,0.0,60.0,0.0,9.192367708,0.049133298,0,140.8559444,129653.8792,20.89577079,129628.8059,-99.09305067,129678.9524,-44.37558123,414.56104,-164.496183,414.74415,75.7221022,414.37793,140.8845922,129653.8792,0.0,0.0,0.0,0.0,-44.38131081,414.56104,0.0,0.0,0.0,0.0,60.0,0.0,8.914732633,-3.050753665,0,140.8387556,129603.7326,20.85566374,129578.6594,-99.13888729,129678.9524,-44.55319815,412.36372,-164.5649379,412.72994,75.64188811,410.53262,140.8502148,129628.8059,0.0,0.0,0.0,0.0,-44.49017279,411.81439,0.0,0.0,0.0,0.0,60.0,0.0,8.957966804,-3.042406133,0,147.010806,131975.0,27.06756669,131816.625,-92.97454989,132054.625,144.302677,407.3352661,24.17266956,408.0562744,-95.71564141,407.6385498,147.0382771,131948.9531,0.0,0.0,0.0,0.0,144.2532401,407.6843262,0.0,0.0,0.0,0.0,60.0,0.0,9.229816532,0.046042843,0.0,0,0,0,0,0,0,0,0,0,0,0,0'''

# test(tdata)