import joblib
import os
def data_process(file):
    datas={}
    labels={}
    for root, dirs, files in os.walk(file):
        for file in files:
            dataname=file.strip('.wav')
            datas[dataname] = os.sep.join((root,file))
            labels[dataname] = dataname.split('-')[0]
    return datas , labels

m=joblib.load('models.pkl')
testdatas ,testlabels = data_process('test1_data')
m.test(testdatas,testlabels)