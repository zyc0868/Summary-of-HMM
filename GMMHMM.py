from hmmlearn import hmm
import numpy as np
import os
from scipy.io import wavfile
from python_speech_features import mfcc
import joblib
#——————————制作音源文件与标签的对应对应——————————

def data_process(file):
    datas={}
    labels={}
    for root, dirs, files in os.walk(file):
        for file in files:
            dataname=file.strip('.wav')
            datas[dataname] = os.sep.join((root,file))
            labels[dataname] = dataname.split('-')[0]
    return datas , labels




def get_mfcc_feat(file):
    fs , audio = wavfile.read(file)
    mfcc_feat=mfcc(audio,samplerate=20000,numcep=26,nfft=600)
    return mfcc_feat


class Model():
    def __init__(self,labelset = None, n_components = 1,n_mix=4, covariance_type='diag', n_iter=2000):
        super(Model,self).__init__()
        self.labelset = labelset
        self.label_num = len(labelset)
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models=[]

    def get_models(self):
        for i in range(self.label_num):
            model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix,covariance_type=self.covariance_type, n_iter=self.n_iter)
            self.models.append(model)


    def train(self,datas = None, labels = None):
        for i in range(self.label_num):
            model = self.models[i]
            for dataname in datas:
                if labels[dataname] == self.labelset[i]:
                    mfcc_feat = get_mfcc_feat(datas[dataname])
                    model.fit(mfcc_feat)

    def test(self, datas=None, labels = None):
        real = []
        predict = []
        for dataname in datas:
            scores=[]
            for i in range(self.label_num):
                model = self.models[i]
                mfcc_feat = get_mfcc_feat(datas[dataname])
                score = model.score(mfcc_feat)
                scores.append(score)
            index = scores.index(max(scores))
            predict.append(self.labelset[index])
            real.append(labels[dataname])

        accuracy = 0
        print("predict:",predict)
        print("real",real)
        for i in range(len(real)):
            if real[i] == predict [i]:
                accuracy += 1
        print("识别率： ","percent: {:.2%}".format(accuracy/len(real)))

    def save(self, path="models.pkl"):
        joblib.dump(self.models, path)

    def load(self, path="models.pkl"):
        self.models = joblib.load(path)


if __name__ == "__main__":
    labelset=[ '1' , '2' , '3' , '4' , '5']
    datas , labels = data_process('train1_data')
    testdatas ,testlabels = data_process('test1_data')

    m = Model(labelset)
    m.get_models()
    m.train(datas,labels)
    m.save()
    m.load()
    print("训练数据的识别：")
    m.test(datas,labels)
    print("测试数据的识别：")
    m.test(testdatas,testlabels)








