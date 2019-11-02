from hmmlearn import hmm
import numpy as np
import os
from scipy.io import wavfile
from python_speech_features import mfcc
import joblib
#——————————制作音源文件与标签的对应对应——————————
#对于同一数据，datas 与 labels 对应的键是相同的，datas的值是对应文件的路径，labels的值是该数据的标签
def data_process(file):
    datas={}
    labels={}
    for root, dirs, files in os.walk(file):
        for file in files:
            dataname=file.strip('.wav')
            datas[dataname] = os.sep.join((root,file))
            #文件格式是x-n，x表示该数据的内容，n表示该数据的第几个训练样本
            #如，2-5表示，第五遍说2的音频文件
            labels[dataname] = dataname.split('-')[0]
    return datas , labels



#——————————获取MFCC特征——————————
def get_mfcc_feat(file):
    fs , audio = wavfile.read(file)
    #fs 为 44100Hz
    mfcc_feat=mfcc(audio,samplerate=(fs/2),nfft=1024)
    return mfcc_feat

#——————————构造模型——————————
class Model():
    #模型参数，由于是独立词识别，且WAV文件都是同一个人录制，所以n_components = 1，表示每个模型隐藏状态是1
    #n_mix5表示混合的高斯分布由几个高斯分布组成，一般取4或5
    #covariance_type='diag' 表示协方差矩阵的类型
    #n_iter=2000，训练2000次
    def __init__(self,labelset = None, n_components = 1,n_mix=4, covariance_type='diag', n_iter=2000):
        super(Model,self).__init__()
        self.labelset = labelset
        self.label_num = len(labelset)
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        #创建模型列表，数量对应独立词的数量
        self.models=[]

#——————————生成模型——————————
    def get_models(self):
        for i in range(self.label_num):
            model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix,covariance_type=self.covariance_type, n_iter=self.n_iter)
            self.models.append(model)

#——————————训练模型——————————
#遍历整个数据集，若数据集的对应的标签的值与当前模型匹配，则将该数据的MFCC特征放入该模型中
    def train(self,datas = None, labels = None):
        for dataname in datas:
            for i in range(self.label_num):
                model = self.models[i]
                if labels[dataname] == self.labelset[i]:
                    mfcc_feat = get_mfcc_feat(datas[dataname])
                    model.fit(mfcc_feat)

# ——————————测试模型——————————
#测试每个数据在所有模型中的得分情况，最后将得分最高的模型对应的标签值作为预测值
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








