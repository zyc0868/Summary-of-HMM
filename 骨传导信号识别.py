import scipy.io as sio
from hmmlearn import hmm
import numpy as np
from python_speech_features import mfcc
import joblib
import matplotlib.pyplot as plt

#——————————制作音源文件与标签的对应对应——————————

#data['originalData'][i][0][0]  i从0到19表示数字0，20-39表示数字1依次类推

def data_process():
    data = sio.loadmat('originalData.mat')
    datas={}
    labels={}
    for i in range(196):
        print(i)
        data1 = np.array(data['originalData'][i][0][0])
        datas[i] = data1
        labels[i] = int(i/20)
    return datas , labels


#——————————获取MFCC特征——————————
def get_mfcc_feat(data):
    # if len(data) % 3:
    #     pro_data = data[:-(len(data) % 3)].reshape(-1,3)
    # else:
    #     pro_data = data.reshape(-1,3)
    pro_data = data.reshape(-1)
    mfcc_feat = mfcc(pro_data, samplerate=1200, nfft=256)
    return mfcc_feat

#——————————构造模型——————————
class Model():
    #模型参数，由于是独立词识别，且WAV文件都是同一个人录制，所以n_components = 1，表示每个模型隐藏状态是1
    #n_mix5表示混合的高斯分布由几个高斯分布组成，一般取4或5
    #covariance_type='diag' 表示协方差矩阵的类型
    #n_iter=2000，训练2000次
    def __init__(self,labelset = None, n_components = 3,n_mix=4, covariance_type='diag', n_iter=2000,each_number=20,):
        super(Model,self).__init__()
        self.labelset = labelset
        self.label_num = len(labelset)
        self.each_number = each_number
        self.setrange = self.label_num*self.each_number
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
        #for i in range(self.setrange):
        for i in range(196):
            model = self.models[int(i/self.each_number)]
            print(i,int(i/self.each_number))
            print("labels",labels[i],"labelset[int(i/self.each_number)]",self.labelset[int(i/self.each_number)])
            if labels[i] == self.labelset[int(i/self.each_number)]:
                mfcc_feat = get_mfcc_feat(datas[i])
                model.fit(mfcc_feat)
                print("scess")
# ——————————测试模型——————————
#测试每个数据在所有模型中的得分情况，最后将得分最高的模型对应的标签值作为预测值
    def test(self, datas=None, labels = None):
        real = []
        predict = []
        #for j in range(len(datas)):
        for j in range(196):
            scores=[]
            for i in range(self.label_num):
                model = self.models[i]
                mfcc_feat = get_mfcc_feat(datas[j])
                score = model.score(mfcc_feat)
                scores.append(score)
            index = scores.index(max(scores))
            predict.append(self.labelset[index])
            real.append(labels[j])

        accuracy = 0
        print("predict:", predict)
        print("real", real)
        for i in range(len(real)):
            if real[i] == predict[i]:
                accuracy += 1
        print("识别率： ", "percent: {:.2%}".format(accuracy / len(real)))

    def save(self, path="models.pkl"):
        joblib.dump(self.models, path)


    def load(self, path="models.pkl"):
        self.models = joblib.load(path)



if __name__ == "__main__":
    labelset=[0,1,2,3,4,5,6,7,8,9]
    datas , labels = data_process()
    testdatas ,testlabels = data_process()

    m = Model(labelset)
    m.get_models()
    m.train(datas,labels)
    m.save()
    m.load()
    print("训练数据的识别：")
    m.test(datas,labels)
    print("测试数据的识别：")
    m.test(testdatas,testlabels)








