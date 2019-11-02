import numpy as np
from hmmlearn import hmm

#——————————初始化隐藏状态，观测序列，开始转移概率，转移概率，发射概率——————————
hiddenstates = ("box1", "box2", "box3")
observations = ("red", "green", "blue")

start_prob = np.array([0.3,0.4,0.3])

transmat_prob = np.array([[0.2,0.4,0.4],
                          [0.5,0.2,0.3],
                          [0.1,0.4,0.5]])

emission_prob = np.array([[0.1,0.3,0.6],
                          [0.3,0.5,0.2],
                          [0.4,0.4,0.2]])


#——————————将参数放入模型中——————————
n_hiddenstates = len(hiddenstates)
model = hmm.MultinomialHMM(n_components=n_hiddenstates,n_iter=2000)
model.startprob_ = start_prob
model.transmat_ = transmat_prob
model.emissionprob_ = emission_prob


#——————————1.求解最可能的观测序列————————————
seen = np.array([[1],[1],[0]])
#decode函数得到两个返回值，一个是产生观测序列的对数概率，一个是预测的观测序列
logprob, boxs = model.decode(seen)    #默认的算法是维比特算法
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.flatten())))
print("The hidden box", ", ".join(map(lambda x: hiddenstates[x], boxs)))

#predict函数返回的直接是预测的观测序列
boxs_ = model.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.flatten())))
print("The hidden box", ", ".join(map(lambda x: hiddenstates[x], boxs_)))

#decode,predict都属于解码问题，即一直hmm模型所需的参数和观测序列，求解最大可能的状态序列
#用到了基于动态规划的维比特算法


#——————————2.求解得到当前观测序列的概率————————————
print (model.score(seen))
#score属于计算得到当前观测序列的概率，用到了向前向后算法


#——————————3.构建一个新模型，求解模型参数——————————
model2 = hmm.MultinomialHMM(n_components=n_hiddenstates, n_iter=2000)

observations2 = np.array([[2,1,2,1,2,2,1,2,0,2],
                          [0,2,0,1,0,1,1,1,2,1],
                          [0,1,0,2,0,1,0,1,2,1]])


#将得到的观测矩阵输入到模型中
model2.fit(observations2)
boxs2 = model2.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.flatten())))
print("The hidden box", ", ".join(map(lambda x: hiddenstates[x], boxs2)))
#参数学习的问题用到了类似于EM算法的鲍姆-维奇算法