import numpy as np
import pandas as pd
from tqdm import *

# Predefined parameter
rightNumber = 0
questionNumber = 0
skillNumber = 0

data0 = pd.read_csv("data1.csv", header=None)
# print(data0.values)
data1 = list(data0.values)
for i in range(len(data1)):
    if (i+1)%4 == 2:
        if skillNumber < max(data1[i]):
            skillNumber = (int)(max(data1[i]))
    elif (i+1)%4 == 3:
        if questionNumber < max(data1[i]):
            questionNumber = (int)(max(data1[i]))
rightNumber = questionNumber + 2
questionNumber = questionNumber-skillNumber
skillNumber = skillNumber + 1






# print(data1)
lam = 0.01
pros = []
tt = tf = ft = ff = np.ones([questionNumber,questionNumber])
for i in tqdm(range(0, len(data1), 4)):
    lis = [data1[i + 3][j] for j in range(len(data1[i + 3])) if data1[i + 3][j] != np.NaN]
    # print(lis)
    l = (int)(data1[i][0])
    for x in range(l):
        for y in range(l):
            X = (int)(data1[i+2][x]-skillNumber)
            Y = (int)(data1[i+2][y]-skillNumber)
            if lis[x] == rightNumber and lis[y] == rightNumber:
                tt[X][Y]+=1
            elif lis[x] == rightNumber and lis[y] == rightNumber-1:
                tf[X][Y]+=1
            elif lis[x] == rightNumber-1 and lis[y] == rightNumber:
                ft[X][Y]+=1
            elif lis[x] == rightNumber-1 and lis[y] == rightNumber-1:
                ff[X][Y]+=1

F = np.zeros([questionNumber,questionNumber])
sim = np.zeros([questionNumber,questionNumber])
for i in tqdm(range(questionNumber)):
    for j in range(questionNumber):
        P = (tt[i][j] + lam)/(tt[i][j] + ft[i][j] + lam)
        R = (tt[i][j] + lam)/(tt[i][j] + tf[i][j] + lam)
        F[i][j] = 2*P*R/(P+R)
for i in tqdm(range(questionNumber)):
    for j in range(questionNumber):
        sim[i][j] = (F[i][j] + F[j][i])/2
pd.DataFrame(sim).to_csv('relation.csv')
# print(ans)
# exit(0)
