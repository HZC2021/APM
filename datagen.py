import numpy as np
import math
import seaborn as sns
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

def distance(coord1,coord2):
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    d = math.sqrt(dx**2+dy**2)
    return d

def CalAdjMat(pos):
    A = np.zeros(shape = (len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(i, len(pos)):
            dist = math.sqrt(math.pow(pos[i, 0] - pos[j, 0], 2) + math.pow(pos[i, 1] - pos[j, 1], 2))
            if dist != 0:
                A[i, j] = 1.0 / dist
                A[j, i] = 1.0 / dist
    return A

def norm_adjacency_matrix(A, n):
    I = np.eye(A.shape[0])
    A_hat = A + n * I
    D_inv = np.diag(np.float_power(np.sum(A_hat, axis = 0), -0.5))
    A_hat = D_inv @ A_hat @ D_inv
    return A_hat


col = ["id","x","y","z","mc","tof","vdc","vp","detx","dety","pulse","ions"]
all = np.load("dataset.npy")

Total = int(1e6)
dataset = all[len(all)-Total:]
data = dataset[:Total,:] ##down sampling 1M points

def countPoints(coord):
    cnt = 0
    for i in range(len(data)):
        if distance(data[i,8:10],coord) < 0.5:
            cnt += 1
    return cnt

sample = 100
sample_data = []
density_label = []
AdjMat = []
AX = []
for i in range(Total//sample):

    sample_data.append(data[i*sample:(i+1)*sample])
    # density = countPoints(data[i*sample,8:10])
    # density_label.append(density)
    A = norm_adjacency_matrix(CalAdjMat(data[i*sample:(i+1)*sample,8:10]), 1)
    AdjMat.append(A)
    ax = np.dot(A,sample_data[i])
    AX.append(ax)
for j in range(Total):
    # print(j)
    density = countPoints(data[j,8:10])
    density_label.append(density)

np.save("sample_data_100_1M.npy",np.array(sample_data))
np.save("density_label_100_1M.npy",np.array(density_label))
np.save("AdjMat_100_1M.npy",np.array(AdjMat))
np.save("AX_100_1M.npy",np.array(AX))



