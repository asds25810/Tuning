# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from label_propagation import CAMLP
from scipy import sparse
 
data_path = 'bicg_data.csv'
np_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)        
# total size of data
size = 6000
# initial training size
init_size = 20
# maximum training size
max_size = 200
# number of new samples in each iteration
batch_size = 4
# number of neighbors
n_neighbor = 50

np_data = np.random.permutation(np_data)
X = np_data[:, 0:-1]
X = preprocessing.StandardScaler().fit_transform(X)
Y = np_data[:, -1]
best = np.amin(Y)
threshold = np.percentile(Y[0:init_size],5)
Y_cate = (Y<=threshold).astype(int)


def construct_graph(X):
    distance = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            distance[i][j] = np.sum(abs(X[i]-X[j]))
    
    G = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        idx = np.argpartition(distance[i],n_neighbor+1)
        G[i][idx[0:n_neighbor+1]]=1
        G[i][i]=0
    
    return G

#G = sparse.csr_matrix(construct_graph(X))
#sparse.save_npz("Graph.npz",G)
G = sparse.load_npz("Graph.npz").tolil()
# Modulation matrix , [[1,0],[0,1]] for homophily, [[0,1],[1,0]] for heterophily
Modulation = np.array([[0,1],[1,0]])
clf = CAMLP(graph=G,H=Modulation)
output = np.zeros((1, len(range(init_size, max_size + 1, batch_size))))
k=0
n_exp = 20
for k in range(n_exp):
    train_size = init_size
    n_batch = 0
    X_rest = np.arange(init_size,X.shape[0])
    Y_rest = Y_cate[init_size:]
    X_train = np.arange(init_size)
    Y_train = Y_cate[0:init_size]
    while train_size <= max_size:
        clf.fit(X_train,Y_train)
        print('Exp %d, %d samples picked, %d optimal point' % (k, train_size,sum(Y_train)))
        output[0][n_batch] = output[0][n_batch] + np.amin(Y[X_train])
        pred = clf.predict(X_rest)
        index = np.random.permutation(np.nonzero(pred)).reshape(-1)
        optim_candidates =X_rest[index].reshape(-1)
        train_size += batch_size
        n_batch += 1
        X_train = np.concatenate((X_train,optim_candidates[0:batch_size]))
        Y_train = np.concatenate((Y_train,Y_cate[optim_candidates[0:batch_size]]))
        X_rest = np.delete(X_rest,index[0:batch_size])

    
output = output/n_exp
print(output)
np.savetxt("output_GEIST.txt",output)
