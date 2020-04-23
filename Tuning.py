import pyltr
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import random
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from label_propagation import CAMLP
from scipy.stats import norm
from scipy import sparse
from scipy import spatial
import sys
from skgarden import forest

marker_list = ['o', 'x', 'D', '^', '+', 's']
label_list = [
    'GBDT_Rank',
    'GBDT_Regress',
    'Random',
    'GP_opt',
    'GEIST',
    'SMAC',
]
# read data
data_path = 'data.csv'
if len(sys.argv) > 1:
    data_path = sys.argv[1]
print(data_path)
np_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)


class Pool:
    def __init__(self, data_X, data_Y):
        self.X = np.copy(data_X)
        self.Y = np.copy(data_Y)

    def __len__(self):
        return self.X.shape[0]

    def pop(self, index):
        tmp_X = self.X[index]
        tmp_Y = self.Y[index]
        self.X = np.delete(self.X, index, 0)
        self.Y = np.delete(self.Y, index)
        return tmp_X, tmp_Y

    def pop_random(self):
        index = random.randint(0, len(self.X) - 1)
        return self.pop(index)


metric = pyltr.metrics.NDCG(k=30)  


class Selector:
    def __init__(self):
        model = 0
        self.name = 'None'

    def pick(self, pool, batch_size):
        return X, Y

    def update(self, X_train, Y_train):
        return 0

    def update(self, X_train, Y_train, G):
        return 0


class Selector_Rank(Selector):
    def __init__(self):
        super(Selector, self).__init__()
        self.model = pyltr.models.LambdaMART(
            metric=metric,
            n_estimators=50,
            learning_rate=0.2,
            # subsample=1.0,
            # max_depth=20,
            verbose=0,
            warm_start=False
        )
        self.name = 'Rank'

    def pick(self, pool, batch_size):
        Y_pred = self.model.predict(pool.X)
        pick_index = np.argpartition(- Y_pred, batch_size)
        return pool.pop(pick_index[0:batch_size])

    def update(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, qids=np.repeat(1, X_train.shape[0]))
        return 0


class Selector_Rand(Selector):
    def __init__(self):
        super(Selector, self).__init__()
        self.model = 0
        self.name = 'Rand'

    def pick(self, pool, batch_size):
        pick_index = np.random.permutation(len(pool))
        return pool.pop(pick_index[0:batch_size])

    def update(self, X_train, Y_train):
        return 0


def get_uncertainty(model, X):
    predicts = np.zeros(shape=(model.n_estimators, X.shape[0]))
    for i, tree in enumerate(model.estimators_):
        predicts[i] = tree[0].predict(X) * 0.2
    predicts = np.transpose(predicts)
    uncertainty = np.zeros(shape=(X.shape[0]))
    for i in range(predicts.shape[0]):
        uncertainty[i] = np.log10(abs(np.std(predicts[i]) / np.mean(predicts[i])) + 1.0)
        # uncertainty[i] = np.std(predicts[i])
    return uncertainty


class Selector_GP_opt(Selector):
    def __init__(self):
        super(Selector, self).__init__()
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        # self.model = GaussianProcessRegressor(alpha=0.2 ** 2)
        self.model = GaussianProcessRegressor(kernel, alpha=0.2 ** 2)
        self.name = "GP_opt"

    def expected_improvement(self, X, X_train, xi=0.01):
        mu, sigma = self.model.predict(X, return_std=True)
        mu_train = self.model.predict(X_train)

        mu_train_opt = np.min(mu_train)

        with np.errstate(divide='warn'):
            imp = mu - mu_train_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def pick(self, pool, batch_size, X_train=None):
        EI = self.expected_improvement(pool.X, X_train)
        pick_index = np.argpartition(EI, batch_size)
        return pool.pop(pick_index[0:batch_size])

    def update(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)


class Selector_GEIST(Selector):
    def __init__(self):
        super(Selector, self).__init__()
        self.name = "GEIST"
        self.model = CAMLP(graph=None, beta=0.1, max_iter=30)

    def pick(self, X_rest, batch_size):
        probas = self.model.predict_proba(X_rest)
        # pred = self.model.predict(X_rest)
        # index = np.random.permutation(np.nonzero(pred)[0]).reshape(-1)
        # optim_candidates =X_rest[index].reshape(-1)

        index = np.argpartition(-probas[:, 1], batch_size)

        return index[0:batch_size]

    def update(self, X_train, Y_train, G):
        self.model.graph = G
        self.model.fit(X_train, Y_train)


class Selector_SMAC(Selector):
    def __init__(self):
        super(Selector,self).__init__()
        self.model = forest.ExtraTreesRegressor(min_variance=0.0001)
        self.name = "SMAC"

    def expected_improvement(self,X,X_train,xi=0.01):
        mu, sigma = self.model.predict(X,return_std=True)
        mu_train = self.model.predict(X_train)

        mu_train_opt = np.min(mu_train)

        with np.errstate(divide='warn'):
            imp = mu - mu_train_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def pick(self,pool,batch_size,X_train=None):
        EI = self.expected_improvement(pool.X,X_train)
        pick_index = np.argpartition(EI, batch_size)
        return pool.pop(pick_index[0:batch_size])

    def update(self,X_train,Y_train):
        self.model.fit(X=X_train,y=Y_train)

def construct_graph(X, n_neighbor=100):
    distance = np.zeros((len(X), len(X)))
    # for i in range(len(X)):
    #     for j in range(len(X)):
    #         distance[i][j] = np.sum(abs(X[i] - X[j]))
    distance = spatial.distance.squareform(spatial.distance.pdist(X, 'cityblock'))

    G = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        idx = np.argpartition(distance[i], n_neighbor + 1)
        G[i][idx[0:n_neighbor + 1]] = 1
        G[i][i] = 0

    return G


# G = sparse.load_npz("Graph.npz").tolil()

# total size of data
size = np_data.shape[0]
# initial training size
init_size = 10
# maximum training size
max_size = 100
# number of new samples in each iteration
batch_size = 5

np_data[:, 0:-1] = preprocessing.StandardScaler().fit_transform(np_data[:, 0:-1])
G = sparse.csr_matrix(construct_graph(np_data[:, 0:-1], n_neighbor=20))

selector_list = [
    Selector_Rank(),
    Selector_Rand(),
    Selector_GP_opt(),
    Selector_GEIST(),
    Selector_SMAC(),
]
k = 0  #
n_exp = 30
output = np.zeros((len(selector_list), len(range(init_size, max_size + 1, batch_size))))
output_raw = np.zeros((len(selector_list), len(range(init_size, max_size + 1, batch_size)), n_exp))
for k in range(n_exp):  # repeat several times
    index = np.random.permutation(len(np_data))
    np_data = np_data[index]
    # np_data = np.random.permutation(np_data)
    G = G[index]
    G = G[:, index]

    X = np_data[:, 0:-1]
    Y_raw = np_data[:, -1]
    lb = 0.0
    ub = 1.0
    width = ub - lb
    Y_min = np.min(1.0 / Y_raw)
    Y_max = np.max(1.0 / Y_raw)
    scale = width / (Y_max - Y_min)
    Y = (1.0 / Y_raw - Y_min) * scale + lb
    best = np.amin(Y_raw)
    X_rest = X[init_size:, :]
    Y_rest = Y[init_size:]

    # GEIST_X_rest = index[init_size:X.shape[0]]
    # GEIST_X_train = index[0:init_size]
    # GEIST_threshold = np.percentile(Y_raw[GEIST_X_train], 10)
    # GEIST_Y_cate = (Y_raw <= GEIST_threshold).astype(int)
    # GEIST_Y_train = GEIST_Y_cate[0:init_size]

    GEIST_threshold = np.percentile(Y_raw[0:init_size], 5)
    GEIST_Y_cate = (Y_raw <= GEIST_threshold).astype(int)
    GEIST_X_rest = np.arange(init_size, X.shape[0])
    GEIST_Y_rest = GEIST_Y_cate[GEIST_X_rest]
    GEIST_X_train = np.arange(init_size)
    GEIST_Y_train = GEIST_Y_cate[0:init_size]

    n_optimal = sum(GEIST_Y_cate)
    for i, selector in enumerate(selector_list):  # evaluate different strategies
        train_size = init_size
        X_train = X[0:init_size, :]
        Y_train = Y[0:init_size]
        pool = Pool(X_rest, Y_rest)
        n_batch = 0
        while train_size <= max_size:  # simulate optimization process
            if selector.name == 'GEIST':
                print('Exp %d, Selector %d, %d samples picked, %d/%d optimal points found, ' % (
                    k, i, train_size, sum(GEIST_Y_train), n_optimal),end='')
            else:
                print('Exp %d, Selector %d, %d samples picked' % (k, i, train_size))
            if selector.name == 'GEIST':
                selector.update(GEIST_X_train, GEIST_Y_train, G)
                output[i][n_batch] = output[i][n_batch] + np.min(Y_raw[GEIST_X_train])
                output_raw[i][n_batch][k] = np.min(Y_raw[GEIST_X_train])
                train_size += batch_size
                n_batch += 1

                pred = selector.model.predict(GEIST_X_rest)
                acc = sum((pred == GEIST_Y_rest)) / len(pred)
                print('rest acc: %f ' % acc, end='')
                optim_candidates_index = selector.pick(GEIST_X_rest, batch_size)
                GEIST_X_new = GEIST_X_rest[optim_candidates_index]
                GEIST_X_train = np.concatenate((GEIST_X_train, GEIST_X_new))
                GEIST_Y_train = np.concatenate((GEIST_Y_train, GEIST_Y_cate[GEIST_X_new]))
                GEIST_X_rest = np.delete(GEIST_X_rest, optim_candidates_index)
                GEIST_Y_rest = np.delete(GEIST_Y_rest, optim_candidates_index)

                pred = selector.model.predict(GEIST_X_new)
                Y_true = GEIST_Y_cate[GEIST_X_new]
                acc = sum((pred == Y_true)) / len(pred)
                print('batch acc: %f' % acc)

                continue
            elif selector.name == 'GP_opt' or selector.name == 'SMAC':
                selector.update(X_train, -Y_train)
            else:
                selector.update(X_train, Y_train)
            output[i][n_batch] = output[i][n_batch] + (1.0 / ((np.amax(Y_train) - lb) / scale + Y_min))
            output_raw[i][n_batch][k] = (1.0 / ((np.amax(Y_train) - lb) / scale + Y_min))
            train_size += batch_size
            n_batch += 1
            if selector.name == 'GP_opt' or selector.name == 'SMAC':
                X_new, Y_new = selector.pick(pool, batch_size, X_train)
            else:
                X_new, Y_new = selector.pick(pool, batch_size)
            X_train = np.concatenate((X_train, X_new))
            Y_train = np.concatenate((Y_train, Y_new))

            # for debug
            if i == 10:
                Y_test_pred = selector.model.predict(pool.X)
                Y_train_pred = selector.model.predict(X_train)

                uncertainty_test = get_uncertainty(selector.model, pool.X)
                uncertainty_train = get_uncertainty(selector.model, X_train)

                Y_total = np.concatenate((Y_train, pool.Y))
                Y_total_pred = np.concatenate((Y_train_pred, Y_test_pred))
                uncertainty_total = np.concatenate((uncertainty_train, uncertainty_test))
                best_index = np.argpartition(- Y_total, 20)[0:20]

                plt.subplot(1, 3, 1)
                plt.scatter(pool.Y, Y_test_pred, 3)
                plt.scatter(Y_train, Y_train_pred, 10, color='r')
                plt.scatter(Y_total[best_index], Y_total_pred[best_index], linewidths=30, color='y', marker='+')
                plt.subplot(1, 3, 2)
                plt.scatter(Y_test_pred, uncertainty_test, 3)
                plt.scatter(Y_train_pred, uncertainty_train, 10, color='r')
                plt.scatter(Y_total_pred[best_index], uncertainty_total[best_index], linewidths=30, color='y',
                            marker='+')
                plt.subplot(1, 3, 3)
                plt.scatter(pool.Y, uncertainty_test, 3)
                plt.scatter(Y_train, uncertainty_train, 10, color='r')
                plt.scatter(Y_total[best_index], uncertainty_total[best_index], linewidths=30, color='y', marker='+')
                plt.show()
            # for debug
output = output / n_exp

np.savetxt('output/output_' + data_path, X=output, delimiter=',')
np.save('output/raw_' + data_path, output_raw)

std = np.std(output_raw, axis=2)
np.savetxt('output/std_' + data_path, X=std, delimiter=',')

if len(sys.argv) <= 1:
    count = output.shape[1]
    for i, selector in enumerate(selector_list):
        picked_size = range(init_size, max_size + 1, batch_size)
        plt.plot(picked_size, output[i, :], marker=marker_list[i], label=label_list[i], markevery=int((count - 1) / 10))
    plt.legend()
    plt.xlabel('Number of samples')
    plt.ylabel('Execution time (sec)')
    plt.show()
