# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

class Base(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta
    def __init__(self,graph,max_iter=30):
        self.max_iter = max_iter
        self.graph = graph

    @abstractmethod
    def _build_propagation_matrix(self):
        raise NotImplementedError("Propagation matrix construction must be implemented to fit a model.")

    @abstractmethod
    def _build_base_matrix(self):
        raise NotImplementedError("Base matrix construction must be implemented to fit a model.")

    def _init_label_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        return np.zeros((n_samples,n_classes))

    def _arrange_params(self):
        """Do nothing by default"""
        pass

    def fit(self,x,y):
        """Fit a graph-based semi-supervised learning model
        All the input data is provided array X (labeled samples only)
        and corresponding label array y.
        Parameters
        ----------
        x : array_like, shape = [n_labeled_samples]
            Node IDs of labeled samples
        y : array_like, shape = [n_labeled_samples]
            Label IDs of labeled samples
        Returns
        -------
        self : returns an instance of self.
        """
        self.x_ = x
        self.y_ = y

        self._arrange_params()

        self.F_ = self._init_label_matrix()

        self.P_ = self._build_propagation_matrix()
        self.B_ = self._build_base_matrix()

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F_ = self._propagate()
            remaining_iter -= 1

        return self

    def _propagate(self):
        return self.P_.dot(self.F_) + self.B_

    def predict(self,x):
        """Performs prediction based on the fitted model
        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs
        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input node IDs
        """
        probas = self.predict_proba(x)
        #print(probas[40:50])
        return np.argmax(probas,axis=1)

    def predict_proba(self,x):
        """Predict probability for each possible label
        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs
        Returns
        -------
        probabilities : array_like, shape = [n_samples, n_classes]
            Probability distributions across class labels
        """
        z = np.sum(self.F_[x], axis=1)
        z[z==0] += 1 # Avoid division by 0
        return (self.F_[x].T / z).T


class CAMLP(Base):
    """Confidence-Aware Modulated Label Propagation (CAMLP) for GBSSL
    Parameters
    ----------
    beta : float > 0 (default = 0.1)
      Define importance between prior and evidence from neighbors
    H : array_like, shape = [n_classes, n_classes]
      Define affinities between labels
      if None, identity matrix is set
    max_iter : float
      maximum number of iterations allowed
    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.
    References
    ----------
    Yamaguchi, Y., Faloutsos, C., & Kitagawa, H. (2016, May).
    CAMLP: Confidence-Aware Modulated Label Propagation.
    In SIAM International Conference on Data Mining.
    """

    def __init__(self,graph=None,beta=0.1,H=None,max_iter=30):
        super(CAMLP,self).__init__(graph,max_iter)
        self.beta=beta
        self.H=H

    def _arrange_params(self):
        if self.H is None:
            n_classes = self.y_.max()+1
            self.H = np.identity(n_classes)
        self.Z = self._build_normalization_term()

    def _propagate(self):
        return self.P_.dot(self.F_).dot(self.H) + self.B_

    def _build_normalization_term(self):
        d = self.graph.sum(axis=1).T.A[0]
        return sparse.diags(1.0/(1.0+d*self.beta),offsets=0)

    def _build_propagation_matrix(self):
        return self.Z.dot(self.beta*self.graph)

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.ones((n_samples,n_classes))/float(n_classes)
        B[self.x_] = 0
        B[self.x_,self.y_] = 1
        return self.Z.dot(B)