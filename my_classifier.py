from numpy import *

from sklearn.tree import DecisionTreeClassifier
from skmultiflow.core.utils.data_structures import InstanceWindow
from skmultiflow.core.utils.utils import *


class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.window = InstanceWindow(max_size=window_size)
        self.H = []
        # Current classifier
        self.h = 0
        self.max_models = max_models

    def partial_fit(self, X, y=None, classes=None):
        # if not initialized
        if self.H is None:
            self.H = []
        r, c = get_dimensions(X)
        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        self.h %= 10
        self.H[clf] = clf
        self.h += 1

        # N.B.: The 'classes' option is not important for this classifier
        # HINT: You can build a decision tree model on a set of data like this:
        #       h = DecisionTreeClassifier()
        #       h.fit(X_batch,y_batch)
        #       self.H.append(h) # <-- and append it to the ensemble

        return self

    def predict(self, X):
        N, D = X.shape
        # You also need to change this line to return your prediction instead of 0s:
        # TODO predict using the max_models, and return the majority class
        y = []
        for clf, i in self.H, range(self.max_models):
            y.append(clf.predict(X))

        return zeros(N)
