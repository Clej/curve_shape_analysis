import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import cm
# metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
# outlier detection
from sklearn.ensemble import IsolationForest
from FIF import FIF, MFIF
# dimensionality reduction 
from ot.dr import wda
# curve analysis
from CurveAnalysis import fda_feature
from skfda.representation.grid import FDataGrid

def _evaluate(scores, pred_labels, true_labels, out_code=-1):
    """Compute evaluation metrics for a binary classifier (part. outlier detection)
    Arguments:
        - scores : the higher, the more the sample is considered as an anomaly
        - pred_labels : labels of each sample - out_code for outliers
        - true_labels : true labels of the samples : 1 for outliers, 0 for inliers
        - out_code : code for outliers : -1 in this library
    Returns :
        - metrics : dictionary with the following metrics :
            - TP : number of True Positive predictions
            - FP : number of False Postive predictions
            - TN : number of True Negative predictions
            - FN : number of False Negative predictions
            - Precision : TP / (TP+FP)
            - Recall : TP / (TP+FN)
            - Accuracy : (TP+TN) / (TP+TN+FP+FN)
            - Balanced accuracy : 1/2 * (TP/(TP+FN) + TN/(FP+TN))
            - AUC : area under the ROC curve
            - AP : area under the precision-recall curve
    """
    pred = np.where(pred_labels==out_code)[0]
    true_outliers = np.where(true_labels==1)[0]
    false_outliers = np.where(true_labels==0)[0]
    fpr, tpr, _ = roc_curve(true_labels, scores)
    area = auc(fpr, tpr)
    ap = average_precision_score(true_labels, scores)
    
    metrics = {}
    metrics['TP'] = len([i for i in pred if i in true_outliers])
    metrics['FP'] = len([i for i in pred if i not in true_outliers])
    metrics['TN'] = len([i for i in false_outliers if i not in pred])
    metrics['FN'] = len([i for i in true_outliers if i not in pred])
    metrics['Precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics['Recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    if metrics['Precision']==0 and metrics['Recall']==0:
        metrics["f1-score"] = 0
    else:
        metrics["f1-score"] = 2*(metrics["Precision"] * metrics["Recall"])/(metrics["Precision"] + metrics["Recall"])
    metrics['Accuracy'] = (metrics['TP'] + metrics['TN']) \
                            / (metrics['TP'] + metrics['TN'] \
                               + metrics['FP'] + metrics['FN'])
    metrics['BA'] = 1/2 * (metrics['TP'] / (metrics['TP'] + metrics['FN']) \
                           + metrics['TN'] / (metrics['FP'] + metrics['TN']))
    metrics['AUC'] = area
    metrics['AP'] = ap
    
    return metrics


##################################################################################
############################ ISOLATION FOREST ####################################
##################################################################################

class IForest():
    """Different IsolationForest implementations adapted to be able to work with FDataGrid objects 
        from scikit-fda library.
    Arguments :
        - params : parameters to initialize the model
        - functional : either to use a functional implementation of the algorithm
        - contamination : contamination level of the dataset
    Attributes :
        - fit : fit the outlier detection model to a training data
        - predict : predict the labels for a testing data
        - score_sampels : returns the anomaly scores for a testing data
        - eval_performances : computes classification metrics to evaluate the model in a testing data
        - plot_detection : plots the results of the outlier detection"""

    def __init__(self, contamination, functional: bool = False, **params):
        self.params = params
        self.functional = functional
        self.contamination = contamination
        self.is_scored = False
        if functional == False :
            # In a non functional context, we use IsolationForest from scikit-learn
            self.model = IsolationForest(**self.params, contamination=self.contamination)

    def fit(self, fd_train: FDataGrid):
        if self.functional == False :
            if fd_train.dim_codomain > 1 :
                # multivariate functional data cannot be fed to this model
                raise ValueError("Functional Data must be univariate")
            else :
                self.model.fit(fd_train.data_matrix[...,0])
        else :
            if fd_train.dim_codomain == 1 :
                # univariate functional data
                self.model = FIF.FIForest(fd_train.data_matrix[...,0], time=fd_train.sample_points[0], 
                                            innerproduct='auto', **self.params)
            else :
                # multivariate functional data
                self.model = MFIF.MFIForest(np.transpose(fd_train.data_matrix, axes=(0,2,1)), 
                                            time=fd_train.sample_points[0], innerproduct='auto1', **self.params)
    
    def predict(self, fd_test: FDataGrid):
        """Predicts the labels in a given testing set
        Arguments :
            - fd_test : FDataGrid 
        Returns :
            - y_pred : predicted labels. Either 1 or -1 if the sample is considered as 
                an inlier (1) or outlier (-1)"""

        self.fd_test = fd_test
        if self.functional == False :
            if fd_test.dim_codomain > 1 :
                raise ValueError("Functional Data must be univariate")
            else :
                self.y_pred = self.model.predict(fd_test.data_matrix[...,0])
        else :
            if fd_test.dim_codomain == 1 :
                self.scores = self.model.compute_paths(fd_test.data_matrix[...,0])
                self.y_pred = self.model.predict_label(self.scores, contamination=self.contamination)
            else :
                self.scores = self.model.compute_paths(np.transpose(fd_test.data_matrix, axes=(0,2,1)))
                self.y_pred = self.model.predict_label(self.scores, contamination=self.contamination)
            self.is_scored = True
        return self.y_pred

    def score_samples(self, fd_test: FDataGrid, return_threshold: bool = False):
        """Returns the anomaly scores of samples in a testing set
        Arguments :
            - fd_test : FDataGrid
            - return_threshold : either to return the value of the threshold based on the contamination level
        Returns :
            - scores : np.array of the anomaly scores of each sample in fd_test"""

        self.fd_test = fd_test
        if self.is_scored :
            # scores have been already computed (in functional context, we use the scores to predict the labels)
            if return_threshold == True :
                return self.scores, np.percentile(self.scores, 100 * (1-self.contamination))
            else :
                return self.scores
        else :
            if self.functional == False :
                if fd_test.dim_codomain > 1 :
                    raise ValueError("Functional Data must be univariate")
                else :
                    self.scores = - self.model.score_samples(fd_test.data_matrix[...,0])
            else :
                if fd_test.dim_codomain == 1 :
                    self.scores = self.model.compute_paths(fd_test.data_matrix[...,0])
                else :
                    self.scores = self.model.compute_paths(np.transpose(fd_test.data_matrix, axes=(0,2,1)))
            if return_threshold == True :
                return self.scores, np.percentile(self.scores, 100 * (1-self.contamination))
            else :
                return self.scores
            self.is_scored = True

    def eval_performances(self, fd_test: FDataGrid, y_test: np.array):
        """Evaluate the performances of the model in a given testing set. 
        Uses the function _evaluate defined as the beginning of the module"""

        if hasattr(self, 'fd_test') and self.fd_test == fd_test :
            # prediction or scoring has already been done using this testing set
            if self.is_scored :
                # scoring has been done in this testing set
                if not hasattr(self, 'y_pred') :
                    # only scoring has been done
                    self.y_pred = self.predict(fd_test)
            else :
                # only prediction has been done (can be only non-functional context here)
                assert self.functional == False
                self.scores = self.score_samples(fd_test)
        else :
            # neither has been done
            self.scores = self.score_samples(fd_test)
            self.y_pred = self.predict(fd_test)

        return _evaluate(self.scores, self.y_pred, y_test)

    def plot_detection(self, fd_test: FDataGrid, plot_interaction: bool = False) :
        if hasattr(self, 'fd_test') and self.fd_test == fd_test :
            # prediction or scoring have already been done in this testing set
            if not hasattr(self, 'y_pred') :
                # predictions haven't been done in this testing set
                self.y_pred = self.predict(fd_test)
        else :
            self.y_pred = self.predict(fd_test)
        curve_analysis = fda_feature.CurveAnalysis(fd_test)
        if not plot_interaction :
            targets = np.array(self.y_pred, dtype='int')
            targets[targets==-1] = 0
            curve_analysis.plot_grids(targets=targets, target_names=["outlier","inlier"])
        else : 
            targets = np.array(self.y_pred, dtype='int')
            targets[targets==-1] = 0
            curve_analysis.plot_interaction(targets=targets, target_names=["outlier","inlier"])

    def plot_scores(self, fd_test: FDataGrid, targets=None, target_names=None):
        
        # score the samples and get the threshold value
        self.scores, self.threshold = self.score_samples(fd_test, return_threshold=True)

        order = np.argsort(self.scores)
        ranks = np.argsort(order)
        S_sort = np.sort(self.scores)
        if targets is not None :
            n_targets = len(target_names)
            col_map = [cm.jet(i) for i in np.linspace(0, 1, n_targets)]
            colors = {t : col_map[t] for t in targets}
            for i in range(n_targets):
                plt.scatter(ranks[np.where(targets==i)[0]], S_sort[ranks[np.where(targets==i)[0]]],
                                color=colors[i])
            for k in range(n_targets):
                plt.plot([], [], color=col_map[k], label=target_names[k])
        else :
            plt.scatter(range(len(self.scores)), S_sort, color="grey")
        
        plt.hlines(y=self.threshold, xmin=0, xmax=len(self.scores), linestyle='dashed', label="threshold")
        plt.legend(loc='best')
        plt.title("Anomaly scores of the Test curves with outliers' colored")
        plt.xlabel("Index of sorted curves")
        plt.ylabel("Scores")
        plt.show()



##################################################################################
################## WASSERSTEIN DISCRIMINANT ANALYSIS + KNN #######################
##################################################################################

def _predict_label_rule(test, clf, perc, out_label=1):
    """Apply the outlier decision rule given a percentage of contamination 
    Arguments :
        - X_test : testing samples - ndarray of shape (n_samples, n_points, dimension)
        - clf : fitted sklearn classifier
        - perc : contamination level (proportion of outliers to detect)
    Returns :
        - pred_lab : predicted labels after applying decision rule with perc
        - props_out : proportion of outlying points for each curve, considered as anomaly scores"""
    pred_lab = np.ones(len(test))
    props_out = np.empty(len(test))
    for i in range(len(test)):
        preds_points = clf.predict(test[i])
        prop_out = len(np.where(preds_points==out_label)[0])/len(preds_points)
        props_out[i] = prop_out
    threshold = np.percentile(props_out, (1-perc)*100)
    pred_lab[np.where(props_out > threshold)[0]] = -1
    return pred_lab, props_out

class CustomClassifierWDA():
    """Implements a custom SUPERVISED BINARY classifier for input samples obtained after projection with WDA
    Using a known classifier from scikit-learn, training and predicting steps are different from classical ones :
    - fitting : use training samples corresponding to the projected mean curves of each class
    - prediction : predict the labels for all the points in each curve, and score each curve 
    according to the percentage of points detected as outliers by the classifier. 
    The scores will correspond to the probabilities and outlier detection for each curve is done using a percentile
    defined by the user at the beginning
    """
    def __init__(self, clf, projection_size, reg, contamination, out_label=1, **params):
        """Initialize the classifier
            - clf : classifier (sklearn)
            - contamination : contamination level to separate outliers/non-outliers
            - params : parameters for clf (depends on the classifier - to be consistent with clf)"""
        self.clf = clf(**params)
        self.p = projection_size
        self.reg = reg
        self.params = params
        self.contamination = contamination
        self.is_scored = False
        self.out_label = out_label

    def fit(self, fd_train: FDataGrid, y_train: np.array):
        """Fits the classifier using median of each class of the training samples 
        Arguments :
            - X_train : training data - ndarray of shape (n_points, n_dim)
            - y_train : training labels - ndarray of shape (n_samples,)"""
        X_train = fd_train.data_matrix
        # compute mean of each class and label them
        mean_outliers = np.mean(X_train[np.where(y_train==1)[0]], axis=0).transpose()
        mean_inliers = np.mean(X_train[np.where(y_train==0)[0]], axis=0).transpose()
        train = np.concatenate((mean_outliers, mean_inliers))
        train_lab = np.zeros(len(train))
        train_lab[0:len(mean_outliers)] = 1
        # project sampels
        P0 = np.random.randn(train.shape[1], self.p)
        P0 /= np.sqrt(np.sum(P0**2, 0, keepdims=True))
        self.Pwda, self.projwda = wda(train, train_lab, self.p, self.reg, k=10, maxiter=100, P0=P0)
        train_proj = self.projwda(train)
        # fit classifier
        self.clf.fit(train_proj, train_lab)

    def predict(self, fd_test: FDataGrid):
        """Predict labels for each sample in X_test using decision rule defined in predict_label_rule and a contamination level
        Arguments :
            - X_test : testing samples - ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - pred : prediction using the specified decision rule"""
        self.fd_test = fd_test
        X_test = self.fd_test.data_matrix
        # project test samples
        test_proj = np.empty((len(X_test), X_test.shape[2], self.p))
        for i in range(len(X_test)):
            test_proj[i] = self.projwda(X_test[i].transpose())
        # predict labels
        self.y_pred = _predict_label_rule(test_proj, self.clf, self.contamination, self.out_label)[0]
        return self.y_pred

    def score_samples(self, fd_test: FDataGrid):
        """Predict scores for each sample in X_test using scoring function defined in predict_label_rule
        Arguments :
            - X_test : testing samples - ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - prob : scores of the samples using the specified scoring in predict_label_rule"""
        self.fd_test = fd_test
        X_test = self.fd_test.data_matrix
        # project test samples
        test_proj = np.empty((len(X_test), X_test.shape[2], self.p))
        for i in range(len(X_test)):
            test_proj[i] = self.projwda(X_test[i].transpose())
        self.scores = _predict_label_rule(test_proj, self.clf, self.contamination, self.out_label)[1]
        self.is_scored = True
        return self.scores

    def plot_scores(self, fd_test: FDataGrid, targets=None, target_names=None):
        
        # score the samples and get the threshold value
        self.scores = self.score_samples(fd_test)

        order = np.argsort(self.scores)
        ranks = np.argsort(order)
        S_sort = np.sort(self.scores)
        if targets is not None :
            n_targets = len(target_names)
            col_map = [cm.jet(i) for i in np.linspace(0, 1, n_targets)]
            colors = {t : col_map[t] for t in targets}
            for i in range(n_targets):
                plt.scatter(ranks[np.where(targets==i)[0]], S_sort[ranks[np.where(targets==i)[0]]],
                                color=colors[i])
            for k in range(n_targets):
                plt.plot([], [], color=col_map[k], label=target_names[k])
        else :
            plt.scatter(range(len(self.scores)), S_sort, color="grey")
        
        # plt.hlines(y=self.threshold, xmin=0, xmax=len(self.scores), linestyle='dashed', label="threshold")
        # plt.legend(loc='best')
        plt.title("Anomaly scores of the Test curves with outliers' colored")
        plt.xlabel("Index of sorted curves")
        plt.ylabel("Scores")
        plt.show()

    def eval_performances(self, fd_test: FDataGrid, y_test: np.array):
        """Evaluate the performances of the model in a given testing set. 
        Uses the function _evaluate defined as the beginning of the module"""

        if hasattr(self, 'fd_test') and self.fd_test == fd_test :
            # prediction or scoring has already been done using this testing set
            if self.is_scored :
                # scoring has been done in this testing set
                if not hasattr(self, 'y_pred') :
                    # only scoring has been done
                    self.y_pred = self.predict(fd_test)
            else :
                # only prediction has been done
                self.scores = self.score_samples(fd_test)
        else :
            # neither has been done
            self.scores = self.score_samples(fd_test)
            self.y_pred = self.predict(fd_test)

        return _evaluate(self.scores, self.y_pred, y_test)