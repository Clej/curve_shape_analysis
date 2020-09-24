import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats
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
# directional outlier detector
from skfda.exploratory.depth import outlyingness_to_depth
from skfda.exploratory.outliers import DirectionalOutlierDetector
from skfda.exploratory.outliers._directional_outlyingness import directional_outlyingness_stats

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
            - fd_train : training data - FDataGrid
            - y_train : training labels - ndarray of shape (n_samples,)"""
        X_train = np.transpose(fd_train.data_matrix, axes=(0,2,1))
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
            - fd_test : FDataGrid of testing samples - its data matrix is an ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - pred : prediction using the specified decision rule"""
        self.fd_test = fd_test
        X_test = self.fd_test.data_matrix
        # project test samples
        test_proj = np.empty((len(X_test), X_test.shape[1], self.p))
        for i in range(len(X_test)):
            test_proj[i] = self.projwda(X_test[i])
        # predict labels
        self.y_pred = _predict_label_rule(test_proj, self.clf, self.contamination, self.out_label)[0]
        return self.y_pred

    def score_samples(self, fd_test: FDataGrid, return_threshold=False):
        """Predict scores for each sample in X_test using scoring function defined in predict_label_rule
        Arguments :
            - fd_test : FDataGrid of testing samples - its data matrix is an ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - prob : scores of the samples using the specified scoring in predict_label_rule"""
        self.fd_test = fd_test
        X_test = self.fd_test.data_matrix
        # project test samples
        test_proj = np.empty((len(X_test), X_test.shape[1], self.p))
        for i in range(len(X_test)):
            test_proj[i] = self.projwda(X_test[i])
        self.scores = _predict_label_rule(test_proj, self.clf, self.contamination, self.out_label)[1]
        self.is_scored = True
        if return_threshold :
            self.threshold = np.percentile(self.scores, (1-self.contamination)*100)
            return self.scores, self.threshold
        else :
            return self.scores

    def plot_projection(self, fd : FDataGrid, y=None, y_names=None):
        """Projects the samples in fd in the p-dimensional space used for Wasserstein Discriminant Analysis and plot them, according to their true labels y
        Arguments :
            - fd : FDataGrid whose elements have to be projected
            - y : true labels of the elements if possible
        Returns :
            - Plots the projected samples, with different values depending on the target
        """
        X = fd.data_matrix
        # project test samples
        projection = np.empty((len(X), X.shape[1], self.p))
        for i in range(len(X)):
            projection[i] = self.projwda(X[i])
        ca_proj = fda_feature.CurveAnalysis(FDataGrid(projection, sample_points=fd.sample_points[0]))
        if self.p == 1 :
            ca_proj.plot_grids(targets=y, target_names=y_names)
        else :
            ca_proj.plot_interaction(targets=y, target_names=y_names)


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


##################################################################################
##################### DIRECTIONAL OUTLYINGNESS DETECTOR ##########################
##################################################################################

def _SDO_multivariate(fd, pointwise=True):
    """Returns the Stagel-Donoho outlyingness for multivariate data.
    $$SDO(X(t)) = sup_{\lVert u\rVert=1}\frac{\lVert u^TX(t) - median(u^TX(t))\rVert}{MAD(u^TX(t))}$$
    
    In dimension 1 (already implemented in scikit-fda), the result is exactly computed ;
    but in higher dimensions, we have to compute this maximum.
    
    We use the parametric representation of the unit sphere in R^p and 
    choose a set of candidates to compute the value of this supremum.
    If $X = (x_1, x_2, \dots, x_p)$ is in the unit sphere in $R^p$, then it exists $\phi_1, \dots, \phi_{p-2}\in[0,\pi]$
        and $\phi_{p-1}\in[0,2\pi[$ such that :
    - $x_1 = \cos(\phi_1)$
    - $x_2 = \sin(\phi_1)\cos(\phi_2)$
    - ...
    - $x_{p-1} = \sin(\phi_1)\dots\sin(\phi_{p-2})\cos(\phi_{p-1})$
    - $x_p = \sin(\phi_1)\dots\sin(\phi_{p-1})$
    """
    X = fd.data_matrix 
    
    p = X.shape[2] # dimension of the multivariate data
    
    # creation of the set of candidates for u
    u = np.empty((p, 100))
    for dim in range(1, p):
        phi = np.random.uniform(0, np.pi, (dim,100))
        if dim==1:
            u[dim-1] = np.cos(phi)
        else :
            prod_sin = 1
            for j in range(dim-1):
                 prod_sin *= np.sin(phi[j])
            u[dim-1] = prod_sin*np.cos(phi[dim-1])
    prod_sin_last = 1
    phi_last = np.random.uniform(0, 2*np.pi, (p,100))
    for j in range(p):
        prod_sin_last *= np.sin(phi_last[j])
    u[p-1] = prod_sin_last
    
    u = u.transpose()
                 
    SDO = np.empty((X.shape[0], X.shape[1]))
    # for each $t$, find the value of the outlyingness thanks to the set of candidates from $u$
    for t in range(X.shape[1]):

        uX = np.dot(u, X[:,t,:].transpose()).transpose()
        SDO_candidates = np.abs(uX - np.median(uX, axis=0)) / \
                            scipy.stats.median_abs_deviation(uX, axis=0, scale=1 / 1.4826)
        SDO[:,t] = np.max(SDO_candidates, axis=1)
        
    return SDO

def _PD_multivariate(X, *, pointwise=True):
    """Returns the projection depth for multivariate data.

    The projection depth is the depth function associated with the
    Stagel-Donoho outlyingness.
    """

    depth = outlyingness_to_depth(_SDO_multivariate)

    return depth(X, pointwise=pointwise)


def _evaluate_without_scores(pred_labels, true_labels, out_code=-1):
    """Compute evaluation metrics for a binary classifier (part. outlier detection)
    Arguments:
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
    """
    pred = np.where(pred_labels==out_code)[0]
    true_outliers = np.where(true_labels==1)[0]
    false_outliers = np.where(true_labels==0)[0]
    
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
    return metrics

class DirOutlierDetector():
    """Directional Outlier Detector using Projection Depth.
    Adapted from DirectionalOutlierDetector of scikit-fda to make it able to work with multivariate functional data
    Attributes :
        - fit_predict : call fit_predict from scikit-fda method with projection depth : implemented in skfda for 1D case, 
            implemented in this file for multivariate case
        - eval_performances : only evaluation metrics computed from predictions are computed since scores are not defined for this method
        """
    def __init__(self, alpha=0.993):
        self.alpha = alpha
    def fit_predict(self, fd):
        if fd.dim_codomain == 1 :
            # simple call to the detector from scikit-fda with default depth
            self.model = DirectionalOutlierDetector(alpha=self.alpha)
        else :
            # call to the detector from scikit-fda with adapted projection depth
            self.model = DirectionalOutlierDetector(depth_method=_PD_multivariate, alpha=self.alpha)
        self.fd_test = fd
        self.y_pred = self.model.fit_predict(self.fd_test)
        return self.y_pred

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

    def DirOutlierStats(self, fd: FDataGrid):
        """Returns features used by the detector.
        Arguments :
            - fd : FDataGrid
        Returns :
            - A dictionary with the value of
                - the mean directional outlyingness
                - the variation directional outlyingness
        """
        if fd.dim_codomain == 1 :
            SO_stats = directional_outlyingness_stats(fd)
        else :
            SO_stats = directional_outlyingness_stats(fd, depth_method=_PD_multivariate)
        return {'Mean Outlyingness' : SO_stats.mean_directional_outlyingness,
                'Variational Outlyingness' : SO_stats.variation_directional_outlyingness}

    def MSplot(self, fd, variables, y=None, y_names=None):
        """Plots the Magnitude-Shape plot of some variables in a multivariate functional data
        Arguments :
            - fd : FDataGrid
            - variables : list of dimensions whose MS-plot has to be computed (list of integers)
            Example : 
                - if you want to plot the MS-plot of X_1, you put [0] in variables ; 
                - if you want to plot the MS-plot of (X_1, X_2), you put [0,1])
        Returns : 
            - Plots the MS-plot of variables"""
        dict_stats = self.DirOutlierStats(fd)
        MO, VO = dict_stats['Mean Outlyingness'], dict_stats['Variational Outlyingness']
        if len(variables) > 2 :
            raise NotImplementedError('We cannot plot Magnitude-Shape of more than 2 variables ; otherwise the plot is going to be in more than 4D...')
        elif len(variables) == 2 :
            ax = plt.subplot(111, projection='3d')
            if y is None :
                ax.scatter(MO[:,variables[0]], MO[:,variables[1]], VO)
                ax.set_xlabel("MO($X_{}$)".format(variables[0]+1))
                ax.set_ylabel("MO($X_{}$)".format(variables[1]+1))
                ax.set_zlabel("VO")
                ax.set_title("Magnitude-Shape plot of $X_{}$ and $X_{}$".format(variables[0]+1, variables[1]+1))
                plt.show()
            else :
                n_targets = len(y_names)
                if n_targets > 2:
                    col_map = [cm.jet(i) for i in np.linspace(0, 1, n_targets)]
                    colors = [col_map[t] for t in y]
                    ax.scatter(MO[:,variables[0]], MO[:,variables[1]], VO, color=colors)
                    ax.set_xlabel("MO($X_{}$)".format(variables[0]+1))
                    ax.set_ylabel("MO($X_{}$)".format(variables[1]+1))
                    ax.set_zlabel("VO")
                    ax.set_title("Magnitude-Shape plot of $X_{}$ and $X_{}$".format(variables[0]+1, variables[1]+1))
                    for k in range(n_targets):
                        ax.plot([], [], [], color=col_map[k],
                                label=y_names[k])
                    ax.legend()
                    plt.show()
                else :
                    target_counts = np.unique(y, return_counts=True)
                    maj_class = np.argmax(target_counts[1])
                    colors = ["grey" if t == target_counts[0][maj_class] else "red" for t in y]
                    ax.scatter(MO[:,variables[0]], MO[:,variables[1]], VO, color=colors)
                    ax.set_xlabel("MO($X_{}$)".format(variables[0]+1))
                    ax.set_ylabel("MO($X_{}$)".format(variables[1]+1))
                    ax.set_zlabel("VO")
                    ax.set_title("Magnitude-Shape plot of $X_{}$ and $X_{}$".format(variables[0]+1, variables[1]+1))
                    ax.plot([], [], [], color="grey", label="inlier")
                    ax.plot([], [], [], color="red", label="outlier")
                    ax.legend()
                    plt.show()

        elif len(variables) == 1:
            if y is None :
                plt.scatter(MO[:,variables[0]], VO)
                plt.xlabel("MO($X_{}$)".format(variables[0]+1))
                plt.ylabel("VO")
                plt.title("Magnitude-Shape plot of $X_{}$".format(variables[0]+1))
            else :
                n_targets = len(y_names)
                if n_targets > 2:
                    col_map = [cm.jet(i) for i in np.linspace(0, 1, n_targets)]
                    colors = [col_map[t] for t in y]

                    plt.scatter(MO[:,variables[0]], VO,color=colors)
                    plt.xlabel("MO($X_{}$)".format(variables[0]+1))
                    plt.ylabel("VO")
                    plt.title("Magnitude-Shape plot of $X_{}$".format(variables[0]+1))
                    for k in range(n_targets):
                        plt.plot([], [], color=col_map[k],
                                label=y_names[k])
                    plt.legend()
                    plt.show()
                else :
                    target_counts = np.unique(y, return_counts=True)
                    maj_class = np.argmax(target_counts[1])
                    colors = ["grey" if t == target_counts[0][maj_class] else "red" for t in y]
                    plt.scatter(MO[:,variables[0]], VO, color=colors)
                    plt.xlabel("MO($X_{}$)".format(variables[0]+1))
                    plt.ylabel("VO")
                    plt.title("Magnitude-Shape plot of $X_{}$".format(variables[0]+1))
                    plt.plot([], [], color="grey", label="inlier")
                    plt.plot([], [], color="red", label="outlier")
                    plt.legend()
                    plt.show()
        else:
            raise ValueError('You need to pass at least one integer in variables argument...')

    
    def eval_performances(self, fd_test: FDataGrid, y_test: np.array):
        """Evaluate the performances of the model in a given testing set. 
        Uses the function _evaluate defined as the beginning of the module"""

        if hasattr(self, 'fd_test') :
            if self.fd_test != fd_test :
                self.fd_test = fd_test
                self.y_pred = self.fit_predict(fd_test)

        else :
            self.fd_test = fd_test
            self.y_pred = self.fit_predict(fd_test)

        return _evaluate_without_scores(self.y_pred, y_test)