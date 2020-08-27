import numpy as np
from skfda.representation.grid import FDataGrid
from sklearn.ensemble import IsolationForest
import FIF, MFIF
# metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score


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
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
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

class IForest():
    def __init__(self, functional : bool = False, contamination, **params):
        self.params = params
        self.functional = functional
        self.contamination = contamination
        if functional == False :
            self.model = IsolationForest(**self.params)

    def fit(self, fd_train : FDataGrid):
        if self.functional == False :
            if fd_train.dim_codomain > 1 :
                raise ValueError("Data must be univariate")
            else :
                self.model.fit(fd_train.data_matrix[...,0])
        else :
            if fd_train.dim_codomain == 1 :
                self.model = FIF.FIForest(fd_train.data_matrix[...,0], **params)
            else :
                self.model = MFIF.MFIForest(np.transpose(fd_train.data_matrix, axes=(0,2,1)), **params)
    
    def predict(self, fd_test : FDataGrid):
        if self.functional == False :
            if fd_train.dim_codomain > 1 :
                raise ValueError("Data must be univariate")
            else :
                y_pred = self.model.predict(fd_test.data_matrix[...,0])
        else :
            if fd_train.dim_codomain == 1 :
                scores = self.model.compute_paths(fd_test.data_matrix[...,0])
                y_pred = self.model.predict_label(scores, contamination=self.contamination)
            else :
                scores = self.model.compute_paths(np.transpose(fd_test.data_matrix, axes=(0,2,1)))
                y_pred = self.model.predict_label(scores, contamination=self.contamination)
        return y_pred

    def score_samples(self, fd_test : FDataGrid, return_threshold : bool = False):
        if self.functional == False :
            if fd_train.dim_codomain > 1 :
                raise ValueError("Data must be univariate")
            else :
                scores_test = - self.model.score_samples(fd_test.data_matrix[...,0])
        else :
            if fd_train.dim_codomain == 1 :
                scores_test = self.model.compute_paths(fd_test.data_matrix[...,0])
            else :
                scores_test = self.model.compute_paths(np.transpose(fd_test.data_matrix, axes=(0,2,1)))
        if return_threshold == True :
            return scores_test, np.percentile(scores_test, 100 * (1-self.contamination), interpolation = 'lower')
        else :
            return scores_test

    def eval_performances(self, fd_test : FdataGrid, y_test : np.array):
        scores_test = self.score_samples(fd_test)
        y_pred = self.model.predict(fd_test)
        return _evaluate(scores_test, y_pred, y_test)

def predict_label_rule(test, clf, perc, out_label=1):
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
    """Implements a custom classifier for input samples obtained after projection with WDA
    Using a known classifier from scikit-learn, training and predicting steps are different from classical ones :
    - fitting : use training samples corresponding to the projected mean curves of each class
    - prediction : predict the labels for all the points in each curve, and score each curve 
    according to the percentage of points detected as outliers by the classifier. 
    The scores will correspond to the probabilities and outlier detection for each curve is done using a percentile
    defined by the user at the beginning
    """
    def __init__(self, clf, contamination=0.01, out_label=-1, **params):
        """Initialize the classifier
            - clf : classifier (sklearn)
            - contamination : contamination level to separate outliers/non-outliers
            - params : parameters for clf (depends on the classifier - to be consistent with clf)"""
        self.clf = clf(**params)
        self.params = params
        self.contamination = contamination
        self.out_label = out_label
    def fit(self, fd_train : FDataGrid, y_train : np.array):
        """Fits the classifier using training samples 
        Arguments :
            - X_train : training data - ndarray of shape (n_points, n_dim)
            - y_train : training labels - ndarray of shape (n_samples,)"""
        X_train = fd_train.data_matrix[...,0]
        self.clf.fit(X_train, y_train)

    def predict(self, fd_test : FDataGrid):
        """Predict labels for each sample in X_test using decision rule defined in predict_label_rule and a contamination level
        Arguments :
            - X_test : testing samples - ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - pred : prediction using the specified decision rule"""
        X_test = fd_test.data_matrix[...,0]
        pred = _predict_label_rule(X_test, self.clf, self.contamination, self.out_label)[0]
        return pred

    def score_samples(self, fd_test : FDataGrid):
        """Predict scores for each sample in X_test using scoring function defined in predict_label_rule
        Arguments :
            - X_test : testing samples - ndarray of shape (n_samples, n_points, n_dim)
        Returns :
            - prob : scores of the samples using the specified scoring in predict_label_rule"""
        X_test = fd_test.data_matrix[...,0]
        score = _predict_label_rule(X_test, self.clf, self.contamination, self.out_label)[1]
        return score

    def eval_performances(self, fd_test : FdataGrid, y_test : np.array):
        scores_test = self.score_samples(fd_test)
        y_pred = self.model.predict(fd_test)
        return _evaluate(scores_test, y_pred, y_test)