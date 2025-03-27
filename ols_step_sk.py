"""forward/backward features selection based on AIC/BIC.

This module do classical forward or backward selection
based on AIC on BIC using SciKit-Learn LinearRegression
(found in sklearn.linear_model)
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class LinearRegressionSelectionFeatureIC(LinearRegression):
    """
    Ordinary least squares Linear Regression with feature selection (AIC/BIC).
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation. Feature
    are selected using backward/forward/both algorithm using BIC or AIC.
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.
    start : a list of int giving the columns index of the starting model
            (ie the starting point); if empty only intercept in model
            (defaut is empty)
    lower : a list of int giving the columns index of the lower model
            (ie the minimal model allowed); if empty only intercept in model
            (defaut is empty)
    upper : a list of int giving the columns index of the upper model
            (ie the maximal model allowed) or "max" to select all variable
            in X (default to "max")
    direction : either "both", "forward" or "backward"
    crit : either "aic"/"AIC" or "bic"/"BIC"
    verbose : int, if 0 no verbose
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.
    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    feature_selected_ : list of variable included in the model
    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.
    """

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
        start=[],
        lower=[],
        upper="max",
        crit="bic",
        direction="both",
        verbose=0,
    ):
        """Setter for LinearRegressionSelectionFeatureIC class.

        Initialize object with all needed attributes.
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.start = start
        self.lower = lower
        self.upper = upper
        self.crit = crit
        self.direction = direction
        self.verbose = verbose


    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        sel = self.olsstep(X, y)
        if self.verbose>1:
            print("starting fitting")
        if isinstance(X, pd.DataFrame):
            XX = X.iloc[:, sel]
        else:
            XX = np.copy(X[:, sel])
        self.feature_selected_ = sel
        return super().fit(XX, y, sample_weight)

    def olsstep(self, XX, yy):
        """Backward selection for linear model with sklearn.

        Parameters:
        -----------
        self: object of class LinearRegressionSelectionFeatureIC
        XX (pandas DataFrame or numpy array): Dataframe or array with all
                   possible predictors
        yy (pandas DataFrame or numpy array): dataframe or array with response

        Returns:
        --------
        list: list of indexes selected by selected by forward/backward
              or both algorithm with crit criterion (intercept is excluded)
        """
        # direction
        if self.verbose>1:
            print("selection step (", XX.shape[1],"is intercept )")
        if isinstance(self.upper, str):
            if self.upper == "max":
                self.upper = list(range(XX.shape[1]))
            else:
                raise ValueError("upper must be a list of int or 'max'")
        if not (self.direction == "both" or self.direction == "forward" or
                self.direction == "backward"):
            raise ValueError(
                "direction error (should be both, forward or backward)")
        # self.criterion
        if not (self.crit == "aic" or self.crit == "AIC" or
                self.crit == "bic" or self.crit == "BIC"):
            raise ValueError("criterion error (should be AIC/aic or BIC/bic)")
        # dimensions
        n = XX.shape[0]
        p = XX.shape[1]
        # test of indexes
        if len(self.start) > 0:
            res = test_index(p, self.start)
            if not res:
                raise ValueError("index error in start")
        if len(self.lower) > 0:
            res = test_index(p, self.lower)
            if not res:
                raise ValueError("index error in lower")
        if len(self.upper) == 0:
            raise ValueError("no index in upper")
        else:
            res = test_index(p, self.upper)
            if not res:
                raise ValueError("index error in upper")
        # use numpy ndarray and intercept
        if isinstance(XX, pd.DataFrame):
            X = np.append(XX.values, np.ones((n, 1)), axis=1)
        else:
            X = np.append(XX, np.ones((n, 1)), axis=1)
        if isinstance(yy, pd.DataFrame):
            y = yy.values
        else:
            y = yy
        # explanatory variables for the 3 models (and add intercept)
        if len(self.start) > 0:
            start_explanatory = set(self.start) | {p}
        else:
            start_explanatory = {p}
        if len(self.lower) > 0:
            lower_explanatory = set(self.lower) | {p}
        else:
            lower_explanatory = {p}
        upper_explanatory = set(self.upper) | {p}
        # setting up the set "add" which contains the possible variable to add
        if self.direction == "both" or self.direction == "forward":
            add = upper_explanatory - start_explanatory
            # setting up the set "remove" which contains the
            # possible variable to remove
        if self.direction == "both" or self.direction == "backward":
            remove = start_explanatory - lower_explanatory
        # current point
        selected = start_explanatory
        Xs = X[:, list(selected)]
        reglin = LinearRegression(fit_intercept=False).fit(Xs, y)
        if self.crit == "aic" or self.crit == "AIC":
            current_score = aic(reglin, Xs, y)
        elif self.crit == "bic" or self.crit == "BIC":
            current_score = bic(reglin, Xs, y)
        if self.verbose > 1:
            print("----------------------------------------------")
            print((current_score, "Starting", selected))
        # main loop
        while True:
            scores_with_candidates = []
            if self.direction == "both" or self.direction == "backward":
                for candidate in remove:
                    tobetested = selected - set([candidate])
                    Xtbt = X[:, list(tobetested)]
                    reglin = LinearRegression(fit_intercept=False).fit(Xtbt, y)
                    if self.crit == "aic" or self.crit == "AIC":
                        score = aic(reglin, Xtbt, y)
                    elif self.crit == "bic" or self.crit == "BIC":
                        score = bic(reglin, Xtbt, y)
                    if self.verbose > 2:
                        print((score, "-", candidate))
                    scores_with_candidates.append((score, "-", candidate))
            if self.direction == "both" or self.direction == "forward":
                for candidate in add:
                    tobetested = selected | set([candidate])
                    Xtbt = X[:, list(tobetested)]
                    reglin = LinearRegression(fit_intercept=False).fit(Xtbt, y)
                    if self.crit == "aic" or self.crit == "AIC":
                        score = aic(reglin, Xtbt, y)
                    elif self.crit == "bic" or self.crit == "BIC":
                        score = bic(reglin, Xtbt, y)
                    if self.verbose > 2:
                        print((score, "+", candidate))
                    scores_with_candidates.append((score, "+", candidate))
            scores_with_candidates.sort()
            best_new_score, dircur, best_candidate = scores_with_candidates.pop(0)
            if current_score > best_new_score:
                if dircur == "+":
                    add = add - set([best_candidate])
                    selected = selected | set([best_candidate])
                    if self.direction == "both":
                        remove = remove | set([best_candidate])
                else:
                    remove = remove - set([best_candidate])
                    selected = selected - set([best_candidate])
                    if self.direction == "both":
                        add = add | set([best_candidate])
                current_score = best_new_score
                if self.verbose > 1:
                    print("----------------------------------------------")
                    print((current_score, "New Current", selected))
            else:
                break
        ll = list(selected)
        if len(ll) > 1:
            ll.sort()
            ll.pop()
            if self.verbose > 1:
                print("----------------------------------------------")
                print((current_score, "Final", ll))
            elif self.verbose == 1:
                print("Final:", ll)
            return ll
        else:
            if ll[0] == p:
                if self.verbose > 1:
                    print("----------------------------------------------")
                    print((current_score, "Final (only intercept)", ll))
                elif self.verbose == 1:
                    print("Final:", ll)
                return []
            else:
                print(ll)
                raise ValueError("only one variable but no intercept")
        # reglin = LinearRegression(fit_intercept=False).fit(Xf, y)
        print("oh oh")
        return None

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        if isinstance(X, pd.DataFrame):
            XX = X.iloc[:, self.feature_selected_]
        else:
            XX = X[:, self.feature_selected_]
        return self._decision_function(XX)

def bic(modele, X, y):
    """BIC for scikitlearn regression model.

    Parameters:
    ----------
    modele (scikitlearn linear_model.LinearRegression): linear regression model
    X (numpy array or pandas array): explanatory variables
    y (numpy array): response variable
    X (numpy array): variables explicatives
    y (numpy array): variable a expliquer

    Returns:
    --------
    BIC: BIC criterion smaller is better
    """
    n = X.shape[0]
    yhat = modele.predict(X)
    xi = modele.coef_.shape[1]
    e2 = np.square(y - yhat)
    bic = (n * (1 + np.log(2 * np.pi)) + n * np.log(e2.sum()/n)
           + (xi + 2) * np.log(n))
    return bic


def aic(modele, X, y):
    """AIC for scikitlearn regression model.

    Parameters:
    ----------
    modele (scikitlearn linear_model.LinearRegression): linear regression model
    X (numpy array or pandas array): explanatory variables
    y (numpy array): response variable

    Returns:
    --------
    AIC: AIC criterion smaller is better
    """
    n = X.shape[0]
    yhat = modele.predict(X)
    xi = modele.coef_.shape[1]
    e2 = np.square(y - yhat)
    aic = n * (1 + np.log(2 * np.pi)) + n * np.log(e2.sum()/n) + (xi + 2) * 2
    return aic


def test_index(p, indexes):
    """Test if indexes are between 0 and p-1.

    Parameters:
    ----------
    p (integer): number of columns of X
    indexes (list): indexes of variables to be chosen in X

    Returns:
    --------
    boolean: True if OK
    """
    mini = p
    maxi = -1
    for i in indexes:
        mini = min(mini, i)
        maxi = max(maxi, i)
    if maxi >= p or maxi < 0:
        return False
    if mini >= p or mini < 0:
        return False
    return True
