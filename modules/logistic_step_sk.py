"""forward/backward features selection based on AIC/BIC.

This module do classical forward or backward selection
based on AIC on BIC using SciKit-Learn LogisticRegression
(found in sklearn.linear_model)
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class LogisticRegressionSelectionFeatureIC(LogisticRegression):
    """
    Logistic least squares Linear Regression with feature selection (AIC/BIC).
    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,). Should be 0/1.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Numerical solver to use.

    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

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

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """

    def __init__(self, *, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='newton-cholesky', max_iter=100,
                 multi_class='deprecated', warm_start=False, n_jobs=None,
                 start=[], lower=[], upper="max", crit="bic", direction="both",
                 verbose=0):
        """Setter for LogisticRegressionSelectionFeatureIC class.

        Initialize object with all needed attributes.
        """
        self.penalty = None
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = None
        self.start = start
        self.lower = lower
        self.upper = upper
        self.crit = crit
        self.direction = direction
        self.verbose = verbose


    def fit(self, X, y, sample_weight=None):
        """
        Fit logistic model.
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
        sel = self.glmstep(X, y)
        if self.verbose>1:
            print("starting fitting")
        if isinstance(X, pd.DataFrame):
            XX = X.iloc[:, sel]
        else:
            XX = np.copy(X[:, sel])
        self.selected_features_ = sel
        return super().fit(XX, y, sample_weight)

    def glmstep(self, XX, yy):
        """Stepwise selection for Logistic Regression with sklearn.

        Parameters:
        -----------
        self: object of class LogisticRegressionSelectionFeatureIC
        XX (pandas DataFrame or numpy array): Dataframe or array with all
                   possible predictors
        yy (pandas DataFrame or numpy array): dataframe or array with response

        Returns:
        --------
        list: list of indexes selected by selected by forward/backward
              or both algorithm with crit criterion (intercept is excluded)
        """
        # test y 0/1
        if set(yy) != set((0,1)):
            raise ValueError("Y values must be 0/1")
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
            X = np.append(XX.to_numpy(), np.ones((n, 1)), axis=1)
        else:
            X = np.append(XX, np.ones((n, 1)), axis=1)
        if isinstance(yy, pd.DataFrame):
            y = yy.to_numpy()
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
        reglog =  LogisticRegression(penalty=None, dual=self.dual, \
                                                tol = self.tol, C=self.C, \
                                                fit_intercept=False,\
                                                intercept_scaling = self.intercept_scaling,\
                                                class_weight = self.class_weight, \
                                                random_state = self.random_state, \
                                                solver = self.solver, \
                                                max_iter = self.max_iter, \
                                                warm_start = self.warm_start, \
                                                n_jobs = self.n_jobs).fit(Xs, y)
        if self.crit == "aic" or self.crit == "AIC":
            current_score = aic(reglog, Xs, y)
        elif self.crit == "bic" or self.crit == "BIC":
            current_score = bic(reglog, Xs, y)
        if self.verbose > 1:
            print("----------------------------------------------")
            print(f"Crit: {current_score:.3e}, Starting with: {selected}")
        # main loop
        while True:
            scores_with_candidates = []
            if self.direction == "both" or self.direction == "backward":
                for candidate in remove:
                    tobetested = selected - set([candidate])
                    Xtbt = X[:, list(tobetested)]
                    reglog = LogisticRegression(penalty=None, dual=self.dual, \
                                                tol = self.tol, C=self.C, \
                                                fit_intercept=False,\
                                                intercept_scaling = self.intercept_scaling,\
                                                class_weight = self.class_weight, \
                                                random_state = self.random_state, \
                                                solver = self.solver, \
                                                max_iter = self.max_iter, \
                                                warm_start = self.warm_start, \
                                                n_jobs = self.n_jobs).fit(Xtbt, y)
                    if self.crit == "aic" or self.crit == "AIC":
                        score = aic(reglog, Xtbt, y)
                    elif self.crit == "bic" or self.crit == "BIC":
                        score = bic(reglog, Xtbt, y)
                    if self.verbose > 2:
                        print((score, "-", candidate))
                    scores_with_candidates.append((score, "-", candidate))
            if self.direction == "both" or self.direction == "forward":
                for candidate in add:
                    tobetested = selected | set([candidate])
                    Xtbt = X[:, list(tobetested)]
                    reglog = LogisticRegression(penalty=None, dual=self.dual, \
                                                tol = self.tol, C=self.C, \
                                                fit_intercept=False,\
                                                intercept_scaling = self.intercept_scaling,\
                                                class_weight = self.class_weight, \
                                                random_state = self.random_state, \
                                                solver = self.solver, \
                                                max_iter = self.max_iter, \
                                                warm_start = self.warm_start, \
                                                n_jobs = self.n_jobs).fit(Xtbt, y)
                    if self.crit == "aic" or self.crit == "AIC":
                        score = aic(reglog, Xtbt, y)
                    elif self.crit == "bic" or self.crit == "BIC":
                        score = bic(reglog, Xtbt, y)
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
                    print(f"Crit: {current_score:.3e}, New Current: {selected}")
            else:
                break
        ll = list(selected)
        if len(ll) > 1:
            ll.sort()
            ll.pop()
            if self.verbose > 1:
                if (self.fit_intercept):
                    print(f"Crit: {current_score:.3e}, Final (intercept must be added): {ll}")
                else:
                    print(f"Crit: {current_score:.3e}, Final : {ll}")
            elif self.verbose == 1:
                print("Final:", ll)
            return ll
        else:
            if ll[0] == p:
                if self.verbose > 1:
                    print("----------------------------------------------")
                    print(f"Crit: {current_score:.3e}, Final (intercept must be added): {ll}")
                elif self.verbose == 1:
                    print("Final:", ll)
                return []
            else:
                print(ll)
                raise ValueError("only one variable but no intercept")
        print("uh oh, return should be done before, bug...")
        return None

    def predict(self, X):
        """
        Predict using the logistic model.

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
            XX = X.iloc[:, self.selected_features_]
        else:
            XX = X[:, self.selected_features_]
        return super().predict(XX)

    def predict_proba(self, X):
        """
        Predict proba using the logistic model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns proba values.
        """
        if isinstance(X, pd.DataFrame):
            XX = X.iloc[:, self.selected_features_]
        else:
            XX = X[:, self.selected_features_]
        return super().predict_proba(XX)

def bic(modele, X, y):
    """BIC for scikitlearn regression model.

    Parameters:
    ----------
    modele (scikitlearn linear_model.LogisticRegression): logistic regression model
    X (numpy array or pandas array): explanatory variables
    y (numpy array): response variable
    X (numpy array): variables explicatives
    y (numpy array): variable a expliquer

    Returns:
    --------
    BIC: BIC criterion smaller is better
    """
    n = X.shape[0]
    xi = modele.coef_.shape[1]
    ll1 = y *modele.predict_log_proba(X)[:,1]
    ll0 = (1 - y) * modele.predict_log_proba(X)[:,0]
    ll = ll0.sum() + ll1.sum()
    bic = -2 * ll  + (xi + 1) * np.log(n)
    return bic


def aic(modele, X, y):
    """AIC for scikitlearn regression model.

    Parameters:
    ----------
    modele (scikitlearn linear_model.LogisticRegression): logistic regression model
    X (numpy array or pandas array): explanatory variables
    y (numpy array): response variable

    Returns:
    --------
    AIC: AIC criterion smaller is better
    """
    n = X.shape[0]
    xi = modele.coef_.shape[1]
    ll1 = y *modele.predict_log_proba(X)[:,1]
    ll0 = (1 - y) * modele.predict_log_proba(X)[:,0]
    ll = ll0.sum() + ll1.sum()
    aic = -2 * ll + (xi + 1) * 2
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


if __name__ == "__main__":
    print("glm_step.py is being run directly ??")
