import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd


def bestols(data, upper, mustbe="1"):
    """Backward selection for linear model (with formula)
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    upper : a string giving the upper model (containing mustbe variables)
    mustbe : a string giving the variables that must be included (separated by +)
    Returns:
    --------
    df: a dataframe with all possible results
    """
    ## starting point
    import itertools
    formula = upper.split("~")
    response = formula[0].strip()
    mustbeset = set([ item.strip() for item in mustbe.split("+") ])
    ## number var to add
    pmustbe = len(mustbeset)
    ## setting up the set "add" which contains the possible variable to add
    explanatory = set([ item.strip() for item in formula[1].split("+") ]) - mustbeset
    ## results
    aic_list = []
    bic_list = []
    ssr_list = []
    var_list = []
    nb_var = []
    R2_list = []
    R2adj_list = []
    ## main loop
    for k in range(len(explanatory),0,-1):
        #Looping over all possible combinations of k elt
        for combo in itertools.combinations(explanatory, k):
            # add variables and make the formula
            formula = "{} ~ {} + {}".format(response,
                                   ' + '.join(combo),
                                            mustbe)
            # calculate the criterion
            current = smf.ols(formula, data=data).fit()
            # results
            ssr_list.append(current.ssr)
            bic_list.append(current.bic)
            aic_list.append(current.aic)
            var_list.append(combo)
            nb_var.append(k+pmustbe)
            R2_list.append(current.rsquared)
            R2adj_list.append(current.rsquared_adj)
    df = pd.DataFrame({'nb_var': nb_var, 'var_added': var_list, 'AIC': aic_list, 'BIC': bic_list, "SSR": ssr_list, 'R2': R2_list, 'R2adj': R2adj_list})
    return df


def olsstep(data, start: str, lower: str, upper: str, direction="both", crit="aic", verbose = False):
    """Backward selection for linear model (with smf and formula)
    Parameters:
    -----------
    data (pandas DataFrame): DataFrame with all possible predictors
            and response
    start (string): a string giving the starting model
            (ie the starting point)
    lower (string): a string giving the lower model
            (ie the minimal model allowed)
    upper (string): a string giving the upper model
            (ie the maximal model allowed)
    direction (string): direction "both", "forward" or "backward"
    crit (string): "aic"/"AIC" or "bic"/"BIC"
    verbose (boolean): if True verbose print

    Returns:
    --------
    model: an "optimal" linear model fitted with statsmodels
           with an intercept and
           selected by forward/backward or both algorithm with crit criterion
    """
    # direction
    if not (direction == "both" or direction == "forward" or
            direction == "backward"):
        raise ValueError(
            "direction error (should be both, forward or backward)")
    # criterion
    if not (crit == "aic" or crit == "AIC" or crit == "bic" or crit == "BIC"):
        raise ValueError("criterion error (should be AIC/aic or BIC/bic)")
    # starting point
    formula_start = start.split("~")
    response = formula_start[0].strip()
    # lower
    formula_lower = lower.split("~")
    if formula_lower[0].strip() != response:
        raise ValueError("not the same response in lower and start formula")
    # upper
    formula_upper = upper.split("~")
    if formula_upper[0].strip() != response:
        raise ValueError("not the same response in start and upper formula")
    # explanatory variables for the 3 models
    start_explanatory = set([item.strip() for item in
                             formula_start[1].split("+")]) - set(['1'])
    upper_explanatory = set([item.strip() for item in
                            formula_upper[1].split("+")]) - set(['1'])
    lower_explanatory = set([item.strip() for item in
                             formula_lower[1].split("+")]) - set(['1'])
    # declarations (not mandatory but useful for bounding variables-> no checkers complaints)
    add = set()
    remove = set()
    score = 0.0
    current_score = 0.0
    critdisp = ""
    # setting up the set "add" which contains the possible variable to add
    if direction == "both" or direction == "forward":
        add = upper_explanatory - start_explanatory
    # setting up the set "remove" which contains the possible
    # variable to remove
    if direction == "both" or direction == "backward":
        remove = start_explanatory - lower_explanatory
    # current point
    selected = start_explanatory
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(list(selected)))
    if crit == "aic" or crit == "AIC":
        critdisp = "AIC"
        current_score = smf.ols(formula, data).fit().aic
    elif crit == "bic" or crit == "BIC":
        critdisp = "BIC"
        current_score = smf.ols(formula, data).fit().bic
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Starting", selected))
    # main loop
    while True:
        scores_with_candidates = []
        if direction == "both" or direction == "backward":
            for candidate in remove:
                tobetested = selected - set([candidate])
                formula = "{} ~ {} + 1".format(response,
                                               ' + '.join(list(tobetested)))
                if crit == "aic" or crit == "AIC":
                    score = smf.ols(formula, data).fit().aic
                elif crit == "bic" or crit == "BIC":
                    score = smf.ols(formula, data).fit().bic
                if verbose:
                    print((score, "-", candidate))
                scores_with_candidates.append((score, "-", candidate))
        if direction == "both" or direction == "forward":
            for candidate in add:
                tobetested = selected | set([candidate])
                formula = "{} ~ {} + 1".format(response,
                                               ' + '.join(list(tobetested)))
                if crit == "aic" or crit == "AIC":
                    score = smf.ols(formula, data).fit().aic
                elif crit == "bic" or crit == "BIC":
                    score = smf.ols(formula, data).fit().bic
                if verbose:
                    print((score, "+", candidate))
                scores_with_candidates.append((score, "+", candidate))
        scores_with_candidates.sort()
        best_new_score, dircur, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            if dircur == "+":
                add = add - set([best_candidate])
                selected = selected | set([best_candidate])
                if direction == "both":
                    remove = remove | set([best_candidate])
            else:
                remove = remove - set([best_candidate])
                selected = selected - set([best_candidate])
                if direction == "both":
                    add = add | set([best_candidate])
            current_score = best_new_score
            if verbose:
                print("----------------------------------------------")
                print((current_score, "New Current", selected))
        else:
            break
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Final", selected))
    formula = "{} ~ {} + 1".format(response, ' + '.join(list(selected)))
    model = smf.ols(formula, data).fit()
    return model


if __name__ == "__main__":
    print("choixolsstats.py is being run directly ??")
