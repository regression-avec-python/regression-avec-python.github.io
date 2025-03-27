import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd


def bestglm(data, upper, mustbe="1", family=sm.families.Binomial()):
    """Backward selection for generalized linear model (with formula)
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    upper : a string giving the upper model (containing mustbe variables)
    mustbe : a string giving the variables that must be included (separated by +)
    family : family argument passed to glm
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
    deviance_list = []
    var_list = []
    nb_var = []
    ## main loop
    for k in range(len(explanatory),0,-1):
        #Looping over all possible combinations of k elt
        for combo in itertools.combinations(explanatory, k):
            # add variables and make the formula
            formula = "{} ~ {} + {}".format(response,
                                   ' + '.join(combo),
                                            mustbe)
            # calculate the criterion
            current = smf.glm(formula, data=data, family=family).fit()
            deviance_list.append(current.deviance)
            # results
            aic_list.append(current.aic)
            bic_list.append(current.bic_llf)
            var_list.append(combo)
            nb_var.append(k+pmustbe)
    df = pd.DataFrame({'nb_var': nb_var, 'var_added': var_list, 'AIC': aic_list, 'BIC': bic_list, 'deviance': deviance_list})
    return df


if __name__ == "__main__":
    print("choixglmstats.py is being run directly ??")
