
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")

# Regression Metrics

def metrics_reg(y, baseline):
    """
    send in y_true, y_pred & returns RMSE, R2 --Misty's
    """
    rmse = mean_squared_error(y, baseline, squared=False)
    r2 = r2_score(y, baseline)
    return rmse, r2

def plot_residuals(y, yhat):
    """This function calculates and plots residuals (actual - predicted)
    ---
    Format: residuals = function()
    ---
    Then add to df as a new column.
    """
    import matplotlib.pyplot as plt

    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title(' Residual vs Home Value Plot')
    plt.show()

    return residuals

def regression_errors(y, yhat):
    """This function returns regression error metrics.
    ---
    Format: SSE, ESS, TSS, MSE, RMSE = function()
    ---
    MSE: SSE / len(df) \\ Average Squared Errors
    SSE: (yhat - target) ^2 \\ Sum of Squared Errors
    RMSE: sqrt(MSE) \\ Root Mean Squared Error
    ESS: sum((yhat - y.mean())**2) \\ Explained Sum of Squares (for r2 value)
    TSS: ESS + SSE \\ Total Sum of Squares (for r2 value)
    """

    from sklearn.metrics import mean_squared_error

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE ** .5
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE

    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """This function calculates baseline mean errors from regression errors function
    ---
    Format: MSE, SSE, RMSE = function()    
    """
    import numpy as np 
    from sklearn.metrics import mean_squared_error

    baseline = np.repeat(y.mean(),len(y))

    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE ** .5

    return MSE, SSE, RMSE

def better_than_baseline(SSE, SSE_baseline):
    """This function returns true if model performed better than baseline, otherwise false"""
    if SSE < SSE_baseline:
        return True
    else:
        return False
    
# -------------------------------------------------------------------------

# Feature Engineering

def select_kbest(X_train_scaled, y_train, n_features):
    """This function returns the top features to model based on K Best selection."""
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_regression

    # MAKE the thing
    kbest = SelectKBest(f_regression, k=n_features)

    # FIT the thing
    kbest.fit(X_train_scaled, y_train)
    
    # USE the thing and put into DF
    X_train_KBtransformed = pd.DataFrame(kbest.transform(X_train_scaled),\
                                     columns = X_train_scaled.columns[kbest.get_support()],\
                                     index=X_train_scaled.index)
    
    # return column names
    print(f"These are the top {n_features} columns selected from Select K Best model:")
    return X_train_KBtransformed.columns.tolist()

def rfe(X_train_scaled, y_train, n_features):
    """This function returns the top features to model based on RFE selection."""
    import pandas as pd    
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    lr = LinearRegression()

    # make the thing
    rfe = RFE(lr, n_features_to_select = n_features)

    # fit the thing
    rfe.fit(X_train_scaled, y_train)
    
    # use the thing
    X_train_RFEtransformed = pd.DataFrame(rfe.transform(X_train_scaled),index=X_train_scaled.index,
    columns = X_train_scaled.columns[rfe.support_])
    
     # return column names
    print(f"These are the top {n_features} columns selected from RFE model:")
    return X_train_RFEtransformed.columns.tolist()

def baseline_test(X_train_scaled, y_train, X_test_scaled, y_test):
    """This function built to establish baseline on train and then test on test,
    returns a dataframe of results along with SSE_baseline and SSE.
    ---
    Format: SSE, SSE_baseline, metrics_df = function()
    """

    # baseline
    baseline = y_train.mean().round(2)
    baseline

    # make array
    baseline_array = np.repeat(baseline, len(X_train_scaled))
    baseline_array

    # evaluate
    rmse, r2 = metrics_reg(y_train, baseline_array)
    rmse, r2

    # put baseline prediction into metrics df
    metrics_df = pd.DataFrame(data=[
        {
            'Model':'Baseline',
            'RMSE':rmse,
            'R2':r2
        }
        
    ])
    metrics_df

    # USE POLY-2 ON TEST
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled
    X_test_degree = pf.transform(X_test_scaled)

    #make it
    pr = LinearRegression()

    #fit it
    pr.fit(X_train_degree, y_train)

    #use it on train and test
    pred_lr1 = pr.predict(X_train_degree)
    pred_pr = pr.predict(X_test_degree)

    #validate
    rmse, r2 = metrics_reg(y_test, pred_pr)
    rmse, r2

    # MSE
    MSE = mean_squared_error(y_train, pred_lr1)
    SSE = MSE * len(X_train_scaled)

    # SSE_baseline
    MSE_baseline = mean_squared_error(y_train, baseline_array)
    SSE_baseline = MSE_baseline * len(X_train_scaled)

    #add to my metrics df
    metrics_df.loc[1] = ['POLY_2', rmse, r2]
    metrics_df

    return SSE, SSE_baseline, metrics_df


def to_csv(train, test):
# create a dataframe with parcelid and house worth
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    X_train = train[[
        'phone_service',
        'multiple_lines',
        'monthly_charges',
        'total_charges',
        'contract_type_One year',
        'contract_type_Two year',
        'internet_service_type_Fiber optic',
        'internet_service_type_None']]
    X_test = test[[
        'phone_service',
        'multiple_lines',
        'monthly_charges',
        'total_charges',
        'contract_type_One year',
        'contract_type_Two year',
        'internet_service_type_Fiber optic',
        'internet_service_type_None']]
    y_train = train['churn']
    y_test = test['churn']
    rf = RandomForestClassifier(random_state = 123,max_depth = 6)
    rf.fit(X_test, y_test)
    prediction_df = pd.DataFrame({'customer_id': test.customer_id,
                                  'probability_of_churn': rf.predict_proba(X_test)[:,1],
                                  'prediction_of_churn': rf.predict(X_test)})
    prediction_df
    # export the dataframe to csv
    prediction_df.to_csv('predictions.csv')
