import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from prophet import Prophet

def median_filter(df, varname = None, window=24, std=3): 
    """
    A simple median filter, removes (i.e. replace by np.nan) observations that exceed N (default = 3) 
    tandard deviation from the median over window of length P (default = 24) centered around 
    each observation.
    Parameters
    ----------
    df : pandas.DataFrame
        The pandas.DataFrame containing the column to filter.
    varname : string
        Column to filter in the pandas.DataFrame. No default. 
    window : integer 
        Size of the window around each observation for the calculation 
        of the median and std. Default is 24 (time-steps).
    std : integer 
        Threshold for the number of std around the median to replace 
        by `np.nan`. Default is 3 (greater / less or equal).
    Returns
    -------
    dfc : pandas.Dataframe
        A copy of the pandas.DataFrame `df` with the new, filtered column `varname`
    """
    
    dfc = df.loc[:,[varname]]
    
    dfc['median']= dfc[varname].rolling(window, center=True).median()
    
    dfc['std'] = dfc[varname].rolling(window, center=True).std()
    
    dfc.loc[dfc.loc[:,varname] >= dfc['median']+std*dfc['std'], varname] = np.nan
    
    dfc.loc[dfc.loc[:,varname] <= dfc['median']-std*dfc['std'], varname] = np.nan
    
    return dfc.loc[:, varname]

def grangers_causation_matrix(data, variables, maxlag, test='ssr_chi2test', verbose=False):    
    """
    The function takes in a dataframe, a list of variables, and a maxlag value. It then creates a
    dataframe of zeros with the same dimensions as the number of variables. It then loops through the
    columns and rows, and for each pair of variables, it runs a granger causality test with a maxlag
    value. It then stores the p-values in a list, and then stores the minimum p-value in the dataframe

    Note:
    Read matrix as testing for var_x granger-causes var_y, reject the null (no granger-causation) when 
    values are less than 0.05.
    
    Args:
      data: The dataframe containing the time series variables
      variables: The list of variables to be tested for Granger Causality.
      maxlag: The number of lags to use in the test.
      test: The type of test to use. Can be either 'ssr_chi2test', 'lrtest', 'ssr_ftest',
    'ssr_chi2test', 'ssr_ftest', 'lrtest', 'ssr_chi2test', 'ssr_. Defaults to ssr_chi2test
      verbose: If True, prints out the results of the tests for each combination of variables. Defaults
    to False
    
    Returns:
      A dataframe with the p-values for each pair of variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def setup_prophetData(data, y_col, ds_col):
    """
    This function takes in a dataframe, a column name for the y-values, and a column name for the
    dates, and returns a dataframe with the columns renamed to 'y' and 'ds' for use with Prophet
    
    Args:
      data: the dataframe that contains the data
      y_col: the column name of the data you want to predict
      ds_col: the column name of the date column
    
    Returns:
      A dataframe with the columns 'ds' and 'y'
    """
    if y_col is not None:
        prophetData = data[[ds_col, y_col]].rename({y_col:'y', ds_col:'ds'}, axis=1)
    else:
        prophetData = data[[ds_col]].rename({ds_col:'ds'}, axis=1)
        
    return prophetData

def add_prophetRegressor(data, regressor, ds_col): 
    """
    This function takes in a dataframe, a regressor dataframe, and a column name. It merges the
    dataframe and the regressor dataframe on the column name.
    
    Args:
      data: the dataframe that you want to add the regressor to
      regressor: the dataframe with the regressor data
      ds_col: the column name of the date column in the dataframe
    """
    
    data_with_regressors = pd.merge(data, regressor, on=ds_col)
    
    return data_with_regressors

def RMSLE(y, yhat):
    "Compute Root Mean Squared Log Error"
    metric = (np.log(1+yhat) - np.log(1+y))**2
    return round(np.sqrt(metric.mean()), 3)

def parse_oil_prophet(oil):
    """
    It takes in a dataframe, and returns a dataframe with the same columns, but with the date column as
    the index, and with the null values filled in. 
    
    The function is a bit more complicated than that, but that's the gist of it. 
    
    Args:
      oil: the dataframe containing the oil prices
    
    Returns:
      A dataframe with the date and the oil price
    """
    oil_prophet = setup_prophetData(oil, 'dcoilwtico', 'date').rename({'y':'dcoilwtico'}, axis=1)
    oil_prophet = oil_prophet.set_index('ds').resample('d')
    # fill null values first with closest values
    # bfill is used only for the first value of the series, which is missing
    oil_prophet = oil_prophet.fillna(method='nearest').fillna(method='bfill').reset_index()
    
    return oil_prophet 

def parse_holidays(holidays, city, state):
    """
    It takes the holidays dataframe, the city and state and returns a dataframe with the dates and
    descriptions of the holidays that are relevant to the city and state
    
    Args:
      holidays: a pandas dataframe containing the holidays
      city: The city for which to create the model, as present in the data
      state: The state in which the city is located.
    
    Returns:
      A dataframe with the dates and descriptions of the holidays
    """

    # find the holidays within the country, city and state
    relevant_holidays = holidays[holidays.locale_name.isin(['Ecuador', city, state])]
    relevant_holidays = relevant_holidays[['date', 'description']]
    relevant_holidays.columns = ['ds', 'holiday']
    earthquake = {'ds':pd.to_datetime('2016-04-16'), 'holiday':'Earthquake'}
    relevant_holidays = relevant_holidays.append(earthquake, ignore_index=True)
    relevant_holidays.sort_values('ds', inplace=True)

    # further cleaning should be done here, "traslados" holidays might be problematic but lets just proceed like this
    return relevant_holidays

def train_prophet(prophet_df, holidays=None, regressor_list=None):
    """
    It takes a dataframe with a 'ds' and 'y' column, and a list of regressors, and returns a fitted
    Prophet model and a forecast dataframe
    
    Args:
      prophet_df: the dataframe that Prophet will use to fit the model.
      holidays: a dataframe containing a list of holidays
      regressor_list: a list of columns in the dataframe that you want to use as regressors.
    
    Returns:
      The model and the forecast
    """

    l = len(prophet_df)
    
    train_df = prophet_df[prophet_df.split=='train'].drop('split', 1)
    
    # first fit a model to the train data
    m = Prophet(growth='linear', holidays=holidays, yearly_seasonality=True, weekly_seasonality=True)
    if regressor_list is not None:
        for name in regressor_list:
            m.add_regressor(name)

    # given our EDA, we know there is a monthly seasonality as well,
    # a lot of the sales occur the first and last couple of days of the month
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(train_df)
    
    # assess fit on the validation set afterwards
    forecast = m.predict(prophet_df)
    forecast = pd.merge(forecast, prophet_df[['ds', 'y', 'split']], on='ds')

    # sanity check
    assert len(forecast) == l

    return m, forecast

def visualize_forecasts(fcst, title=None):
    """
    It takes a forecast dataframe and plots the actual values (y) and the forecasted values (yhat) along
    with the upper and lower bounds of the forecast (yhat_upper and yhat_lower)
    
    Args:
      fcst: the forecast dataframe
      title: The title of the plot.
    
    Returns:
      The plot of forecasted and observed values
    """
    f, ax = plt.subplots()

    ax.plot(fcst.ds, fcst.y, '--k', markersize=1, lw=0.5, label='y')
    ax.plot(fcst.ds, fcst.yhat, '-', markersize=1, color='steelblue', lw=0.5, label='yhat')

    ax.fill_between(fcst.ds, fcst.yhat_lower, fcst.yhat_upper, color='steelblue', alpha=0.2)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend()
    plt.grid(linewidth=0.2)

    return ax

class prophetPipeline():
    
    "Pipeline that takes in a dataframe, and returns a dataframe with the predictions of the Prophet model."

    def __init__(self, storeFamily_daily, oil_prophet, holidays, filter):
        """
        We use the Granger Causality test to determine whether the oil price is a significant predictor
        of sales. If it is, we include it in the model
        
        Args:
          storeFamily_daily: a dataframe with the daily sales data for the store family
          oil_prophet: a dataframe with the following columns:
          holidays: a dataframe containing the dates of holidays
        
        Returns:
          The class is returning the following:
            - regressor_list
            - prophet_df
            - oil_prophet
            - trained_model
            - forecast
        """

        if filter:
            storeFamily_daily['sales'] = median_filter(storeFamily_daily, 'sales', 28)
            
        prophet_df = setup_prophetData(storeFamily_daily, 'sales', 'date')
        prophet_df = add_prophetRegressor(prophet_df, oil_prophet, 'ds')
        prophet_df = pd.merge(
            prophet_df, storeFamily_daily[['date', 'split']], left_on='ds',right_on='date'
            ).drop('date', 1)

        if prophet_df.y.sum()==0:
            forecast = prophet_df.copy()
            forecast['yhat'] = 0
            trained_model = None
            regressor_list = None

        else:
            gcm = grangers_causation_matrix(prophet_df.dropna(), ['y', 'dcoilwtico'], maxlag=14, test='ssr_chi2test')
            if gcm.loc['y_y', 'dcoilwtico_x']<0.05:
                print(
                    'Reject non-causality hypothesis for dcoilwtico on sales. Minimum p-value: %.03f' %gcm.loc['y_y', 'dcoilwtico_x'])
                regressor_list = ['dcoilwtico']
            else:
                regressor_list = []

            trained_model, forecast = train_prophet(prophet_df, holidays, regressor_list)

            # prediction must be non-negative
            forecast['yhat'] = forecast.yhat.map(lambda x: max(0, x))

        self.regressor_list = regressor_list
        self.prophet_df = prophet_df
        self.oil_prophet = oil_prophet
        self.trained_model = trained_model
        self.forecast = forecast

        # compute error metrics for validation set
        y = forecast[forecast.split=='validation'].y
        yhat = forecast[forecast.split=='validation'].yhat
        self.RMSLE_validation = RMSLE(y, yhat)

        return

    def prophet_predictTest(self, storeFamily_daily_test):
        """
        This function takes in a dataframe of the test data, and returns a dataframe of the predicted
        values
        
        Args:
          storeFamily_daily_test: the test dataframe
        
        Returns:
          The forecasted values for the test data.
        """
        prophet_test = setup_prophetData(storeFamily_daily_test, None, 'date')

        if self.trained_model:
            if 'dcoilwtico' in self.regressor_list:
                prophet_test = add_prophetRegressor(prophet_test, self.oil_prophet, 'ds')
            else:
                pass

            test_forecast = self.trained_model.predict(prophet_test)
        
        else:
            test_forecast = prophet_test.copy()
            test_forecast['yhat'] = 0

        return test_forecast