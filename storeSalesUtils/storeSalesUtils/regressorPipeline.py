from storeSalesUtils.prophetPipeline import median_filter, RMSLE

import pandas as pd
import numpy as np
import os
from glob import glob

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

import warnings

def make_shifted(df, shift_col, minlag, maxlag):
    """
    It takes a dataframe, a column name, and a range of lags, and returns a new dataframe with the
    original column and the lagged columns
    
    Args:
      df: the dataframe you want to shift
      shift_col: the column to shift
      minlag: the minimum lag to use
      maxlag: the maximum number of days to shift the data back
    
    Returns:
      A new dataframe with the shifted columns
    """
    new_df = df.copy()
    for i in range(minlag, maxlag+1):
        new_df.loc[:, '%s_%i'%(shift_col, i)] = new_df[shift_col].shift(i)
        
    return new_df

def backshift_df(df, shift_vars, minlag, maxlag):
    """
    It takes a dataframe, a list of variables to shift, and a minimum and maximum lag, and returns a
    dataframe with the shifted variables.
    
    Args:
      df: the dataframe you want to shift
      shift_vars: the variables to shift
      minlag: the minimum lag to create
      maxlag: the maximum number of lags to create
    
    Returns:
      A dataframe with the original data and the shifted data.
    """
    
    for var in shift_vars:
        df = make_shifted(df, var, minlag, maxlag)
    
    return df

def add_date_features(df, date_col):
    """
    It takes a dataframe and a date column as input, and returns a dataframe with the following date
    features: year, month, day, dayofyear, dayofweek, weekofyear, is_weekend
    
    Args:
      df: the dataframe
      date_col: The name of the column that contains the date
    
    Returns:
      A dataframe with the date features added.
    """
    # Date Features
    df_dateFeatures = df.copy()
    df_dateFeatures['year'] = df_dateFeatures[date_col].dt.year
    df_dateFeatures['month'] = df_dateFeatures[date_col].dt.month
    df_dateFeatures['day'] = df_dateFeatures[date_col].dt.day
    df_dateFeatures['dayofyear'] = df_dateFeatures[date_col].dt.dayofyear
    df_dateFeatures['dayofweek'] = df_dateFeatures[date_col].dt.dayofweek
    df_dateFeatures['weekofyear'] = df_dateFeatures[date_col].dt.weekofyear
    df_dateFeatures['is_weekend'] = np.where(df_dateFeatures.dayofweek.isin([5, 6]), 1, 0)
    
    return df_dateFeatures

def make_model_df(df, group_col, backshift_cols, minlag, maxlag, category_cols):
    """
    > We take a dataframe, group it by a column, and then apply a function to each group
    
    Args:
      df: the dataframe you want to backshift
      group_col: the column that you want to group by. For example, if you want to make a model for each
    store, you would use 'store_id'
      backshift_cols: the columns to backshift
      minlag: the minimum number of days to look back
      maxlag: the maximum number of days to look back
      category_cols: a list of columns that should be converted to categorical variables
    
    Returns:
      A dataframe with the backshifted columns
    """
    new_df = df.groupby(group_col, as_index=False).apply(
        lambda x: backshift_df(x, backshift_cols, minlag, maxlag)
        ).reset_index()

    new_df.drop('index', 1, inplace=True, errors='ignore')
    new_df = add_date_features(new_df, 'date')
    
    if category_cols:
        for col in category_cols:
            new_df[col] = new_df[col].astype('category')
        
    return new_df

# > The class takes in a model, a dataframe with daily sales, and a bunch of parameters to construct
# the autoregressive features. It then trains the model on the training data, and computes the
# validation error
class regressorPipeline():
    def __init__(
        self, model, storeFamily_daily, group_col, backshift_cols, ar_minlags, ar_maxlags, category_cols,
        filter
        ):
        """
        It takes in a dataframe of daily sales, and a model, and returns a dataframe of predictions for
        the validation period
        
        Args:
          model: the model to use, e.g., random forest regressor
          storeFamily_daily: the dataframe with the daily sales information
          group_col: the column that will be used to group the data. In this case, it's the store
        number.
          backshift_cols: the columns that we want to use to predict the future.
          ar_minlags: the minimum number of days to look back
          ar_maxlags: the maximum number of days to look back for autoregressive features
          category_cols: columns that are categorical
          filter: whether to apply a median filter to the sales data
        
        Returns:
          The class is being returned.
        """
        if filter:
            storeFamily_daily['sales'] = median_filter(storeFamily_daily, 'sales', 28)

        self.dep_var = 'sales'
        self.date_col = 'date'
        self.drop_cols = [self.dep_var, 'family', 'split', self.date_col, 'id', 'store_nbr']

        # some daily values are missing, we interpolate missing information
        storeFamily_daily = storeFamily_daily.set_index(self.date_col).resample('d').first()
        storeFamily_daily[self.dep_var] = storeFamily_daily[self.dep_var].interpolate('linear')

        # when we predict the future, no information is available.
        # I need to do this to set up the proper autorregresive values
        # and prevent information leakage when running the validation
        train_missingSales = storeFamily_daily.copy()
        train_missingSales.loc[storeFamily_daily.split=='validation', self.dep_var] = np.nan

        model_df = make_model_df(
            train_missingSales, group_col, backshift_cols, 
            ar_minlags, ar_maxlags, category_cols
            )
        model_df = model_df.fillna(method='bfill')

        self.model = model
        self.model_df = model_df
        self.storeFamily_daily = storeFamily_daily
        self.group_col = group_col 
        self.backshift_cols = backshift_cols
        self.ar_minlags = ar_minlags
        self.ar_maxlags = ar_maxlags
        self.category_cols = category_cols

        self.train_regressor()

        self.val_df = self.compute_val_df()

        self.RMSLE_validation = RMSLE(self.val_df[self.dep_var], self.val_df['prediction'])

        return

    def train_regressor(self):
        """
        > The function takes a dataframe of features and a target variable, and uses Lasso regression to
        select the features that are most important for predicting the target variable
        
        Returns:
          The model is being returned.
        """
        
        y_train = self.model_df[self.model_df.split=='train'][self.dep_var]
        X_train = self.model_df[self.model_df.split=='train'].drop(self.drop_cols, 1)

        # lasso doesn't allow nan values, so we fill all nans for feature selection
        X_aux = X_train.fillna(method='ffill').fillna(method='bfill')

        lasso = Lasso(alpha=1).fit(X_aux, y_train)
        featureSelector = SelectFromModel(lasso, prefit=True)
        self.featureSelector = featureSelector

        self.model_features = X_train.columns[featureSelector.get_support()]

        if len(self.model_features)==0:
            store_nbr = self.storeFamily_daily.store_nbr.unique()[0]
            family = self.storeFamily_daily.family.unique()[0]
            self.model_features = X_train.columns
            warnings.warn(f'Ignoring feature selector step because number of selected features for Store {store_nbr}, {family} was zero.')

        # if len(self.model_features)<3:
        #     store_nbr = self.storeFamily_daily.store_nbr.unique()[0]
        #     family = self.storeFamily_daily.family.unique()[0]
        #     warnings.warn(f'Number of selected features for Store {store_nbr}, {family} was {len(self.model_features)}.')

        # use only L-1 reduced feature set
        X_train = X_train[self.model_features]

        self.model.fit(X_train, y_train)

        return

    def compute_val_df(self):
        """
        The function takes the model_df dataframe, which contains the training and validation data, and
        computes the validation predictions. 
        
        The function then updates the validation dataframe with the actual values of the dependent
        variable, and the validation predictions. 
        
        The function then computes the RMSLE and signed error for the validation data. 
        
        The function returns the validation dataframe.
        
        Returns:
          A dataframe with the validation data, the predicted values, and the RMSLE and signed error.
        """

        X_val = self.model_df[self.model_df.split=='validation'].drop(self.drop_cols, 1)
        X_val = X_val[self.model_features]

        # Compute validation Prediction
        val_pred = self.model.predict(X_val)

        # validation data
        val_df = self.model_df[self.model_df.split=='validation']
        # update dependent variable with actual values to compute errors
        val_df[self.dep_var] = self.storeFamily_daily[self.storeFamily_daily.split=='validation'][self.dep_var].values

        val_df['prediction'] = val_pred
        val_df['prediction'] = val_df.prediction.map(lambda x: max(0, x))
        val_df['RMSLE'] = RMSLE(val_df[self.dep_var], val_df['prediction'])
        val_df['signed_error'] = val_df[self.dep_var] - val_df['prediction']

        return val_df

    def predict_test(self, storeFamily_daily_test):
        """
        We fit the model to the full dataset, then we use the model to predict the test set
        
        Args:
          storeFamily_daily_test: the test dataframe
        
        Returns:
          The model_test dataframe with the predicted values.
        """
        model_df = make_model_df(
            self.storeFamily_daily, self.group_col, self.backshift_cols,
            self.ar_minlags, self.ar_maxlags, self.category_cols
            )

        y = model_df[self.dep_var]
        X = model_df.drop(self.drop_cols, 1)
        X = X[self.model_features]

        # fit model to full dataset
        self.model.fit(X, y)

        # we need this to construct the AR values
        model_test = pd.concat([self.storeFamily_daily, storeFamily_daily_test])
        model_test['ly_sales'] = model_test.sales.shift(52*7)

        model_test = make_model_df(
            model_test, self.group_col, self.backshift_cols,
            self.ar_minlags, self.ar_maxlags, self.category_cols
            )

        model_test['pred'] = self.model.predict(model_test[X.columns])
        model_test['pred'] = model_test.pred.map(lambda x: max(0, x))

        return model_test

    def plot_feature_importances(self, figsize=(16, 6)):
        """
        It returns a plot of the
        feature importances.
        
        Args:
          figsize: The size of the figure to plot.
        
        Returns:
          The feature importances are being returned.
        """
        feature_importances = pd.DataFrame(self.model.feature_importances_, index=self.model_features, columns=['feat_importance'])
        ax = feature_importances.sort_values('feat_importance').plot.barh(figsize=figsize)

        return ax