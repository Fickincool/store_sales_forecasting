#!/usr/bin/env python3
from storeSalesUtils.dataloading import load_stores, load_train, load_holidays, load_test, load_oil, DATA_FOLDER
from storeSalesUtils.plotting import plot_joint_plot, plot_autocorrelations
from storeSalesUtils.prophetPipeline import (prophetPipeline, parse_holidays, parse_oil_prophet,
visualize_forecasts, RMSLE)

import sys

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np
from numpy.linalg import svd
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

import logging
logging_folder = '/home/jeronimo/Desktop/Freshflow_techChallenge/store_sales_forecasting/scripts/logs/prophet_dataPooling/'
logging.basicConfig(
    filename=os.path.join(logging_folder, "logging.log"),
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %I:%M:%S%p",
)

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
logging.getLogger('fbprophet').setLevel(logging.WARNING) 

from tqdm.auto import tqdm

import warnings
warnings.simplefilter("ignore", SettingWithCopyWarning)
warnings.simplefilter("ignore", FutureWarning)

from hts import HTSRegressor


def load_data():
    train = load_train()
    stores = load_stores()

    test = load_test()

    # only use two years of data available
    maxDate = train.date.max()
    minDate = maxDate - pd.DateOffset(years=2)
    print('Using data within the dates: ', maxDate, minDate)
    train = train[train.date.between(minDate, maxDate)]

    # simulate test set scenario using the last 15 days of data for validation
    val_dates = maxDate - pd.DateOffset(15)
    train['split'] = np.where(train.date>=val_dates, 'validation', 'train')

    return train, test, stores

def get_low_rank_df(df, thresh=0.8):
    u, s, vh = svd(df, full_matrices=True)
    cumulative_comps = s.cumsum()/s.sum()

    # select the number of components which account for the threshold of the cumulative singular values
    lowRank_components = np.where(cumulative_comps>thresh)[0][0]
    low_rank_df = np.dot(u[:, :lowRank_components] * s[:lowRank_components], vh[:lowRank_components, :])

    low_rank_df = pd.DataFrame(low_rank_df, index=df.index, columns=df.columns)

    # standardize data for SVD and future rescaling
    low_rank_df = (low_rank_df - low_rank_df.mean())/low_rank_df.std()

    # we want to ignore streaks of consecutive zeros
    zero_streak_condition = np.where(df.rolling(7).mean() == 0, np.nan, 1)
    # we propagate NaNs for Prophet model
    aux = pd.DataFrame(zero_streak_condition, index=low_rank_df.index, columns=low_rank_df.columns)
    low_rank_df = low_rank_df * aux

    return low_rank_df

def get_date_store_pivot_table(train_df, family):

    pivot_train = train_df[(train_df.family==family)]
    pivot_train = pivot_train.pivot_table(values='sales', index='date', columns='store_nbr')
    pivot_train.columns = pivot_train.columns.astype(str)

    return pivot_train

def predict_one(train_lowRank, steps_ahead):
    stores = train_lowRank.columns.to_list()
    hierarchy = {'total': stores}

    # add totals for the grouped TS model
    train_lowRank['total'] = train_lowRank.sum(axis=1)

    reg = HTSRegressor(model='prophet', revision_method='OLS', 
                       **{'yearly_seasonality':False, 'weekly_seasonality':True})
    reg = reg.fit(df=train_lowRank, nodes=hierarchy, show_warnings=False)

    preds = reg.predict(steps_ahead=steps_ahead)

    return reg, preds

def validate_one(prediction_matrix, ground_truth_df):
    
    val_assessment = prediction_matrix.melt(ignore_index=False).reset_index()
    val_assessment.columns = ['date', 'store_nbr', 'pred']
    val_assessment['store_nbr'] = val_assessment.store_nbr.astype(int)
    
    val_assessment = val_assessment.merge(ground_truth_df, on=['date', 'store_nbr'])
    
    error = RMSLE(val_assessment.pred, val_assessment.sales)

    return val_assessment, error

def adjust_predictions(preds, pivot_train, n_days_back=None):
    if isinstance(n_days_back, int):
        threshold_date = pivot_train.index.max() - pd.DateOffset(days=n_days_back)
        recent_train = pivot_train[pivot_train.index >= threshold_date]
    elif threshold_date is None:
        recent_train = pivot_train.copy()
    # go back to actual scale of values
    adjusted_preds = (preds.drop('total', axis=1)*recent_train.std())+recent_train.mean()
    # we want only non-negative predictions
    adjusted_preds = adjusted_preds.where(adjusted_preds>=0, 0)
    
    return adjusted_preds



def main():

    train, test, stores = load_data()

    test_predictions = []

    for family in train.family.unique():
        print(family)
        pivot_train = get_date_store_pivot_table(train, family=family)
        # if almost all entries are zero, drop those stores
        all_zeros_condition = (pivot_train==0).sum() >= 0.9*len(pivot_train)
        all_zero_stores = all_zeros_condition[all_zeros_condition].index.tolist()
        pivot_train = pivot_train.drop(all_zero_stores, axis=1)
        # use rolling mean as target. Drop first rows which are NaN
        window = 14
        pivot_train_smoothed = pivot_train.rolling(window).mean()[window-1::]

        train_lowRank = get_low_rank_df(pivot_train_smoothed)

        reg, preds = predict_one(train_lowRank, steps_ahead=16)
        
        # adjust using the most recent 15 days to compute the means and stds
        adjusted_preds = adjust_predictions(preds, pivot_train_smoothed, n_days_back=16)

        # put back the stores with all zeros, this is our best guess if no data is available
        for col in all_zero_stores:
            adjusted_preds[col] = 0

        # only use dates within the validation range
        final_preds = adjusted_preds.melt(ignore_index=False).reset_index()
        final_preds.columns = ['date', 'store_nbr', 'sales']
        final_preds['family'] = family

        final_preds['store_nbr'] = final_preds.store_nbr.astype(int)

        final_preds = final_preds.merge(test, on=['date', 'store_nbr', 'family'])

        test_predictions.append(final_preds)


    test_predictions = pd.concat(test_predictions)
    test_predictions['sales'] = test_predictions.sales.round()

    file = os.path.join(DATA_FOLDER, 'predictions/prophet_dataPooling_weeklySeasonal.csv')
    test_predictions[['id', 'sales']].to_csv(path_or_buf=file, index=False)

    return

if __name__ == '__main__':
    main()