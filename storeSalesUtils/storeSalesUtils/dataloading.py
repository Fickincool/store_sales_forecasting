import pandas as pd
import os

dateparse = lambda x: pd.to_datetime(x, errors='coerce', format='%Y-%m-%d')

DATA_FOLDER = '/home/jeronimo/Desktop/Freshflow_techChallenge/store_sales_forecasting/data/'

def load_train():
    """
    It reads the train.csv file, and parses the date column as a date
    :return: A dataframe
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'train.csv'), parse_dates=['date'], date_parser=dateparse)
    return df

def load_sample_submission():
    """
    It reads the sample_submission.csv file, and parses the date column as a date
    :return: A dataframe
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'sample_submission.csv'), parse_dates=['date'], date_parser=dateparse)
    return df

def load_transactions():
    """
    It reads the transactions.csv file, parses the date column as a date, and returns the resulting
    dataframe
    :return: A dataframe with the transactions data
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'transactions.csv'), parse_dates=['date'], date_parser=dateparse)
    return df

def load_stores():
    """
    It reads the stores.csv file and returns a dataframe
    :return: A dataframe with the columns: date, store_nbr, city, state, type, cluster
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'stores.csv'))
    return df

def load_holidays():
    """
    It reads the holidays_events.csv file, parses the date column as a date, and returns the resulting
    dataframe
    :return: A dataframe with the following columns:
        date: The date of the holiday
        type: The type of holiday
        locale: The locale of the holiday
        locale_name: The name of the locale
        description: A description of the holiday
        transferred: Whether the holiday was transferred from another day
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'holidays_events.csv'), parse_dates=['date'], date_parser=dateparse)
    return df

def load_test():
    """
    It reads the test.csv file, and parses the date column as a date
    :return: A dataframe with the test data
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'test.csv'), parse_dates=['date'], date_parser=dateparse)
    return df

def load_oil():
    """
    It reads the oil.csv file, parses the date column as a date, and returns the resulting dataframe
    :return: A dataframe with the date and the price of oil.
    """
    df = pd.read_csv(os.path.join(DATA_FOLDER,'oil.csv'), parse_dates=['date'], date_parser=dateparse)
    return df