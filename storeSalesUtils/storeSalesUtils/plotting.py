import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from storeSalesUtils.prophetPipeline import RMSLE


def plot_autocorrelations(series, lags):
    """
    It takes a time series and a list of lags, and plots the autocorrelation function and partial
    autocorrelation function for the time series
    
    Args:
      series: The series to plot the autocorrelations for.
      lags: The number of lags to include in the plot.
    """
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_title('ACF Plot')
    ax[0].grid(linewidth=0.5)
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_title('Partial ACF Plot')
    ax[1].grid(linewidth=0.5)
    plt.show()

    return

def plot_joint_plot(verif, x='yhat', y='y', title=None): 
    """
    It takes a dataframe, a column name for the x-axis, a column name for the y-axis, and a title, and
    it returns a jointplot with a linear regression line and a correlation coefficient. 
    
    Args:
      verif: the dataframe containing the verification data
      x: the name of the column in the dataframe that contains the model's estimates. Defaults to yhat
      y: the name of the column in the dataframe that contains the observations. Defaults to y
      title: The title of the plot.
    """
    
    g = sns.jointplot(x=x, y=y, data = verif, kind="reg", color="0.1")
    
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    
    if title is not None: 
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]

    ax.text(0.05, 0.8,
            "R = {:+4.2f}\nRMSLE = {:4.1f}".format(verif.loc[:,[x,y]].corr().iloc[0,1],
                                                 RMSLE(verif[y], verif[x])),
            fontsize=16, transform=ax.transAxes)

    ax.set_xlabel("model's estimates", fontsize=15)
    
    ax.set_ylabel("observations", fontsize=15)
    
    ax.grid(ls=':')

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.grid(ls=':')
    
    return g