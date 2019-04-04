
import pandas as pd
import numpy as np
import pandas.io.data as web
from datetime import datetime
import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats
import matplotlib.mlab as mlab
# plotting

import matplotlib.pyplot as plt

def get_historical_closes(ticker, start_date, end_date):
    # get the data for the tickers.  This will be a panel
    p = web.DataReader(ticker, "yahoo", start_date, end_date)    
    # convert the panel to a DataFrame and selection only Adj Close
    # while making all index levels columns
    d = p.to_frame()['Adj Close'].reset_index()
    # rename the columns
    d.rename(columns={'minor': 'Ticker', 
                      'Adj Close': 'Close'}, inplace=True)
    # pivot each ticker to a column
    pivoted = d.pivot(index='Date', columns='Ticker')
    # and drop the one level on the columns
    pivoted.columns = pivoted.columns.droplevel(0)
    return pivoted

def calc_daily_returns(closes):
    return np.log(closes/closes.shift(1))

def calc_annual_returns(daily_returns):
    grouped = np.exp(daily_returns.groupby(
        lambda date: date.year).sum())-1
    return grouped

def calc_portfolio_var(returns, weights=None):
    if weights is None: 
        weights = np.ones(returns.columns.size) / \
        returns.columns.size
    sigma = np.cov(returns.T,ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var

def sharpe_ratio(returns, weights = None, risk_free_rate = 0.015):
    n = returns.columns.size
    if weights is None: weights = np.ones(n)/n
    # get the portfolio variance
    var = calc_portfolio_var(returns, weights)
    # and the means of the stocks in the portfolio
    means = returns.mean()
    # and return the sharpe ratio
    return (means.dot(weights) - risk_free_rate)/np.sqrt(var)


def negative_sharpe_ratio_n_minus_1_stock(weights, 
                                          returns, 
                                          risk_free_rate):
    """
    Given n-1 weights, return a negative sharpe ratio
    """
    weights2 = sp.append(weights, 1-np.sum(weights))
    return -sharpe_ratio(returns, weights2, risk_free_rate)

def optimize_portfolio(returns, risk_free_rate):
    """ 
    Performs the optimization
    """
    # start with equal weights
    w0 = np.ones(returns.columns.size-1, 
                 dtype=float) * 1.0 / returns.columns.size
    # minimize the negative sharpe value
    w1 = scopt.fmin(negative_sharpe_ratio_n_minus_1_stock, 
                    w0, args=(returns, risk_free_rate))
    # build final set of weights
    final_w = sp.append(w1, 1 - np.sum(w1))
    # and calculate the final, optimized, sharpe ratio
    final_sharpe = sharpe_ratio(returns, final_w, risk_free_rate)
    return (final_w, final_sharpe)


def get_equal_weights(returns):
    n = returns.columns.size
    weights = np.ones(n)/n
    return weights

if __name__ == "__main__":
    closes = get_historical_closes(['MSFT', 'AAPL', 'KO'], '2010-01-01', '2014-12-31')
    
    #closes = get_historical_closes(['GOOG', 'AAPL', 'SPY', 'AMZN'], '2013-01-01', '2015-09-16')
    
    initial_weights = get_equal_weights(closes)
        
    daily_returns = calc_daily_returns(closes)
    annual_returns = calc_annual_returns(daily_returns)
    result = optimize_portfolio(annual_returns, 0.0003)

    #Initial Portfolio
    initial_portfolio = closes.dot(initial_weights)    

    #Final Portfolio
    final_portfolio = closes.dot(result[0])

    
    ax = initial_portfolio.plot(title="Sharpe Ratio Optimized Portfolio", label='Initial Portfolio')
    final_portfolio.plot(label="Optimized Portfolio", ax = ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
    

    print result[0]
