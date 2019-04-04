"""Minimize a portfolio for Sharpe Ratio, using SciPy"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as spo
import pandas.io.data as web
import datetime

def get_historical_closes(ticker, start_date, end_Date):

    #get the data for the tickers.
    p = web.DataReader(ticker, "yahoo", start_date, end_Date)

    #Convert to DataFrame and select only Adj Close
    d = p.to_frame()['Adj Close'].reset_index()

    #Rename the columns
    d.rename(columns={'minor': 'Ticker', 'Adj Close': 'Close'}, inplace=True)

    #Pivot each ticker to a column
    pivoted = d.pivot(index='Date', columns='Ticker')

    pivoted.columns = pivoted.columns.droplevel(0)

    return pivoted

def calculate_daily_returns(closes):
    returns = (closes/closes.shift(1)) - 1
    returns = returns.fillna(0)

    print "My Daily Returns: ", returns[:5], " Their ", np.log(closes/closes.shift(1))[:5]
    return returns

def calculate_annual_returns(daily_returns):
    grouped = daily_returns.groupby(
        lambda date: date.year).sum()

    print "My Grouped:", grouped, " Their ", np.exp(daily_returns.groupby(lambda date: date.year).sum())-1
    return grouped

def sharpe_ratio(returns, weights = None, risk_free_rate = 0.015):

    if weights is None:
        weights = get_equal_weights(returns)

    #Multiply returns with weights
    returns_weighted = returns.dot(weights)
    
    #Mean
    mean_val = returns_weighted.mean()
    std_val = returns_weighted.std()

    print "STD Weighted Mine: ", std_val, " By Covariance: ", np.sqrt(calc_portfolio_var(returns, weights)), " 3rd Way ", np.sum(weights * returns,axis=1).std()
    #print "Mean Mine: ", mean_val, " Theirs: ", returns.mean().dot(weights)
    
    sharpe_ratio_val =  (mean_val - risk_free_rate) / std_val
    return sharpe_ratio_val

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    return -sharpe_ratio(returns, weights, risk_free_rate)

def optimize_portfolio(returns, risk_free_rate):
    """
    Performs optimization by minimizing negative sharpe ratio
    """

    #initial with equal weights
    initial_weights = get_equal_weights(returns)


    #Optimize the weights.
    optimized_weights = spo.minimize(negative_sharpe_ratio,
                          initial_weights, args=(returns, risk_free_rate),
                          method='SLSQP',
                          bounds = [(0., 1.) for _ in range(len(initial_weights))],
                          constraints = { 'type': 'eq',
                                'fun': lambda w: np.array(sum(abs(initial_weights)) - 1.), # sum to 1 constraint
                                'jac': lambda w: np.array([1. for _ in range(len(initial_weights))])}, # gradient
                          options={'disp': True})
                          

    
    #Optimize the weights.
    """
    optimized_weights = spo.fmin(negative_sharpe_ratio,
                          initial_weights, args=(returns, risk_free_rate))
                          """

    return optimized_weights.x


#########################Alternative Optimization############################

def calculate_daily_returns2(closes):
    return np.log(closes/closes.shift(1))

def calculate_annual_returns2(daily_returns):
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

def sharpe_ratio2(returns, weights = None, risk_free_rate = 0.015):
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
    return -sharpe_ratio2(returns, weights2, risk_free_rate)

def optimize_portfolio2(returns, risk_free_rate):
    """ 
    Performs the optimization
    """
    # start with equal weights
    w0 = np.ones(returns.columns.size-1, 
                 dtype=float) * 1.0 / returns.columns.size
    # minimize the negative sharpe value
    w1 = spo.fmin(negative_sharpe_ratio_n_minus_1_stock, 
                    w0, args=(returns, risk_free_rate))

    final_w = sp.append(w1, 1 - np.sum(w1))

    return final_w

#########################End Alternative Optimization############################    

def get_equal_weights(returns):
    n = returns.columns.size
    weights = np.ones(n)/n
    return weights


if __name__ == "__main__":
    """
    Few Caveat's
    If Daily returns are calculated as np.log(closes/closes.shift(1)) instead of (closes/closes.shift(1)) - 1
    and
    If Annual Returns are calculated with np.exp as opposed to without
    and
    Standard Deviation is calculated by sqrt'ing the covariance instead of direct std()
    Then, the optimization seems to give so much better results.
    """
    closes = get_historical_closes(['MSFT', 'AAPL', 'KO'], '2010-01-01', '2014-12-31')
    risk_free_rate = 0.0003
    
    initial_weights = get_equal_weights(closes)
    print "Initial Weights ", initial_weights, " Weights.T ", initial_weights.T
    
    returns = calculate_daily_returns(closes)
    #returns = calculate_daily_returns2(closes)
    #print returns[:5]
    
    annual_returns = calculate_annual_returns(returns)
    #annual_returns = calculate_annual_returns2(returns)
    print "Initial Sharpe Ratio ", sharpe_ratio(annual_returns, initial_weights, risk_free_rate)

    optimized_weights = optimize_portfolio(annual_returns, risk_free_rate)
    print "optimized_weights ", optimized_weights

    

    optimized_weights2 = optimize_portfolio2(annual_returns, risk_free_rate)
    print "optimized_weights2 ", optimized_weights2
    
    print "Optimized Sharpe Ratio", sharpe_ratio(annual_returns, optimized_weights, risk_free_rate)
    print "Alternative Sharpe Ratio", sharpe_ratio2(annual_returns, optimized_weights2, risk_free_rate)

    #Initial Portfolio
    initial_portfolio = closes.dot(initial_weights)

    #Final Portfolio
    final_portfolio = closes.dot(optimized_weights)

    #Alternative Portfolio
    alternative_portfolio = closes.dot(optimized_weights2)

    #Static Weights Portfolio
    static_weights = [ 0.76353353,  0.2103234 ,  0.02614307]
    static_portfolio = closes.dot(static_weights)
    

    print initial_portfolio[:5]
    print final_portfolio[:5]
    
    ax = initial_portfolio.plot(title="Sharpe Ratio Optimized Portfolio", label='Initial Portfolio')
    final_portfolio.plot(label="Optimized Portfolio", ax = ax)
    alternative_portfolio.plot(label="Alternative Portfolio", ax = ax)
    static_portfolio.plot(label="Static Portfolio", ax = ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
    
    

    




    
