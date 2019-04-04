import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab
import numpy as np

#from ..util import get_data, plot_data



def compute_daily_returns(df):

    df_daily = df.copy()

    #1st Way

    #compute the daily returns
    df_daily = (df_daily / df_daily.shift(1)) - 1
    
    #OR

    #2nd Way
    #df_daily = (df[1:] / df[:-1].values) - 1
    
    df_daily.ix[0,:] = 0 # set daily return for row 0 to 0
    #OR
    #df_daily = df_daily.fillna(0) #Set the first (and all other) row's NaN values with 0 so it starts at 0.


    return df_daily

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    '''Plot stock prices'''
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def symbol_to_path(symbol, base_dir="../data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))
    #return "{}.csv".format(str(symbol))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        path=symbol_to_path(symbol)
        df_temp = pd.read_csv(path, index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp, how='inner')

    df = df.sort_index()
    return df

def draw_histogram():
    #Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    #plot_data(df)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    #Plot the histogram.
    daily_returns.hist(bins=20)
    #pylab.hist(daily_returns, normed=1)
    #pylab.show()
    #plt.show()

    #Get mean and SD
    mean = daily_returns['SPY'].mean()
    print "mean=", mean
    std = daily_returns['SPY'].std()
    print "std=", std

    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    #Compute Kurtosis
    print daily_returns.kurtosis()

def multiple_histograms():
    #Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    plot_data(df)
    
    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")


    #Plot the histogram.
    #daily_returns.hist(bins=20)
    #plt.show()

    #Compute and plot both histograms on the same chart
    daily_returns['SPY'].hist(bins=20, label="SPY")
    daily_returns['XOM'].hist(bins=20, label="XOM")
    plt.legend(loc='upper right')
    plt.show()

def scatter_plot():
    #Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    #plot_data(df)
    
    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    #Scatter plot SPY vs XOM
    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    #XOM has higher beta than GLD. So XOM is more reactive to market (SPY) than GLD.
    print "beta_XOM=", beta_XOM 
    print "alpha_XOM", alpha_XOM
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color = 'r')
    plt.show()

    #Scatter plot SPY vs GLD
    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    print "beta_GLD=", beta_GLD
    #GLD has higher alpha than XOM. Hence GLD performed better.
    print "alpha_GLD", alpha_GLD 
    plt.plot(daily_returns['SPY'], beta_GLD * daily_returns['SPY'] + alpha_GLD, '-', color = 'r')
    plt.show()

    #Calculate correlation coefficient
    print daily_returns.corr(method='pearson')
    

if __name__ == "__main__":
    #draw_histogram()
    #multiple_histograms()
    scatter_plot()
