import pandas as pd
import matplotlib.pyplot as plt
import os

def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
    
    # Get stock data
    df = get_data(symbols, dates)

    #Get specific rows
    #slicedRows = df.ix['2010-01-25':'2010-01-26', ['GOOG', 'GLD']]
    #print slicedRows
    #print df.ix['2010-01-25':'2010-01-26']
    #print df['GOOG']
    #print df[['IBM', 'GLD']]

    #print df.ix['2010-01-25':'2010-01-26', ['SPY', 'IBM']]
    #print df.ix['2010-01-04']

    #normalize all stocks to start from the same 1.0 value on the first date.
    #df = normalize_data(df)

    
    plot_data(df)
    #print df.mean()
    #print df.median()
    #Standard Deviation
    print df.std()
    
    #print df

def rolling_statistics():
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)

    #plot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    #Compute rolling mean using a 20-day window
    rm_SPY = pd.rolling_mean(df['SPY'], window=20)

    #Add rolling mean to the same plot
    rm_SPY.plot(label='Rolling mean', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

def bollinger_bands():
    #dates = pd.date_range('2012-01-01', '2012-12-31')
    dates = pd.date_range('2015-01-01', '2016-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)

    #Compute rolling mean using a 20-day window
    rm_SPY = pd.rolling_mean(df['SPY'], window=20)
    
    #Compute rolling standard deviation using a 20-day window
    rstd_SPY = pd.rolling_std(df['SPY'], window=20)

    #Compute Upper and Lower Bollinger Bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    #print upper_band.tail()
    #print lower_band.tail()


    #plot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY Bollinger Bands", label='SPY')
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='Upper Band', ax=ax)
    lower_band.plot(label='Lower Band', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

def daily_returns():
    dates = pd.date_range('2012-07-01', '2012-07-31')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    #plot_data(df)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily Returns")

    plt.show()

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

def get_bollinger_bands(rolling_mean, rolling_std):
    #twoband_rolling_std = 2 * rolling_std
    
    #upper_band = rolling_mean.add(twoband_rolling_std, fill_value=0)
    #lower_band = rolling_mean.subtract(twoband_rolling_std, fill_value=0)

    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    return upper_band, lower_band
    

def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe"""
    #df = df/df.ix['2010-01-04']
    df = df / df.ix[0,:]
    return df

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

if __name__ == "__main__":
    #test_run()
    #rolling_statistics()
    #bollinger_bands()
    daily_returns()
