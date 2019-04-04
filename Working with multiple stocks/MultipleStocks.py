'''Build a dataframe in pandas'''
import pandas as pd
import matplotlib.pyplot as plt

def test_run_test_all():
    #Define date range
    start_date = '2010-01-22'
    end_date = '2010-01-26'
    dates = pd.date_range(start_date, end_date)
    #print dates

    # Create an empty dataframe
    df1 = pd.DataFrame(index=dates)
    #print df1

    #Read SPY into temporary dataframe
    dfSPY = pd.read_csv("SPY.csv",
                        index_col="Date",
                        parse_dates=True,
                        usecols=['Date', 'Adj Close'],
                        na_values=['nan'])

    #rename Adj Close column to SPY to prevent column name clash
    dfSPY = dfSPY.rename(columns={'Adj Close':'SPY'})

    #print len(dfSPY)
    #print dfSPY.head()

    #print dfSPY['2016-07-15']['Adj Close']
    #print dfSPY['2010-01-22']['Adj Close']

    #Join the two dataframes
    #df1 = df1.join(dfSPY)
    #df1 = df1.dropna()
    df1 = df1.join(dfSPY, how='inner')

    symbols = ['GOOG', 'IBM', 'GLD']


    for symbol in symbols:
        df_temp = pd.read_csv("{}.csv".format(symbol),
                        index_col="Date",
                        parse_dates=True,
                        usecols=['Date', 'Adj Close'],
                        na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close':symbol})
        df1=df1.join(df_temp)
    
    print df1


def test_run():
    # Define a date range
    #dates = pd.date_range('2010-01-22', '2010-01-26')
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']
    
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
    df = normalize_data(df)

    
    plot_data(df)
    
    #print df

def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe"""
    #df = df/df.ix['2010-01-04']
    df = df / df.ix[0,:]
    return df

def plot_data(df, title="Stock prices"):
    '''Plot stock prices'''
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    #return os.path.join(base_dir, "{}.csv".format(str(symbol)))
    return "{}.csv".format(str(symbol))


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
    test_run()
