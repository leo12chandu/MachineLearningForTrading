import pandas as pd
import matplotlib.pyplot as plt
import os

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
