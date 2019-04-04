import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    df = pd.read_csv("AAPLStock.csv")
    print "Printing Head"
    print df.head() #top 5 rows.

    print "Printing Tail"
    print df.tail()

    print "Rows between 10 and 20"
    print df[10:21]

    print "max close: ", df['Close'].max()
    print "mean volume: ", df['Volume'].mean()

    df['Adj Close'].plot()
    plt.show()

    df[['Close', 'Adj Close']].plot()
    plt.show()

if __name__ == "__main__":
    test_run()
