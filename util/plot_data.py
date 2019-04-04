import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    '''Plot stock prices'''
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
