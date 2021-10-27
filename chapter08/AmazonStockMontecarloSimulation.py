import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def main():
    amzn_df = pd.read_csv('AMZN.csv', header=0,
                          usecols=['Date', 'Close'], parse_dates=True,
                          index_col='Date')

    print(amzn_df.info())  # Melihat detail dataframe
    print(amzn_df.head())  # Melihat 5 data dari atas
    print(amzn_df.tail())  # Melihat 5 data dari bawah
    print(amzn_df.describe())  # Melihat statistik dataframe

    # Plot amzn_df
    plt.figure(figsize=(10, 5))  # Menentukan size figure
    plt.plot(amzn_df)  # Plotting dataframe
    plt.show()  # Memperlihatkan plotting

    amzn_data_pct_change = amzn_df.pct_change()  # Perubahan persentase
    amzn_log_return = np.log(1 + amzn_data_pct_change)  # Skala logaritmik

    # print("amzn_pct_change")
    # print(amzn_data_pct_change.tail(10))
    #
    # print("amzn_log")
    # print(amzn_log_return.tail(10))

    # Plot amzn_data_pct_change
    # plt.figure(figsize=(10, 5))
    # plt.plot(amzn_data_pct_change)
    # plt.title('amzn_data_pct_change')

    # Plot amzn_log_return
    plt.figure(figsize=(10, 5))  # Menentukan size figure
    plt.plot(amzn_log_return)  # Plotting dataframe
    plt.show()  # Memperlihatkan plotting


if __name__ == '__main__':
    main()
