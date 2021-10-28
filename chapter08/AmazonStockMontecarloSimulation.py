import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def main():
    # Membaca file CSV dan membuat dataframe
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

    # Melihat isi perubahan persentase
    # print(amzn_data_pct_change.head(5))

    # Plot amzn_log_return
    plt.figure(figsize=(10, 5))  # Menentukan size figure
    plt.plot(amzn_log_return)  # Plotting dataframe
    plt.show()  # Memperlihatkan plotting

    mean_log_returns = np.array(amzn_log_return.mean())
    var_log_returns = np.array(amzn_log_return.var())
    stdev_log_returns = np.array(amzn_log_return.std())

    drift = mean_log_returns - (0.5 * var_log_returns)
    print("Drift = ", drift)

    num_intervals = 2518
    iterations = 20
    np.random.seed(7)

    sb_motion = norm.ppf(np.random.rand(num_intervals, iterations))
    daily_returns = np.exp(drift + stdev_log_returns * sb_motion)
    start_stock_prices = amzn_df.iloc[0]
    stock_price = np.zeros_like(daily_returns)
    stock_price[0] = start_stock_prices

    for t in range(1, num_intervals):
        stock_price[t] = stock_price[t - 1] * daily_returns[t]

    plt.figure(figsize=(10, 5))
    plt.plot(stock_price)

    amzn_trend = np.array(amzn_df.iloc[:, 0:1])

    plt.plot(amzn_trend, 'k*')
    plt.show()


if __name__ == '__main__':
    main()
