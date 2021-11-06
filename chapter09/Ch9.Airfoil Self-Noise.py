import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    # Beri nama kolom untuk kolom di file dat
    asn_names = ['Frequency', 'AngleAttack', 'ChordLength', 'FSVelox', 'SSDT', 'SSP']
    # Baca file dat
    asn_data = pd.read_csv('airfoil_self_noise.dat',
                           delim_whitespace=True, names=asn_names)

    # print(asn_data.head())  # Melihat 5 data teratas
    # print(asn_data.info())  # Melihat info asn_data

    basic_stats = asn_data.describe()  # Melihat detail masing-masing kolom
    basic_stats = basic_stats.transpose()  # Transpose kolom dan baris
    # print(basic_stats)

    # Scaling
    scaler_object = MinMaxScaler()
    asn_data_scaled = scaler_object.fit_transform(asn_data)
    asn_data_scaled = pd.DataFrame(asn_data_scaled, columns=asn_names)

    summary = asn_data_scaled.describe()  # Melihat dataframe yang sudah diskalakan
    summary = summary.transpose()
    print(summary)

    # boxplot = asn_data_scaled.boxplot(column=asn_names)  # Box plot
    # for name in asn_names:
    #     plt.plot(asn_data_scaled[name])
    #     plt.show()

    # Menghitung korelasi
    cor_asn_data = asn_data_scaled.corr(method='pearson')
    with pd.option_context('display.max_rows', None, 'display.max_columns', cor_asn_data.shape[1]):
        print(cor_asn_data)

    # Plot korelogram
    plt.matshow(cor_asn_data)
    plt.xticks(range(len(cor_asn_data.columns)), cor_asn_data.columns)
    plt.yticks(range(len(cor_asn_data.columns)), cor_asn_data.columns)
    plt.colorbar()
    # plt.show()

    # Pilih kolom fitur sebagai x
    x = asn_data_scaled.drop('SSP', axis=1)
    # Pilih kolom target (SSP) sebagai y
    y = asn_data_scaled['SSP']

    # Membagi data training (70%) dan testing (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
    # print('X train shape = ', x_train.shape)
    # print('X test shape = ', x_test.shape)
    # print('Y train shape = ', y_train.shape)
    # print('Y test shape = ', y_test.shape)

    linier_model = LinearRegression()  # Membuat model regresi linier
    linier_model.fit(x_train, y_train)  # Training data

    y_pred_lm = linier_model.predict(x_test)  # Prediksi x_test
    mse_lm = mean_squared_error(y_test, y_pred_lm)  # Menghitung MSE
    print('MSE of Linear Regression model: {}'.format(mse_lm))


if __name__ == '__main__':
    main()
