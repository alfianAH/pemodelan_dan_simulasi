import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

    summary = asn_data_scaled.describe()
    summary = summary.transpose()
    print(summary)





if __name__ == '__main__':
    main()