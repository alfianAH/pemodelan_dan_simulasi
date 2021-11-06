import pandas as pd


def main():
    # Beri nama kolom untuk kolom di file dat
    asn_names = ['Frequency', 'AngleAttack', 'ChordLength', 'FSVelox', 'SSDT', 'SSP']
    # Baca file dat
    asn_data = pd.read_csv('airfoil_self_noise.dat',
                           delim_whitespace=True, names=asn_names)

    print(asn_data.head())  # Melihat 5 data teratas
    print(asn_data.info())  # Melihat info asn_data

    basic_stats = asn_data.describe()  # Melihat detail masing-masing kolom
    basic_stats = basic_stats.transpose()  # Transpose kolom dan baris



if __name__ == '__main__':
    main()
