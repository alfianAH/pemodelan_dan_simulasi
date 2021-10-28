import math
import random
import numpy as np
import matplotlib.pyplot as plt


def main():
    n = 10000
    m = 0

    x_circle = []
    y_circle = []
    x_square = []
    y_square = []

    for p in range(n):
        x = random.random()
        y = random.random()

        # Rumus lingkaran
        # x^2 + y^2 = r^2
        # Jika <= 1, maka di dalam lingkaran
        if x**2 + y**2 <= 1:
            m += 1
            x_circle.append(x)
            y_circle.append(y)
        else:  # Selain itu, di luar lingkaran dan di dalam kotak
            x_square.append(x)
            y_square.append(y)

    pi = 4*m/n
    print('n = {}, m = {}, pi = {:.2f}'.format(n, m, pi))

    x_lin = np.linspace(0, 1)
    y_lin = []
    for x in x_lin:
        y_lin.append(math.sqrt(1 - x**2))

    plt.axis('equal')  # Membuat ukuran plot sama panjang
    plt.grid(which='major')  # Membuat tick major
    plt.plot(x_lin, y_lin, color='red', linewidth='4')  # Plot garis lingkaran
    plt.scatter(x_circle, y_circle, color='yellow', marker='.')  # Plot titik dalam lingkaran
    plt.scatter(x_square, y_square, color='blue', marker='.')  # Plot titik luar lingkaran
    plt.title('Mone Carlo method for Pi estimation')  # Membuat judul
    plt.show()  # Menampilkan grafik


if __name__ == '__main__':
    main()
