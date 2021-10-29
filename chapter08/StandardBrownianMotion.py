import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(4)
    n = 1000
    SQN = 1/np.math.sqrt(n)
    ZValues = np.random.randn(n)

    Yk = 0

    SBMotion = list()

    for k in range(n):
        Yk = Yk + SQN*ZValues[k]
        SBMotion.append(Yk)

    plt.plot(SBMotion)
    plt.xlabel("index")
    plt.ylabel("Gerak Brown")
    plt.show()


if __name__ == '__main__':
    main()
