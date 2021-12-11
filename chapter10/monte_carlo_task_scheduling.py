import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def main():
    n = 10000
    total_time = []
    t = np.empty(shape=(n, 6))

    # column 1: optimistic time (a)
    # colume 2: more likely time (c)
    # column 3: pessimistic time (b)
    task_times = [[3, 5, 8],  # Task 1
                  [2, 4, 7],  # Task 2
                  [3, 5, 9],  # Task 3
                  [4, 6, 10],  # Task 4
                  [3, 5, 9],  # Task 5
                  [2, 6, 8]]  # Task 6

    lh = []
    for i in range(6):
        # Lh = (like - opt) / (pess - opt)
        # Lh = (c - a) / (b - a)
        lh.append(
            (task_times[i][1] - task_times[i][0]) /
            (task_times[i][2] - task_times[i][0])
        )

    for p in range(n):
        for i in range(6):
            t_rand = random.random()

            if t_rand < lh[i]:
                t[p][i] = task_times[i][0] + \
                          np.sqrt(
                              t_rand *
                              (task_times[i][2] - task_times[i][0]) *
                              (task_times[i][1] - task_times[i][0])
                          )
            else:
                t[p][i] = task_times[i][2] - \
                          np.sqrt(
                              (1 - t_rand) *
                              (task_times[i][2] - task_times[i][0]) *
                              (task_times[i][2] - task_times[i][1])
                          )

        total_time.append(
            t[p][0] +
            np.maximum(t[p][1], t[p][2]) +
            np.maximum(t[p][3], t[p][4]) + t[p][5]
        )

    data = pd.DataFrame(t, columns=['Task 1', 'Task 2', 'Task 3',
                                    'Task 4', 'Task 5', 'Task 6', ])
    pd.set_option('display.max_columns', None)
    print(data.describe())

    hist = data.hist(bins=10)
    plt.show()

    print("Minimum completion time = {}".format(np.amin(total_time)))
    print("Mean completion time = {}".format(np.mean(total_time)))
    print("Maxiumum completion time = {}".format(np.amax(total_time)))


if __name__ == '__main__':
    main()
