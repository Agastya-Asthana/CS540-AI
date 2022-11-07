import sys
import csv
import numpy as np
from matplotlib import pyplot as plt


def LinearRegression(csv_dict):
    X = np.empty((len(csv_dict) - 1, 2), dtype=np.int64)
    for i in range(len(csv_dict) - 1):
        X[i][0] = 1
        X[i][1] = int(csv_dict[i + 1][0])
    Y = np.empty((len(csv_dict) - 1, 1), dtype=np.int64)
    for i in range(len(csv_dict) - 1):
        Y[i] = int(csv_dict[i + 1][1])
    Z = np.dot(np.transpose(X), X)

    return X, Y, Z


if __name__ == "__main__":
    filename = sys.argv[1]
    csv_dict = []
    x = []
    y = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            csv_dict.append(row)
            try:
                x.append(int(row[0]))
                y.append(int(row[1]))
            except:
                continue

    plt.plot(x, y)
    plt.xlabel(csv_dict[0][0])
    plt.ylabel(csv_dict[0][1])
    plt.savefig("plot.png")

    X, Y, Z = LinearRegression(csv_dict)
    print("Q3a:")
    print(X)

    print("Q3b:")
    print(Y)

    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    PI = np.dot(I, np.transpose(X))
    print("Q3e:")
    print(PI)

    hat_beta = np.dot(PI, Y)
    print("Q3f:")
    print(hat_beta)

    x_test = 2021
    y_test = hat_beta[0] + (hat_beta[1] * x_test)
    print("Q4: " + str(y_test))

    print("Q5a: <")
    print("Q5b: Negative sign means there is an inverse relationship between the variables")

    new_x_test = (-1 * hat_beta[0]) / hat_beta[1]
    print("Q6a: " + str(new_x_test))
    print("Q6b: in the year 1812 the # of toys produced will be zero and this makes sense since the number of toys "
          "produced each year has been going down")
