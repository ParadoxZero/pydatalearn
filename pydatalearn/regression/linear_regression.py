import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:

    def __calculate_theta_normal_equation(self):
        inp = np.matrix(self.X)
        y = np.matrix(self.Y)
        theta = ((inp.transpose() * inp) ** (-1)) * inp.transpose() * y  # The Normal Equation => (X'X)^-1 * X' * Y
        return theta.tolist()

    def __init__(self, input_matrix, output_matrix):
        self.input = input_matrix
        self.output = output_matrix
        self.X = np.matrix(input_matrix)
        self.Y = np.matrix(output_matrix)
        self.theta = self.__calculate_theta_normal_equation()

    def getValueForRow(self,row):
        val = 0
        for i in range(len(self.theta)):
            val += row[i]*self.theta[i][0]
        return val

    def plot_single_variable(self):
        x = [x for (i,x) in self.input]
        y_cal = []
        for i in self.input:
            y_cal.append(self.getValueForRow(i))
        plt.scatter(x, self.output, label="Points")
        plt.plot(x, y_cal, label="Regression")
        plt.legend()
        plt.xlabel("age")
        plt.ylabel("Height")
        plt.title("Regression modeling the height of boys with age")
        plt.show()
