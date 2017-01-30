"""
Author: Sidhin S Thomas
Email: Sidhin.thomas@gmail.com

Simple linear regression

The data set used to test is :
http://openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex2materials/ex2Data.zip

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import partial

from matplotlib import pyplot as plt

from pydatalearn.regression import cost_function
from pydatalearn.regression import linear_regression


def main():
    input_matrix = [[1, float(x)] for x in open("DataSet/hieght_age/ex2x.dat").read().splitlines()]
    output = [[float(x)] for x in open("DataSet/hieght_age/ex2y.dat").read().splitlines()]
    h_theta = partial(cost_function,input_matrix,output)
    linear = linear_regression.LinearRegression(input_matrix,output,False,0.07,[0,0])
    print(linear.theta)
    x = [x for (i, x) in linear.input]
    y_cal = []
    for i in linear.input:
        y_cal.append(linear.getValueForInput(i))
    plt.scatter(x, linear.output, label="Points")
    plt.plot(x, y_cal, label="Regression")
    plt.legend()
    plt.xlabel("age")
    plt.ylabel("Height")
    plt.title("Regression modeling the height of boys with age")
    plt.show()

if __name__ == '__main__':
    main()
