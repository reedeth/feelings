import style
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
# from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# from sklearn.grid_search import GridSearchCV
# import random


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


def remove_zeros(x, y):
    """Given x and y coordinates, if the y is a zero,
    remove that entry from the list."""
    # get indices of every zero
    indices = [num for num in x if y[num] == 0]
    for num in reversed(indices):
        del(x[num])
        del(y[num])
    return x, y


def convert_to_np(x, y, z):
    """given three arrays, format them in a way
    that the polynomial regression will take."""
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    return x, y, z


def prep_data(text):
    """Given a text, prep the data for it."""
    y = text.sentiment_values
    # mess with the data for testing
    # for i in range(0,30) and range(400,500):
    #     y[i] = random.uniform(0.5,1.0)
    x = list(range(0, len(y)))
    x, y = remove_zeros(x, y)
    x_test = x
    x, y, x_test = convert_to_np(x, y, x_test)
    return x, y, x_test


def plot_it(x, y, x_test, y_test, num_polynomial, fig, ax):
    """Take all the stuff and plot it."""
    fig.gca()
    ax.set_xticks(np.arange(0, x.max(), 100))
    ax.set_yticks(np.arange(-1, 1, 0.25))
    plt.plot(x, y, 'b.')
    plt.plot(x_test, y_test, label=num_polynomial)
    plt.grid(b=True, linestyle='--')
    plt.legend()


def main():
    fn = 'corpus/test/sabotage_clean.txt'
    text = style.Text(fn)
    x, y, x_test = prep_data(text)
    fig, ax = plt.subplots()
    for i in [0, 1]:
        y_test = PolynomialRegression(i).fit(x, y).predict(x_test)
        num = i
        plot_it(x, y, x_test, y_test, num, fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
