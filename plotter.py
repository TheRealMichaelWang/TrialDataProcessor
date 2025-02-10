import pickle
import numpy as np
import matplotlib.pylab as plt
import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python plotter.py <input name>")
    quit()

input_file = sys.argv[1]

with open("quadratic_" + input_file + ".pkl", 'rb') as in_file:
    quadratic_model = pickle.load(in_file)
with open("single_var_" + input_file + ".pkl", 'rb') as in_file:
    single_var_model = pickle.load(in_file)

with open("rolling_avgs_"+input_file+".csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    moving_avgs = [float(line[1]) for line in reader]

x = np.arange(0, 100)
y1 = single_var_model.predict(x.reshape(-1, 1))

#x2 = np.array([[i ** (j + 1) for j in range(5)] for i in range(100)])
y2 = quadratic_model.predict(np.array([x, x**2, x**3, x**4, x**5]).reshape(-1, 5))

plt.plot(x, y1, label='single_var_model')
plt.plot(x, y2, label='quadratic_model')

plt.show()