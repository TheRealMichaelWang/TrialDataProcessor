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

trials = []
with open(input_file + ".csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for line in reader:
        assert int(line[0]) == len(trials)
        trials.append((
            1 if line[7] == "correct" else 0
        ))

x = np.arange(0, len(trials), 1)
y1 = single_var_model.predict(x.reshape(-1, 1))

#x2 = np.array([[i ** (j + 1) for j in range(5)] for i in range(100)])
poly_x = np.array([x, x**2, x**3, x**4, x**5]).T
y2 = quadratic_model.predict(poly_x)

#x3 = np.arange(1, len(moving_avgs) + 1)
#plt.plot(x3, np.array(moving_avgs), label='moving_avgs')

#y4 = np.array(trials)
#plt.scatter(x, y4, label='raw data', marker='s', color='red')


#plt.plot(x, y1, label='single_var_model')
plt.plot(x, y2, label='quadratic_model')

plt.xlabel("Trial No.")
plt.legend()
plt.show()