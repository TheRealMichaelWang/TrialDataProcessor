import csv
import sys
import os
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from typing import List

if len(sys.argv) < 2:
    print("Usage: python main.py <csv file> <output name>")
    quit()

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "processed_" + os.path.splitext(sys.argv[1])[0]

def simplify_path(path: str) -> str:
    if '\\' in path:
        path = path.replace('\\', '/')
    if not '/' in path:
        return path
    return os.path.splitext(os.path.basename(path))[0]

#open csv file
new_lines = []
trials = []
categories = { }

with open(input_file) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')

    category_line = next(reader)
    for i in range(len(category_line)):
        categories[category_line[i]] = i

    for line in reader:
        for i in range(len(line)):
            line[i] = simplify_path(line[i])

        assert int(line[categories["trial_num"]]) == len(trials)
        try:
            timestamp = datetime.strptime(line[categories["timesss"]], "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            timestamp = datetime.strptime(line[categories["timesss"]], "%M:%S.%f")
        trials.append((
            1 if line[categories["response_x"]] == "correct" else 0,
            timestamp,
        ))

        new_lines.append(line)

#write corrected output
with open(output_file + ".csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerows(new_lines)

#perform logistic regression with one variable, trial_num
def single_var_regression(reg_trials: (int, datetime)) -> linear_model.LogisticRegression:
    x_train = np.array([i for i in range(len(reg_trials))])
    y_train = np.array([trial[0] for trial in reg_trials])

    model = linear_model.LogisticRegression()
    model.fit(x_train.reshape(-1, 1), y_train)

    return model

def quadratic_regression(reg_trials: (int, datetime)) -> linear_model.LogisticRegression:
    x_train = np.array([[pow(i, j) for j in range(1,6)] for i in range(len(reg_trials))])
    y_train = np.array([trial[0] for trial in reg_trials])

    #x = np.arange(0, len(reg_trials))
    #x_poly = PolynomialFeatures(degree=5, include_bias=False)
    #poly_features = x_poly.fit_transform(x.reshape(-1, 1))

    model = linear_model.LogisticRegression(solver='saga', C=10, max_iter=5000)
    model.fit(x_train, y_train)
    #model.fit(poly_features, y_train)

    predict = model.predict(x_train)

    return model

def rolling_averages(reg_trials: (int, datetime)) -> List[np.floating]:
    moving_averages = []
    for i in range(1, len(reg_trials) - 1):
        moving_averages.append(np.mean([reg_trials[i - 1][0], reg_trials[i][0], reg_trials[i + 1][0]]))
    return moving_averages

single_var = single_var_regression(trials)
quadratic = quadratic_regression(trials)
averages = rolling_averages(trials)

#save models
with open("single_var_" + output_file + ".pkl", 'wb') as single_var_file:
    pickle.dump(single_var, single_var_file)
with open("quadratic_" + output_file + ".pkl", 'wb') as quadratic_var_file:
    pickle.dump(quadratic, quadratic_var_file)
with open("rolling_avgs_" + output_file + ".csv", 'w', newline='') as rolling_avgs_file:
    writer = csv.writer(rolling_avgs_file, delimiter=',')
    writer.writerows([[i + 1, averages[i]] for i in range(len(averages))])