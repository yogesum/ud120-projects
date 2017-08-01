#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

for outlier in data_dict.keys():
  salary = data_dict[outlier]['salary']
  bonus = data_dict[outlier]['bonus']
  if (salary == 'NaN') or (bonus == 'NaN'):
    continue
  if (salary > 1e6) and (bonus > 5e6):
    print('Name of current outlier: ', outlier)

  if (salary > 2.5e7):
    print('Outlier is: ', outlier)
    break

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

