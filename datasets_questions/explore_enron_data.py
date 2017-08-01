#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print('Total data point: {}'.format(len(enron_data.keys())))
k, features = next(iter(enron_data.items()))
print('Total features: {}'.format(len(features)))

import pandas as pd

df = pd.DataFrame([v for v in enron_data.values()], index=enron_data.keys())

print('Total person of interest: ', df[df['poi'] == 1].shape[0])

print('No. of POI exist: ', sum(1 if line[:1] == '(' else 0 for line in open('../final_project/poi_names.txt')))
print('Total value of stock belonging to James Prentice: ', df.loc['PRENTICE JAMES']['total_stock_value'])
print('Total Email message from Wesley Colwell to poi: ', df.loc['COLWELL WESLEY']['from_this_person_to_poi'])
print('Total stock value excercised by Jeffrey K Skilling: ', df.loc['SKILLING JEFFREY K']['exercised_stock_options'])

most_money = df[df.total_payments != 'NaN'].sort_values('total_payments', ascending=False).head(2).tail(1).total_payments
print('Most money taken by: ', most_money.index.values[0], ' worth ', most_money.values[0])

#print(df.columns)
print('Folks having quantified salary: ', df[df.salary != 'NaN'].shape[0])
print('Folks having known email_address: ', df[df.email_address != 'NaN'].shape[0])

print('Percentage of NaN total_payments: ', df[df.total_payments == 'NaN'].shape[0]*100/df.shape[0])

poi = df[df['poi'] == 1]
print('Percentage of POIs having NaN total_payments: ', poi[poi.total_payments == 'NaN'].shape[0]*100/poi.shape[0])
newdf = df.shape[0] + 10
print('New df after poi with 10 NaN payment: ', newdf)
print('New NaN for total_payments: ', df[df.total_payments == 'NaN'].shape[0] + 10)

print('New POI after poi with 10 NaN payment: ', poi.shape[0] + 10)
print('New NaN for POI total_payments: ', 10)
