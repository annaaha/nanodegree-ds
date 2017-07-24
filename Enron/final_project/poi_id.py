#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tested with Python 2.7 and scikit-learn version 0.18.1
import sys
import pickle
import warnings
import numpy
import pandas as pd
from validate_email import validate_email
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from time import time
from collections import OrderedDict
import copy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit  # noqa: E402
from tester import dump_classifier_and_data  # noqa: E402

pd.set_option('display.max_rows', 999)
pd.set_option('display.width', 600)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)

warnings.filterwarnings("ignore")
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Data summary
# Number of points in dataset
print "Number of points in dataset: ", len(data_dict)

# Number of features, same for each key in dataset (dictionary)
print "Number of features: ", len(data_dict.values()[0])
# List all features
all_features = []
for i in data_dict.values()[0].keys():
    all_features.append(i)

print "All features in dataset: ", all_features

# The number of POI's in the E+F dataset. The number of non POI's.
count_poi, count_npoi = 0, 0
for i in range(0, len(data_dict.values())):
    if data_dict.values()[i]['poi']:
        count_poi = count_poi + 1
    else:
        count_npoi = count_npoi + 1

print "Number of POI's in E+F: ", count_poi
print "Number of non POI's: ", count_npoi

print "-------------"
print "Data Cleaning"
print "-------------"


def not_valid_feature(dataset, features):
    """ Checks if features in dataset have invalid values and print them.
Email validation according to RFC 2822
    """
    print "Features with invalid value: "
    for j in features:
        for keys, values in dataset.items():
            if j == 'email_address':
                if validate_email(values[j]) is False and values[j] != 'NaN':
                    print keys, "has invalid %s" % j, values[j], j
            elif j in ['deferred_income', 'restricted_stock_deferred']:
                if values[j] > 0 and values[j] != 'NaN':
                    print keys, "has incorrect %s" % j, values[j], j
            else:
                if values[j] < 0 and values[j] != 'NaN':
                    print keys, "has incorrect %s" % j, values[j]
    return


# Check if features contain invalid values and print them
not_valid_feature(data_dict, all_features)

""" Email address can not be corrected, because I do not have a list with
correct email addresses. I have checked the email_by_address folder and
there are the same invalid addresses.

Studing the results of not_invalid_feature I discovered that the values
exist in enron61702insiderpay.pdf for the specific person, but there are
in the wrong column (feature). It seems that the scan process created
these misstakes.

The audit functions below detect the lines where feature values are shifted
"""


# Make a copy of all_features
features_all = list(all_features)
# Remove 'email_address', causes problems when using featureFormat
features_all.remove('email_address')
data_all = featureFormat(data_dict, features_all, sort_keys=True)

re_data_dict = data_dict.copy()  # Create new dict which is a copy of data_dict
# Reorder the dictionary by key
re_data_dict = OrderedDict(sorted(re_data_dict.items(), key=lambda t: t[0]))

data_dict = re_data_dict

# len(data_ll)!= len(re_data_dict), 145 != 146 It seems that this row
# is deleted when using featureFormat to find data_all, because all values
# are NAN->0. Rows with only 0 is deleted. For matching the data_all and
# re_data_dict 1:1 I have to delete this entry in dictionary re_data_dict so
# that data_all and re_data_dict have the same len.

# Remove key because all his feature values are NaN
index = data_dict.keys().index("LOCKHART EUGENE E")
data_dict.pop("LOCKHART EUGENE E", index)


def map_data_dataset(data, dataset):
    """Create a mapping dictionary, where keys rows in data and values index of
dataset (dataset) """
    mapping = {}
    for i in range(0, len(data)):
        if (data[i][0] == dataset.values()[i]['salary'] and
                data[i][5] == dataset.values()[i]['bonus']) or \
                (data[i][0] == 0.0 and dataset.values()[i]['bonus'] == 'NaN') \
                or (data[i][0] != 0.0 and
                    dataset.values()[i]['bonus'] == 'NaN') or \
                (data[i][0] == 0.0 and dataset.values()[i]['bonus'] != 'NaN'):
            mapping[i] = dataset.keys().index(dataset.keys()[i])

    return mapping


# Mapping between data_all and re_data_dict
map_dict = map_data_dataset(data_all, data_dict)

# Find lines to be corrected


def audit_total_payments(audit_dict, audit_data):
    """ Find keys (persons) where the value in total_payments is not
the sum of value of salary, bonus, long_term_incentive, deferred_income,
deferral_payments, loan_advances, other, expenses, and director_fees.
There is a left/right shift of data due scanning.
"""
    wrong_total_payments, wrong_lines = [], []
    for i in range(0, len(audit_data)):
        if (audit_data[i][0] + audit_data[i][5] + audit_data[i][18] +
                audit_data[i][17] + audit_data[i][2] + audit_data[i][11] +
                audit_data[i][13] + audit_data[i][10] +
                audit_data[i][16]) != audit_data[i][3]:
            wrong_total_payments.append(i)

    for i in wrong_total_payments:
        wrong_lines.append(audit_dict.keys()[i])  # Find person in line i

    return wrong_lines


def audit_total_stock_value(audit_dict, audit_data):
    """Find keys (persons) where the value in total_stock_value \ is not
the sum of value of exercised_stock_options, restricted_stock, \and
restricted_stock_deferred. There is a left/right shift of data due
scanning.
"""
    wrong_total_stock_value, wrong_lines = [], []
    for i in range(0, len(audit_data)):
        if (audit_data[i][4] + audit_data[i][6] +
                audit_data[i][8]) != audit_data[i][9]:
            wrong_total_stock_value.append(i)

    for i in wrong_total_stock_value:
        wrong_lines.append(audit_dict.keys()[i])  # Find person in line i

    return wrong_lines


def update_dict(dataset, Name, payment_art, value):
    """Given employee name and payment category, new value \
is set. Employee name and payment category are strings, \
value is float. """
    dataset[Name][payment_art] = value

    return dataset[Name][payment_art]


wrong_values_total_payments = audit_total_payments(data_dict, data_all)
print "Lines to be corrected:", wrong_values_total_payments, ", \
'total_payments' incorrect"
wrong_values_total_stock = audit_total_stock_value(data_dict, data_all)
print "Lines to be corrected:", wrong_values_total_stock, ", \
'total_stock_value' incorrect"

# Set correct values to line 'BELFER ROBERT' according to
# enron61702insiderpay.pdf

new_salary = update_dict(data_dict, 'BELFER ROBERT', 'salary', 'NaN')
new_bonus = update_dict(data_dict, 'BELFER ROBERT', 'bonus', 'NaN')
new_long_term_incentive = update_dict(data_dict, 'BELFER ROBERT',
                                      'long_term_incentive', 'NaN')
new_deferred_income = update_dict(data_dict, 'BELFER ROBERT',
                                  'deferred_income', -102500)
new_deferral_payments = update_dict(data_dict, 'BELFER ROBERT',
                                    'deferral_payments', 'NaN')
new_loan_advances = update_dict(data_dict, 'BELFER ROBERT',
                                'loan_advances', 'NaN')
new_other = update_dict(data_dict, 'BELFER ROBERT', 'other', 'NaN')
new_expenses = update_dict(data_dict, 'BELFER ROBERT', 'expenses', 3285)
new_director_fees = update_dict(data_dict, 'BELFER ROBERT', 'director_fees',
                                102500)
new_total_payments = update_dict(data_dict, 'BELFER ROBERT', 'total_payments',
                                 3285)
new_exercised_stock_options = update_dict(data_dict, 'BELFER ROBERT',
                                          'exercised_stock_options', 'NaN')
new_restricted_stock = update_dict(data_dict, 'BELFER ROBERT',
                                   'restricted_stock', 44093)
new_restricted_stock_deferred = update_dict(data_dict,
                                            'BELFER ROBERT',
                                            'restricted_stock_deferred',
                                            -44093)
new_total_stock_value = update_dict(data_dict, 'BELFER ROBERT',
                                    'total_stock_value', 'NaN')

# Set correct values to line 'BHATNAGAR SANJAY' according to
# enron61702insiderpay.pdf

new_salary = update_dict(data_dict, 'BHATNAGAR SANJAY', 'salary', 'NaN')
new_bonus = update_dict(data_dict, 'BHATNAGAR SANJAY', 'bonus', 'NaN')
new_long_term_incentive = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                      'long_term_incentive', 'NaN')
new_deferred_income = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                  'deferred_income', 'NaN')
new_deferral_payments = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                    'deferral_payments', 'NaN')
new_loan_advances = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                'loan_advances', 'NaN')
new_other = update_dict(data_dict, 'BHATNAGAR SANJAY',
                        'other', 'NaN')
new_expenses = update_dict(data_dict, 'BHATNAGAR SANJAY',
                           'expenses', 137864)
new_director_fees = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                'director_fees', 'NaN')
new_total_payments = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                 'total_payments', 137864)
new_exercised_stock_options = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                          'exercised_stock_options', 15456290)
new_restricted_stock = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                   'restricted_stock', 2604490)
new_restricted_stock_deferred = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                            'restricted_stock_deferred',
                                            -2604490)
new_total_stock_value = update_dict(data_dict, 'BHATNAGAR SANJAY',
                                    'total_stock_value', 15456290)


# Repeat audit functions to varify that the cleaning process is completed
print "Verify if data is cleaned"
data_all = featureFormat(data_dict, features_all, sort_keys=True)


wrong_values_total_payments = audit_total_payments(data_dict, data_all)
# print "Lines to be corrected:", wrong_values_total_payments
wrong_values_total_stock = audit_total_stock_value(data_dict, data_all)
# print "Lines to be corrected:", wrong_values_total_stock

# Data Analysis
# How many NAN exist in each feature-EDW. Keep features where availability
# is equal or greater than 55%. I have chosen 55%.


def find_features(dataset, features, percent):
    """Find features in dataset with percent available values"""
    count_nan, count, my_features, NAN, AVBLE, Per_nan, Per = 0, 0, [], [], \
        [], [], []
    # Go through each feature in order to count NaN and available values
    for j in range(0, len(features)):
        for i in dataset.values():
            if i.values()[j] == "NaN":
                count_nan = count_nan + 1
            else:
                count = count + 1

        if round(count*100/146.0, 0) >= percent:
            my_features.append(dataset.values()[0].keys()[j])
        NAN.append(count_nan)
        Per_nan.append(round(count_nan*100.0/len(dataset), 0))
        AVBLE.append(count)
        Per.append(round(count*100.0/len(dataset), 0))
        count_nan, count = 0, 0

    df_con = pd.DataFrame(columns=["Number of NaN", "Percent NaN",
                                   "Number of Available",
                                   "Percent Available"],
                          index=features)

    df_con["Number of NaN"] = NAN
    df_con["Percent NaN"] = Per_nan
    df_con["Number of Available"] = AVBLE
    df_con["Percent Available"] = Per

    print df_con
    return my_features


print "------------------------------"
print "Features and data availability"
print "------------------------------"
my_features = find_features(data_dict, all_features, 55.0)
print "---------------------------------------"
print "Pre-selected Features                  "
print "Features with available data >= 55.0%: "
print "---------------------------------------"
print my_features
# print "Legnth of my_features: ", len(my_features)

# The feature email_address which is categorical, causes problem by using
# featureFormat. It has to be deleted.
my_features.remove("email_address")
# Update data_my
data_my = featureFormat(data_dict, my_features, sort_keys=True)

### Task 2 Remove outliers  # noqa: E266
#### 2.1  # noqa: E266
'''I use the Mahalanobis distance squared for finding outliers. I decided to
use this method due to the data is multivariate, my_features contains 14
features.
'''


def MahalanobisDistance(a):
    """Mahalanobis distance squared, where a is numpy ndarray"""
    means = []
    rows, columns = a.shape  # innan x,y respektive
    # Find the means for each feature (column)
    for i in range(0, columns):
        mean_value = round(a.T[i].mean(), 2)
        means.append(mean_value)

    a_diff = numpy.empty(shape=(rows, columns))
    # Substract the resp. mean from each column(feature)
    for i in range(0, columns):
        for j in range(0, rows):
            diff_means = a[j][i]-means[i]
            a_diff[j][i] = diff_means

    covariance = numpy.cov(a, rowvar=0)

    # Find inverse of covariance, covariance_inverse only if determinant is not
    # zero
    if numpy.linalg.det(covariance) != 0:
        covariance_inverse = numpy.linalg.inv(covariance)
        MD_2 = a_diff.dot(covariance_inverse).dot(a_diff.transpose())
        MD_2 = numpy.diagonal(MD_2)
        return MD_2
    else:
        print "Mahalanobis Distance can not be found!"

    return MD_2


# Mahalanobis squared distance is approximated with Chi-square
# distribution. The outlier threshold value is derived from
# #http://uregina.ca/~gingrich/appchi.pdf, where df = len(data_my.T) and
# probability2 p <= 0.001. In this case degree of freedom is df = 14
# (=len(my_features)), and threshold = 36.123)

MD_distance = MahalanobisDistance(data_my)
threshold_outliers = 36.123


def find_outliers(dataset, data, distance, threshold):
    """Find outliers, where dataset the data_dict, data is derived from
    featureFormat, threshold chi-squared value, where df = len(features),
    assuming probabilty p = 0.001
    """
    index_outliers, outliers, possible_outliers = [], {}, []
    for i in range(0, len(distance)):
        if distance[i] >= threshold:
            index_outliers.append(i)
            outliers[i] = {"Name": dataset.keys()[i], "MD_distance":
                           distance[i]}
            possible_outliers.append(dataset.keys()[i])

    return possible_outliers, index_outliers


possible_outliers = find_outliers(data_dict, data_my, MD_distance,
                                  threshold_outliers)

print "-----------------"
print "Possible Outliers"
print "-----------------"
print possible_outliers[0]


def plot_outliers_t(dataset, distance, outliers, threshold):
    """Plot Mahalanobis distance for each row in numpy data. Outliers marked
    with red. Outliers are used by using squared Mahalanobis distance and chi
    square distribution, when p = 0.001
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(range(0, len(distance)), distance, c='b',
                label='Not Outlier')
    plt.ylim(0, max(distance)+5)
    plt.ylabel('Mahalanobis Squared Distance')
    plt.xlabel('Employees represented as index in dataset')
    plt.axhline(threshold, c='r', linestyle='dotted',
                label='Outlier Threshold = 36.123')
    # plt.legend(['Not Outliers', 'Outliers'])
    for i in outliers:
        if i == outliers[0]:
            plt.scatter(i, distance[i], c='r', label='Outlier')
            plt.annotate(dataset.keys()[i], xy=(i, distance[i]),
                         xytext=(3, 3), fontsize=6, textcoords='offset points',
                         rotation=0, horizontalalignment='center',
                         verticalalignment='upper')
        else:
            plt.scatter(i, distance[i], c='r')
            plt.annotate(dataset.keys()[i], xy=(i, distance[i]),
                         xytext=(3, 3), fontsize=6, textcoords='offset points',
                         rotation=0, horizontalalignment='center',
                         verticalalignment='upper')

    plt.title('Representation of Enron Employees as in Inliers and Outliers\n \
    based on Mahalanobis Squared Distance')
    plt.legend(loc='upper left', borderaxespad=0., fontsize=8)
    # plt.savefig('possible_outliers.png')
    # plt.show()
    return


plot_outliers_t(data_dict, MD_distance, possible_outliers[1],
                threshold_outliers)

# Notes Mr. Pai Lou L. did not appear as outlier. I was expecting that
# Mr. Skilling should had a larger Mahalanobis distance. But this can be
# explained, since Mahalanobis distance is calculatated by using fourteen
# variables, and not only two as for example in salary vs bonus.

# TOTAL is not a outliers due to it is not a person. It has to be deleted from
# the dataset. TOTAL is not a valid key,it is not a person.

index_total = data_dict.keys().index('TOTAL')  # Find index of 'TOTAL' in dict
data_dict.pop("TOTAL", index_total)  # Remove 'TOTAL' from dict

# Redefine my data (update)
data_my = featureFormat(data_dict, my_features, sort_keys=True)

print "--------------------------------------"
print "Outliers after deletion of key 'TOTAL'"
print "--------------------------------------"
# find index of TOTAL in possible_outliers[0], which is same as in
# possible_outliers[1]
possible_outliers[0].index('TOTAL')
possible_outliers[0].pop(12)  # Delete 'TOTAL' from list possible_outliers[0]
# Delete index of 'TOTAL' from list possible_outliers[1]
possible_outliers[1].pop(12)
print possible_outliers[0]

# By printing/investigating the keys in data_dict, falls that 'THE TRAVEL
# AGENCY IN THE PARK' is also not a person also not a valid outlier. These two
# key values have to be deleted from the dictionary, because they are not valid
# outliers.

# Remove outlier "THE TRAVEL AGENCY IN THE PARK" from dataset
index = data_dict.keys().index("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", index)

# Redefine my data after deleting the above not valid outliers.
data_my = featureFormat(data_dict, my_features, sort_keys=True)


### Task 3: Create new feature(s)  # noqa: E266
"""Before I create new features I would like to see the relation between
feature 'poi' and the other features in the dataset. I use my_features efter
deleting the feature "email_address" which is categorical and causes problems
by selecting KBEST algorithm.
"""

### START  # noqa: E266
## Tagen from StudentCode.py som finns in Create_new_feature # noqa: E266


def computeFraction(poi_messages, all_messages):
    # given a number messages to/from POI (numerator)
    # and number of all messages to/from a person (denominator),
    # return the fraction of messages to/from that person
    # that are from/to a POI
    """ you fill in this code, so that it returns either
the fraction of all messages to this person that come from POIs
or the fraction of all messages from this person that are sent to POIs
the same code can be used to compute either quantity
beware of "NaN" when there is no known email address (and so
no filled email features), and integer division!
in case of poi_messages or all_messages having "NaN" value, return 0.
"""
    fraction = 0.
    if all_messages != 0 and not (poi_messages == "NaN" or
                                  all_messages == "NaN"):
        fraction = float(poi_messages)/float(all_messages)
    return fraction


# Create new features
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    # print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    # submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
    # "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

# List of all features after adding the new features "fraction_from_poi"
# and "fraction_to_poi"
all_features_new = []
for i in data_dict.values()[0].keys():
    all_features_new.append(i)


print "--------------------------------------"
print "All features after creating new ones: "
print "--------------------------------------"
print all_features_new
print my_features
my_features.append('fraction_from_poi')
my_features.append('fraction_to_poi')
print "------------------------------------------------------"
print "Pre-selected features after creating new one's    "
print "------------------------------------------------------"
print my_features


### Task 1: Select what features you'll use.# noqa: E266
### features_list is a list of strings, each of which is # noqa: E266
### a feature name.# noqa: E266
### The first feature must be "poi".# noqa: E266
# features_list = ['poi','salary']
# You will need to use more features

# Step 0.


def set_poi_first_position(list):
    """ Reorder the pos of "poi" in list or insert\
"poi" in list
"""
    if "poi" in list:
        if list.index("poi") != 0:
            list.remove("poi")
            list.insert(0, "poi")
    else:
        list.insert(0, "poi")

    return list

# 4. Quick Validation of classifiers


print "-----------------------------------------------------------"
print " Validation algorithm                   "
print "-----------------------------------------------------------"
print " a. Use Classification Report-f1_score           "
print " b. Use Cross Validation Score               "
print " c. Use StratifiedShuffleSplit               "
print " d. Use ShuffleSplit                    "
print " e. Use Stratifiedkfold                  "
print "-----------------------------------------------------------"

pipe_svc = Pipeline([('preprocessing', MinMaxScaler()),
                     ('feature_selection', SelectKBest()),
                     ('classifier', SVC(random_state=42))])

pipe_gnb = Pipeline([('preprocessing', MinMaxScaler()),
                     ('feature_selection', SelectKBest()),
                     ('classifier', GaussianNB())])

pipe_dt = Pipeline([('preprocessing', MinMaxScaler()),
                    ('feature_selection', SelectKBest()),
                    ('classifier', DecisionTreeClassifier(random_state=42))])

pipe_knn = Pipeline([('preprocessing', MinMaxScaler()),
                     ('feature_selection', SelectKBest()),
                     ('classifier', KNeighborsClassifier())])


pipes = [pipe_svc, pipe_gnb, pipe_dt, pipe_knn]


def quick_validation(list_pipe, dataset, features):
    '''Take as input a list of pipelines, a dict dataset and a feature
    list. Validates and prints out model performance by using
    classification_report, cross_val_score, ShuffleSplit,
    StratifiedShuffleSplit and StratifiedKFold
'''
    set_poi_first_position(features)
    data = featureFormat(dataset, features, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test =\
        train_test_split(features, labels, test_size=0.3, random_state=42)
    quick_val = pd.DataFrame(index=["SVC", "GaussianNB", "DecisionTree",
                                    "KNeighbors"],
                             columns=["Use Classification Report-f1_score",
                                      "Use Cross Validation Score, f1-score",
                                      "Use ShuffleSplit, f1-score",
                                      "Use StratifiedShuffleSplit, f1-score",
                                      "Use StratifiedKFold, f1-score"])
    rows = []
    for i in range(0, len(list_pipe)):
        clf = pipes[i].fit(features_train, labels_train)
        pred = clf.predict(features_test)
        # Results from metrics.f1_score
        report = f1_score(labels_test, pred)
        scores_val = cross_val_score(pipes[i], features_train, labels_train,
                                     scoring='f1', cv=10)
        cv_ss = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7,
                             random_state=42)
        cv_sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3,
                                        train_size=0.7, random_state=42)
        cv_skfold = StratifiedKFold(n_splits=10, random_state=42)
        # Results cros_val_score, cv = cv_ss
        scores_ss = cross_val_score(pipes[i], features_train, labels_train,
                                    scoring='f1', cv=cv_ss)
        # Rsults cros_val_score, cv = cv_sss
        scores_sss = cross_val_score(pipes[i], features_train,
                                     labels_train, scoring='f1', cv=cv_sss)
        # Results cros_val_score, cv = cv_skfold
        scores_skfold = cross_val_score(pipes[i], features_train, labels_train,
                                        scoring='f1', cv=cv_skfold)
        # list of f1-score results
        row = [report, scores_val.mean(), scores_ss.mean(), scores_sss.mean(),
               scores_skfold.mean()]
        rows.append(row)  # Insert values of row in list rows
    for i in range(0, len(quick_val.index)):
        # Assign each row in quick_val with resp. row
        quick_val.loc[quick_val.index[i]] = rows[i]
    return quick_val  # Return a dataframe


quick_val = quick_validation(pipes, data_dict, my_features)
print quick_val
# TASK 5 - Have choosen GaussianNB and DicisionTreeClassifier to investigate

# Modify test_classifier and name it my_test_classifier,
# because I want to save the evaluation metrics for plotting


def my_test_classifier(clf, dataset, feature_list, folds=1000):
    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
            for jj in test_idx:
                features_test.append(features[jj])
                labels_test.append(labels[jj])
    # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives\
            + true_positives
        accuracy = round(1.0*(true_positives +
                              true_negatives)/total_predictions,
                         5)
        precision = round(1.0*true_positives/(true_positives+false_positives),
                          5)
        recall = round(1.0*true_positives/(true_positives+false_negatives),
                       5)
        f1 = round(2.0 * true_positives/(2*true_positives + false_positives +
                                         false_negatives), 5)
        f2 = round((1+2.0*2.0) * precision*recall/(4*precision + recall), 5)
        # print clf
        # print "Accuracy: ", accuracy
        print "Precision: ", precision
        print "Recall: ", recall
        print "F1-score: ", f1
        # print "F2-Score: ", f2
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack\
of true positive predicitons."

    return accuracy, precision, recall, f1, f2

# Create function that tunes and validates model


def gridSearchCV_validation(pipe_in, param_in, dataset,
                            features_in, min_number_features):
    ''' In parameters: pipeline, dictionary of hyperparameters, dataset
    dictionary, feature list and the min number features which to take in
    acount. Returns two dataframes gs_clf_df, and vd_clf_df. The function works
    only if the pipeline has the following steps: preprocessing,
    feature_selection, SelectKBest() and classifier
    '''
    set_poi_first_position(features_in)
    data = featureFormat(dataset, features_in, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    # Create dataframe for GridSearchCV results
    gs_clf_df = pd.DataFrame(columns=["Number Features",
                                      "Precision",
                                      "Recall",
                                      "F1-Score",
                                      "Best Estimator",
                                      "Feature List",
                                      "Features Scores"])
    # Create dataframe for validation results
    vd_clf_df = pd.DataFrame(columns=["Number Features",
                                      "Accuracy",
                                      "Precision",
                                      "Recall",
                                      "F1-Score",
                                      "F2-score",
                                      "Best Estimator",
                                      "Feature List"])
    cv_strata = StratifiedShuffleSplit(100, test_size=0.3, random_state=42)
    for k in range(min_number_features, len(features_in)):
        param_in['feature_selection__k'] = [k]
        scores = ['precision', 'recall', 'f1']
        score_dict = {}
        t0 = time()
        for score in scores:
            gs = GridSearchCV(pipe_in, param_in, cv=cv_strata,
                              scoring=score, n_jobs=-1,
                              refit=True, iid=False)
            gs.fit(features_train, labels_train)
            score_dict[score] = gs.best_score_
        print "GridSearchCV Results:"
        print "---------------------"
        print("done in %0.3fs" % (time() - t0))
        print "Number of selected Features: ", k
        print "Precision: ", round(score_dict['precision'], 5)
        print "Recall: ", round(score_dict['recall'], 5)
        print "F1-Score: ", round(score_dict['f1'], 5)
        # print gs.best_estimator_.named_steps['feature_selection'].\
        #    get_support(indices=True)
        features_list = [features_in[j+1] for j in
                         gs.best_estimator_.
                         named_steps['feature_selection'].
                         get_support(indices=True)]
        # print gs.best_estimator_, gs.best_estimator_.\
        #    named_steps['feature_selection'].scores_
        # gridSearchCV results for each loop
        row_gs = (k,
                  score_dict['precision'],
                  score_dict['recall'],
                  score_dict['f1'],
                  gs.best_estimator_,
                  features_list,
                  [round(gs.best_estimator_.
                         named_steps['feature_selection'].scores_[i], 5)
                   for i in
                   gs.best_estimator_.
                   named_steps['feature_selection'].get_support(indices=True)])
        gs_clf_df.loc[len(gs_clf_df)] = row_gs
        set_poi_first_position(features_list)
        # print features_list
        clf_test = copy.deepcopy(gs.best_estimator_)
        # Validation of model results. I have modified test_classifier
        # in order to save results, needs for plot.
        print "Validation Results:"
        print "-------------------"
        validation_results_gnb = my_test_classifier(clf_test,
                                                    dataset,
                                                    features_list)
        row_val = (k,
                   validation_results_gnb[0],
                   validation_results_gnb[1],
                   validation_results_gnb[2],
                   validation_results_gnb[3],
                   validation_results_gnb[4],
                   clf_test,
                   features_list)
        vd_clf_df.loc[len(vd_clf_df)] = list(row_val)  # Insert row in table

    gs_clf_df.index = range(min_number_features, len(features_in))
    vd_clf_df.index = range(min_number_features, len(features_in))
    return gs_clf_df, vd_clf_df


# Create function that plot results (dataframes)
def plot_gs_val_results(df, title_text, file_save_name):
    min_nr_features = min(list(df.index))
    pos = range(min_nr_features, len(df.index) + min_nr_features)
    width = 0.25
    plt.bar(pos,
            df["Precision"],
            width, alpha=0.5,
            color="#FFC222",
            label="Precision")
    plt.bar([p + width for p in pos],
            df["Recall"],
            width,
            color="#F78F1E",
            label="Recall")
    plt.bar([p + width*2 for p in pos],
            df["F1-Score"],
            width,
            color="#EE3224",
            label='F1-Score')
    plt.xlabel("Selected Number of Features, 'poi' not included")
    plt.ylabel("'Precision, Recall, F1-score")
    plt.title(title_text)
    plt.xticks(numpy.arange(3.25, 16.25, 1), pos)
    plt.xlim(min(pos)-width, max(pos) + width*3)
    plt.yticks(numpy.arange(0.000, round(max(df["Precision"].max(),
                                             df["Recall"].max(),
                                             df["F1-Score"].max()), 3) +
                            0.05, 0.025))
    plt.axhline(0.3, color='g', linestyle='--',
                label="Minimum Value\nto be fullfilled")
    plt.axhline(df["Precision"].max(),
                alpha=0.5,
                color='#FFC222',
                linestyle='--',
                label="Max Precision Value")
    plt.axhline(df["Recall"].max(),
                color='#F78F1E',
                linestyle='--',
                label='Max Recall Value')
    plt.axhline(df["F1-Score"].max(),
                color='#EE3224',
                linestyle='--',
                label='Max f1 Value')
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0., fontsize=8)
    # plt.savefig(file_save_name)
    # plt.show()
    return


# 5a. Look closer at pipeline with classifier GaussianNB() Classifier
print "--------------------------------------------------------------------"
print "Parameter Tuning Scaling->SelectKBest-> GaussianNB with GridSearchCV"
print "And Validation                           "
print "--------------------------------------------------------------------"

# Here come the pipe_gnb, and para_gnb
pipe_gnb = Pipeline([('preprocessing', MinMaxScaler()),
                     ('feature_selection', SelectKBest()),
                     ('classifier', GaussianNB())])


param_gnb = {'feature_selection__k': [1],
             'classifier__priors': [None]}

gridsearch_validation_gnb = gridSearchCV_validation(pipe_gnb,
                                                    param_gnb,
                                                    data_dict,
                                                    my_features,
                                                    3)
# Recreate plot
grid_gnb_results = gridsearch_validation_gnb[0]
val_gnb_results = gridsearch_validation_gnb[1]

plot_gs_val_results(grid_gnb_results,
                    "GridSearchCV Tuning Results of Pipeline with " +
                    "GaussianNB Classifier\n Precision, Recall and " +
                    "F1-Score vs. Selected KBest Features",
                    "GridSearchCV_GaussianNB.png")

plot_gs_val_results(val_gnb_results,
                    "Validation of Pipeline with " +
                    "GaussianNB Classifier\n" +
                    "Precision, Recall and F1-Score vs. " +
                    "Selected KBest Features",
                    "Validation_Results_GaussianNB.png")

# 5b. Look closer at pipeline with classifier DecisionTreeClassifier()
print "-----------------------------------------------------------------------"
print "Parameter Tuning Scaling->SelectKBest-> Decision Tree with GridSearchCV"
print "And Validation                             "
print "-----------------------------------------------------------------------"
# Pipeline and hyperparameters, DecisionTree

pipe_dt = Pipeline([('preprocessing', MinMaxScaler()),
                    ('feature_selection', SelectKBest()),
                    ('classifier', DecisionTreeClassifier())])

param_dt = {'feature_selection__k': [2],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random'],
            'classifier__max_depth': [2],
            'classifier__random_state': [42]}

# Use gridSearchCV and Validation of model
gridsearch_validation_dt = gridSearchCV_validation(pipe_dt,
                                                   param_dt,
                                                   data_dict,
                                                   my_features,
                                                   3)
# Results of gridSearchCV, dataframe
grid_dt_results = gridsearch_validation_dt[0]
# Results of Validation with my_test_classifier, dataframe
val_dt_results = gridsearch_validation_dt[1]

# Plot Results regarding Precision, Recall and f1-score from gridSearchCV for
# pipe_dt, and validation of pipe_dt


plot_gs_val_results(grid_dt_results,
                    "GridSearchCV Tuning Results of Pipeline with " +
                    "Decision Tree Classifier\n" +
                    "Precision, Recall and F1-Score vs. " +
                    "Selected KBest Features",
                    "GridSearchCV_DecisionTreeClf.png")

plot_gs_val_results(val_dt_results,
                    "Validation Results of Pipeline with " +
                    "Decision Tree Classifier\n Precision, " +
                    "Recall and F1-Score vs. Selected KBest Features",
                    "Validation_DecisionTreeClf.png")


# After comparing the results of the validations,
# the Pipeline with decision tree and with 4 features
# gives the best performance
print "--------------"
print "FINAL RESULTS "
print "--------------"
print "Feature List: "
print "--------------"
# Select from dataframe feature list with seven features + 'poi'
features_list = grid_dt_results.at[4, 'Feature List']
print features_list
print "----------------"
print "Best Estimator: "
print "----------------"
# Best estimator with four features + 'poi'
clf = grid_dt_results.at[4, 'Best Estimator']
print clf
print "----------------"
print "Feature Scores: "
print "----------------"
features_scores = grid_dt_results.at[4, 'Features Scores']
print features_scores
print "---------------------"
print "Feature Importances: "
print "---------------------"
# feature_importances = grid_dt_results.at[4,'Features Importances']
feature_importances = clf.named_steps['classifier'].feature_importances_
print feature_importances
print "--------------------------------------"
print "Results of my_test_classifier functions: "
print "--------------------------------------"
my_dataset = data_dict
my_test_classifier(clf, my_dataset, features_list, 1000)
dump_classifier_and_data(clf, my_dataset, features_list)
