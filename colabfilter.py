import sys
import warnings
import numpy as np
from math import sqrt
import math
import os

dataset_name="TrainingRatings.txt"
#path = os.path.join(os.getcwd(), data_set_name)
#path = os.path.join(path, "TrainingRatings.txt")
with open(dataset_name) as files:
    string_train = files.readlines()
data_set_name="TestingRatings.txt"
with open(dataset_name) as files:
    string_test = files.readlines()
    
traindata = {}
for each in string_train:
    each = each.rstrip()
    string = each.split(',')
    if string[1] not in traindata:
        traindata[string[1]] = {}
        traindata[string[1]][string[0]] = string[2]
    else:
        traindata[string[1]][string[0]] = string[2]
testdata = {}
for each in string_test:
    each = each.rstrip()
    string = each.split(',')
    if string[1] not in testdata:
        testdata[string[1]] = {}
        testdata[string[1]][string[0]] = string[2]
    else:
        testdata[string[1]][string[0]] = string[2]

def calculate_mean_vote_train(traindata):
    mean_vote = {}
    for each_user in traindata:
        if isinstance(traindata[each_user], dict):
            sum_of_reviews = 0
            total_number_of_reviews = 0
            for each_movie in traindata[each_user]:
                sum_of_reviews = sum_of_reviews + np.fromstring(traindata[each_user][each_movie], dtype=np.float64, sep=" ")[0]
                total_number_of_reviews += 1
            mean_vote[each_user] = sum_of_reviews / float(total_number_of_reviews)
    return mean_vote

def calculate_mean_vote_test(test_example, test_movie_id, test_user_id, train_example):
    sum_of_reviews = 0
    total_number_of_reviews = 0
    user_movies = train_example[test_user_id]
    for each_movie in user_movies:
        sum_of_reviews = sum_of_reviews + np.fromstring(user_movies[each_movie], dtype=np.float64, sep=" ")[0]
        total_number_of_reviews += 1
    return sum_of_reviews / float(total_number_of_reviews)

def test(data, test_user, mean_vote_train_data, test_movie_id, test_user_id):
    test_mean_vote = calculate_mean_vote_test(test_user, test_movie_id, test_user_id, data)
    movie_number = test_movie_id
    predicted_vote = test_mean_vote
    normalizing_factor = 1
    sum = 0
    for each_user in data:
        if movie_number in list(data[each_user]):
            sum += get_correlation(data[each_user], test_user, mean_vote_train_data[each_user], test_mean_vote, test_movie_id, test_user_id, data[test_user_id]) * (
                np.fromstring(data[each_user][movie_number], dtype=np.float64, sep=" ")[0] - mean_vote_train_data[each_user])
            normalizing_factor = normalizing_factor + abs(sum)
    normalizing_factor = 1/float(normalizing_factor)
    predicted_vote += normalizing_factor * sum
    return predicted_vote

def get_correlation(train_data_for_user, test_data_for_new_user, mean_vote_train_user, mean_vote_test_point, test_movie_id, test_user_id, all_movies_rated_by_test_user):
    numerator = 0
    denominator_sum_1 = 0
    denominator_sum_2 = 0
    movies_rated_by_test_user = list(all_movies_rated_by_test_user)
    # Here we follow the equation and find the denominator and the numerator and then divide accordingly
    for each in movies_rated_by_test_user:
        if each in train_data_for_user and np.fromstring(train_data_for_user[each], dtype=np.float64, sep=" ")[0] != mean_vote_train_user and (np.fromstring(all_movies_rated_by_test_user[each], dtype=np.float64, sep=" "))[0] != mean_vote_test_point:
            numerator += (np.fromstring(all_movies_rated_by_test_user[each], dtype=np.float64, sep=" ")[0] - mean_vote_test_point) * (
                np.fromstring(train_data_for_user[each], dtype=np.float64, sep=" ")[0] - mean_vote_train_user)
            denominator_sum_1 += (np.fromstring(all_movies_rated_by_test_user[each], dtype=np.float64, sep=" ")[0] - mean_vote_test_point) ** 2
            denominator_sum_2 += (np.fromstring(train_data_for_user[each], dtype=np.float64, sep=" ")[0] - mean_vote_train_user) ** 2
        else:
            return 0
    return numerator / float(math.sqrt(denominator_sum_2 * denominator_sum_1))

def find_output(data, test_user, test_movie_id, test_user_id):
    mean_vote_train_data = calculate_mean_vote_train(data)
    predicted_rating = test(data, test_user, mean_vote_train_data, test_movie_id, test_user_id)
    return data, predicted_rating

def convert_string_to_int(value):
    return np.fromstring(value, dtype=np.float64, sep=" ")[0]

def get_mean_absolute_error(actual_y, predicted_y):
    absolute_error = 0
    for each_prediction in range(len(actual_y)):
        absolute_error += abs(actual_y[each_prediction] - predicted_y[each_prediction])
    return absolute_error / float(len(actual_y))

def get_root_mean_squared_error(actual_y, predicted_y):
    squared_error = 0
    for each_prediction in range(len(actual_y)):
        squared_error += (actual_y[each_prediction] - predicted_y[each_prediction]) ** 2
    return sqrt(squared_error / float(len(actual_y)))

predicted_values = []
actual_values = []
for each_user_key in testdata:
    for each_movie in testdata[each_user_key]:
        user_movie_pair = {each_user_key: {each_movie: testdata[each_user_key][each_movie]}}
        data, predicted_rating = find_output(traindata, user_movie_pair,each_movie, each_user_key)
        predicted_values.append(predicted_rating)
        actual_values.append(np.fromstring(testdata[each_user_key][each_movie], dtype=np.float64, sep=" ")[0])

    mae = get_mean_absolute_error(actual_values, predicted_values)
    rmse = get_root_mean_squared_error(actual_values, predicted_values)
    print("MAE:",mae,"     RMSE:",rmse)