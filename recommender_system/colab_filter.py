#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import sqlite3
from sklearn.decomposition  import NMF

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

src_path = os.path.dirname(os.path.abspath(__file__))

def input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help="verbose logging"
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help="verbose logging"
    )
    parser.add_argument(
        '--user-weights',
        default=os.path.join(src_path, 'pred_user_weights'),
        help="path to user weight matrix"
    )
    parser.add_argument(
        '--stock-weights',
        default=os.path.join(src_path, 'pred_stock_weights'),
        help="path to stock weight matrix"
    )
    parser.add_argument(
        '-u',
        '--user',
        default='cory',
        help="user used for test"
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    return args

def data_base_connect():
    db_path = os.path.join(src_path, './favorites.db')
    if not os.path.isfile(db_path):
        logging.warning(f"invalid database path {db_path}")
    
    connection = sqlite3.connect(db_path)
    return connection

def select_db(conn, sql, parameters=[]):
    c = conn.cursor()
    c.execute(sql, parameters)
    return c.fetchall()


def get_users():
    connection = data_base_connect()
    command = f" SELECT username FROM favorites GROUP BY username"
    unique_users = select_db(
        connection,
        command
    )
    users = []
    for user in unique_users:
        users.append(user[0])

    return users

def get_stocks():
    connection = data_base_connect()
    command = f" SELECT stockticker FROM favorites GROUP BY stockticker"
    unique_stocks = select_db(
        connection,
        command
    )
    stocks = []
    for stock in unique_stocks:
        stocks.append(stock[0])

    return stocks

def build_truth_matrix(users, stocks):
    truth_matrix = np.zeros((len(users), len(stocks)))
    truth_matrix = pd.DataFrame(truth_matrix, index=users, columns=stocks)
    connection = data_base_connect()
    command = f" SELECT * FROM favorites ORDER BY username"
    response = select_db(
        connection,
        command
    )
    stocks = []
    for entry in response:
        truth_matrix.loc[entry[1], entry[3]] = 1.0 # [username][stockname]

    return truth_matrix

def get_user_favorites(username):
    connection = data_base_connect()
    command = f" SELECT * FROM favorites WHERE username=\'{username}\' ORDER BY stockticker"
    response = select_db(
        connection,
        command
    )
    favorited_stocks = []
    for entry in response:
        # Create list of favorited stocks from the database querry response
        favorited_stocks.append(entry[3]) 
    logging.info(f"truth_matrix {favorited_stocks}")

    return favorited_stocks

def train_weights(truth_matrix, features):
    model = NMF(n_components=features, init='nndsvd')
    user_weights = model.fit_transform(truth_matrix.to_numpy())

    stock_weights = model.components_
    return user_weights, stock_weights

def train_new_matrices(
        user_weights_file_path,
        stock_weights_file_path
):
        # pull all user favorites from the site data base
        users = get_users()
        stocks = get_stocks()
        print(f"users len {len(users)}")
        # Take the user list, stock list, and create a truth matrix of all favorited stocks
        truth_matrix = build_truth_matrix(users, stocks)

        # take the sparse truth matrix and factorize it to create weight matrices
        # Matrix one is user prefrence weights for each catagory
        # Matrix two is stock type weights
        number_of_features = 10 # number of features to be factorized
        user_weights, stock_weights = train_weights(truth_matrix, number_of_features)

        # convert the numpy matrices to data frames for easier tracking and usage
        user_weights = pd.DataFrame(user_weights, index=users)
        user_weights_file_path = user_weights_file_path
        user_weights.to_csv(user_weights_file_path)
        stock_weights = pd.DataFrame(stock_weights, columns=stocks)
        stock_weights_file_path = stock_weights_file_path
        stock_weights.to_csv(stock_weights_file_path)
        return user_weights, stock_weights

def recommend_stocks(username, user_weight_path, stock_weight_path):
    print(user_weight_path)
    user_weights = pd.read_csv(user_weight_path, index_col=0)
    stock_weights = pd.read_csv(stock_weight_path, index_col=0)

    user_matrix_np = np.array([user_weights.loc[username,:].to_numpy()]) # get the row for the user weights
    predict = np.dot(user_matrix_np, stock_weights.to_numpy()) # multiple our user weights against stock weights
    predict = pd.DataFrame(predict, index=[username], columns=stock_weights.columns)
    highest_predictions = predict.loc[username,:].nlargest(50)
    print(f"{username} \n {highest_predictions}")
    user_favorites = get_user_favorites(username)
    recommended_stocks = []
    i = 0
    while len(recommended_stocks) < 12:
        print(highest_predictions.index[i])
        if not highest_predictions.index[i] in user_favorites:
            recommended_stocks.append(highest_predictions.index[i])
        i += 1
    return recommended_stocks

def main(args):
    if args.train:
        user_weights, stock_weights = train_new_matrices(args.user_weights, args.stock_weights)
    else:
        # Load the user and stock weight matrices from the stored csv file
        user_weights = pd.read_csv(args.user_weights, index_col=0)
        stock_weights = pd.read_csv(args.stock_weights, index_col=0)
        print(f"stock_weights {stock_weights}")
    logger.debug(f"user_weights shape: {user_weights.shape} stock_weights shape: {stock_weights.shape}")

    # predict all by performing a dot product of all user weights and stock weights
    predict = np.dot(user_weights.to_numpy(), stock_weights.to_numpy())
    predict = pd.DataFrame(predict, index=user_weights.index, columns=stock_weights.columns)
    logger.info(f"predict shape: {predict.shape}")
    logger.debug(f"predict \n {predict}")

    # predict single user by performing a dot product with their row data and the stock weight matrix
    user = args.user
    recommended_stocks = recommend_stocks(user, args.user_weights, args.stock_weights)
    print(f"recommended stocks for {user} \n{recommended_stocks}")

if __name__ == '__main__':
    args = input()
    main(args)
