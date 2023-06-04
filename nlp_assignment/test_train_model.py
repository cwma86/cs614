#!/usr/bin/env python
import argparse
from datetime import datetime, timedelta
import math
import numpy as np
import os
import sqlite3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help'
    )
    parser.add_argument(
        '--todo',
        type=str,
        default=None,
        help="todo"
    )

    args = parser.parse_args()

    return args

def main(args):
    con = sqlite3.connect("news.db")
    cur = con.cursor()
    results = cur.execute("SELECT * FROM news WHERE truth IS NOT \"\" ORDER BY datepublished  ")
    number_words = 7000
    tokenizer = Tokenizer(num_words=number_words, oov_token='<UNK>')
    text = []
    label = []
    for result in results:
        title_desc = result[2] + " " + result[3]
        truth = result[6]
        if truth == "n":
            truth = 0 # negavtive story
        else:
            truth = 1 # postive story
        text.append(title_desc)
        label.append(truth)
    label = np.array(label)

    # Check and fix dataset balance
    label_counts = np.unique(label, return_counts=True)
    print(f"label counts \n\t {label_counts}")
    if label_counts[1][0] * 1.5 < label_counts[1][1]:
        print("unbalanced data set, duplicate negative case data to balance")
        for i in range(math.floor(len(label)/2)):
            if label[i] == 0:
                label = np.append(label, label[i])
                text.append(text[i])
    label_counts = np.unique(label, return_counts=True)
    print(f"label counts \n\t {label_counts}")

    # Tokenize article text
    tokenizer.fit_on_texts(text)
    print(f"text[0] {text[2]} {label[2] }")
    text_token = tokenizer.texts_to_sequences(text)
    maxlen=20
    text_token = pad_sequences(text_token, truncating = 'post', padding='post', maxlen=maxlen)

    # Combine token and label
    print(f"text_token {text_token}")
    print(text_token.shape)
    label = np.array([label]).T
    print(label.shape)
    labeled_dataset = np.hstack((text_token, label))
    print(labeled_dataset.shape)
    print(labeled_dataset[:,-1])
    index = range(labeled_dataset.shape[0])
    
    # add index so we can refrence the data set text after shuffle
    index = np.array([index]).T
    print(index.shape)
    labeled_dataset = np.hstack((labeled_dataset, index))
    print(labeled_dataset.shape)

    # Shuffle the data to randomize the test/train split
    np.random.shuffle(labeled_dataset)
    training_data_length = math.floor(labeled_dataset.shape[0]* 0.8)
    print(training_data_length)
    training_data = labeled_dataset[:training_data_length]
    X_train = training_data[:,:-2]
    Y_train = training_data[:,-2]
    print(f"training_data shape {X_train.shape}, test_data {Y_train.shape}")
    test_data = labeled_dataset[training_data_length:]
    X_test = test_data[:,:-2]
    Y_test = test_data[:,-2]

    # pull data from data base
    # build test/train dataset
    # build a model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(number_words,16,input_length=maxlen),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        metrics=['accuracy']
    )
    # train the model
    h = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=40,

        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)]
    )

    # output results 
    model.evaluate(X_test, Y_test)
    test = []
    test.append("NVIDIA today reported revenue for the first quarter ended April 30, 2023, of $7.19 billion, down 13\% from a year ago and up 19\% from the...")
    test.append( "GameStop (GME) Gains As Market Dips: What You Should Know") #bad
    test.append("General Electric (NYSE:GE) shareholders have earned a 36% CAGR over the last three years") #bad
    test.append("Berkshire Hathaway Reports Major Investment Losses in 2022")
    test.append("Retail investors are sitting on heavy losses despite a 2023 stock rally")
    test.append("Nvidia stock surge could signal the start of an AI bubble")
    test.append("Nvidia gains $185bn in value after predicting AI-driven boom in chip demand")
    test_token = tokenizer.texts_to_sequences(test)
    test_token = pad_sequences(test_token, truncating = 'post', padding='post', maxlen=maxlen)
    preds = model.predict(test_token)
    for i in range(len(test)):
        print(f"Story: \n\t {test[i]}")
        print(f"prediction [negative positive]\n{preds[i]}")
    # save the model

if __name__ == '__main__':
    args = input()
    main(args)
