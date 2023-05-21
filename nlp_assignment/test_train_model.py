#!/usr/bin/env python
import argparse
from datetime import datetime, timedelta
import sqlite3


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
    # pull data from data base
    # build test/train dataset
    # build a model
    # train the model
    # output results 
    # save the model

if __name__ == '__main__':
    args = input()
    main(args)
