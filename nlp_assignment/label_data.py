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
    # TODO pull data from data base
    # ouput comanpy name, title, description, and url
    # ask the user to tag for truth (positive, negative, or neutral) or delete the entry

if __name__ == '__main__':
    args = input()
    main(args)
