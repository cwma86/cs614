#!/usr/bin/env python
import argparse
from datetime import datetime, timedelta
import sqlite3


def inputargs():
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
def update_truth(cur, url, truth):
    command = f"UPDATE news SET truth = '{truth}' WHERE url = '{url}'"
    print(f"command {command}")
    cur.execute(command)
    pass

def delete_entry(cur, url):
    command = f"DELETE FROM news WHERE url = '{url}'"
    cur.execute(command)
    print(f"command {command}")
    pass

def main(args):
    con = sqlite3.connect("news.db")
    cur = con.cursor()
    res = cur.execute("SELECT * FROM news WHERE truth IS \"\" ORDER BY datepublished  ")
    results = res.fetchall()
    for result in results:
        stockname = result[0]
        title = result[2]
        description = result[3]
        content = result[4]
        url = result[5]
        print(f"stockname {stockname}")
        print(f"title {title}")
        print(f"description {description}")
        print(f"content {content}")
        print(f"url {url}")
        while True:
            user_input = input("provide truth label: \n\t P - Positive\n\t N - Negative\n\t D - Delete\n:")
            user_input = user_input.lower()
            print(f"user_input {user_input}")
            if user_input == "p":
                update_truth(cur, url, user_input)
                con.commit()
                break
            elif user_input == "n":
                update_truth(cur, url, user_input)
                con.commit()
                break
            elif user_input == "d":
                delete_entry(cur, url)
                con.commit()
                break
            else:
                print(f"invalid input")
        result = res.fetchone()
        print("\n\n\n")
    # TODO pull data from data base
    # ouput comanpy name, title, description, and url
    # ask the user to tag for truth (positive, negative, or neutral) or delete the entry

if __name__ == '__main__':
    args = inputargs()
    main(args)
