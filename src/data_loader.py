import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_ratings(path):
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return ratings

def load_users(path):
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )
    return users

def convert_to_implicit(ratings):
    ratings = ratings[ratings["rating"] >= 4]
    ratings = ratings[["user_id", "item_id"]]
    return ratings

def encode_ids(ratings):
    user_mapping = {id: idx for idx, id in enumerate(ratings["user_id"].unique())}
    item_mapping = {id: idx for idx, id in enumerate(ratings["item_id"].unique())}

    ratings["user_id"] = ratings["user_id"].map(user_mapping)
    ratings["item_id"] = ratings["item_id"].map(item_mapping)

    return ratings, user_mapping, item_mapping

def get_user_gender(users, user_mapping):
    gender_dict = {}

    for original_id, new_id in user_mapping.items():
        gender = users[users["user_id"] == original_id]["gender"].values[0]
        gender_dict[new_id] = 1 if gender == "F" else 0

    return gender_dict

# def train_test_split_userwise(ratings, test_size=0.2):
#     train_list = []
#     test_list = []

#     grouped = ratings.groupby("user_id")

#     for user, group in grouped:
#         train, test = train_test_split(group, test_size=test_size)
#         train_list.append(train)
#         test_list.append(test)

#     train_df = pd.concat(train_list)
#     test_df = pd.concat(test_list)

#     return train_df, test_df

def train_test_split_userwise(ratings, test_size=0.2):

    train_list = []
    test_list = []

    grouped = ratings.groupby("user_id")

    for user, group in grouped:

        if len(group) < 2:
            # skip users with only 1 interaction
            continue

        test_size_user = max(1, int(len(group) * test_size))

        test = group.sample(n=test_size_user, random_state=42)
        train = group.drop(test.index)

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df

# def train_test_split_userwise(ratings):

#     train_list = []
#     test_list = []

#     grouped = ratings.groupby("user_id")

#     for user, group in grouped:

#         if len(group) < 2:
#             continue

#         # Sort by timestamp if available
#         group = group.sort_values("item_id")

#         test = group.iloc[-1:]
#         train = group.iloc[:-1]

#         train_list.append(train)
#         test_list.append(test)

#     train_df = pd.concat(train_list)
#     test_df = pd.concat(test_list)

#     return train_df, test_df

def prepare_data():

    ratings = load_ratings("data/ml-1m/ratings.dat")
    users = load_users("data/ml-1m/users.dat")

    ratings = convert_to_implicit(ratings)

    ratings, user_mapping, item_mapping = encode_ids(ratings)

    gender_dict = get_user_gender(users, user_mapping)

    train_df, test_df = train_test_split_userwise(ratings)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    train_interactions = list(zip(train_df["user_id"], train_df["item_id"]))
    test_interactions = list(zip(test_df["user_id"], test_df["item_id"]))

    return (
        train_interactions,
        test_interactions,
        gender_dict,
        num_users,
        num_items
    )

