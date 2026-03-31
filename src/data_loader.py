import pandas as pd
import numpy as np

def load_ratings(path):
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return ratings

def load_users(path):
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )
    return users

def load_items(path):
    items = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["item_id", "title", "genres"]
    )
    return items

def convert_to_implicit(ratings):
    ratings = ratings[ratings["rating"] >= 4]
    ratings = ratings[["user_id", "item_id"]]
    return ratings

def balance_users_by_gender(users):
    males = users[users["gender"] == "M"]
    females = users[users["gender"] == "F"]
    
    min_count = min(len(males), len(females))
    
    males_sampled = males.sample(n=min_count, random_state=42)
    females_sampled = females.sample(n=min_count, random_state=42)
    
    balanced_users = pd.concat([males_sampled, females_sampled])
    return balanced_users

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

def get_item_category(items, item_mapping):
    item_category = {}

    for original_id, new_id in item_mapping.items():
        genres = items[items["item_id"] == original_id]["genres"].values[0]

        # take all genres
        categories = genres.split("|")

        item_category[new_id] = categories

    return item_category

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

def train_val_test_split_userwise(ratings, val_ratio=0.1, test_ratio=0.1):

    train_list = []
    val_list = []
    test_list = []

    grouped = ratings.groupby("user_id")

    for user, group in grouped:

        if len(group) < 3:
            continue

        n_test = max(1, int(len(group) * test_ratio))
        n_val = max(1, int(len(group) * val_ratio))

        if n_test + n_val >= len(group):
            n_test = 1
            n_val = 1

        test = group.sample(n=n_test, random_state=42)
        rem = group.drop(test.index)

        val = rem.sample(n=n_val, random_state=42)
        train = rem.drop(val.index)

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    test_df = pd.concat(test_list)

    return train_df, val_df, test_df

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
    items = load_items("data/ml-1m/movies.dat")

    # Balance users by gender
    users = balance_users_by_gender(users)

    ratings = ratings[ratings["user_id"].isin(users["user_id"])]

    ratings = convert_to_implicit(ratings)

    ratings, user_mapping, item_mapping = encode_ids(ratings)

    gender_dict = get_user_gender(users, user_mapping)
    item_category = get_item_category(items, item_mapping)

    train_df, val_df, test_df = train_val_test_split_userwise(ratings)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    train_interactions = list(zip(train_df["user_id"], train_df["item_id"]))
    val_interactions = list(zip(val_df["user_id"], val_df["item_id"]))
    test_interactions = list(zip(test_df["user_id"], test_df["item_id"]))

    return (
        train_interactions,
        val_interactions,
        test_interactions,
        gender_dict,
        num_users,
        num_items,
        item_category
    )

