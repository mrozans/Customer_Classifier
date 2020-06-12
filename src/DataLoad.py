import pandas as pd
import numpy as np

def load_products_data():
    products_data_types = {
        'product_id': 'int64',
        'product_name': 'object',
        'category_path': 'object',
        'price': 'float64'
    }

    products_df = pd.read_json('../data/products.jsonl', 'records', dtype=products_data_types, lines=True)
    categories = []
    for i in products_df['category_path']:
        category = i
        category = category.split(';')
        for j in category:
            if j not in categories:
                categories.append(j)

    for k in categories:
        products_df[k] = np.where(products_df['category_path'].str.contains(k), 1, 0)

    return products_df.drop('category_path', axis=1)


def load_sessions_data():
    sessions_data_types = {
        'session_id': 'int64',
        'timestamp': 'object',
        'user_id': 'int64',
        'product_id': 'int64',
        'event_type': 'object',
        'offered_discount': 'int64',
        'purchase_id': 'int64'
    }

    sessions_df = pd.read_json('../data/sessions.jsonl', 'records', dtype=sessions_data_types, lines=True)
    sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return pd.get_dummies(sessions_df, columns=['event_type'])


def load_users_data():
    users_data_types = {
        'user_id': 'int64',
        'name': 'object',
        'city': 'object',
        'street': 'object'
    }

    users_df = pd.read_json('../data/users.jsonl', 'records', dtype=users_data_types, lines=True)
    return pd.get_dummies(users_df, columns=['city'])

