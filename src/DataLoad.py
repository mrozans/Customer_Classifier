import pandas as pd
import numpy as np
from scipy.special import softmax


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
    return users_df


def user_buys(uid, sessions_df):
    return len(sessions_df.loc[(sessions_df['user_id'] == uid) & (sessions_df['event_type_BUY_PRODUCT'] == 1)].index)


def user_visits(uid, sessions_df):
    return len(sessions_df.loc[sessions_df['user_id'] == uid].index)


def mean_accepted_discount(uid, sessions_df):
    return sessions_df.loc[(sessions_df['user_id'] == uid) & (sessions_df['event_type_BUY_PRODUCT']
                                                              == 1)]['offered_discount'].mean()


def mean_rejected_discounts(uid, sessions_df):
    return sessions_df.loc[(sessions_df['user_id'] == uid) & (sessions_df['event_type_BUY_PRODUCT']
                                                              == 0)]['offered_discount'].mean()
															  
def fav_categories(uid, sessions_df, products_df):
    products_id = sessions_df.loc[sessions_df['user_id'] == uid]['product_id']
    fav = products_df.loc[products_df['product_id'].isin(products_id.tolist())]
    fav = fav[fav.columns[3:]]
    return fav.sum()

def favourite_products(users_df, sessions_df, products_df):
    arr = np.zeros(shape =(len(users_df['user_id']), 28))
    j = 0
    for i in users_df['user_id']:
        arr[j] = softmax(fav_categories(i, sessions_df, products_df))
        j += 1
    fv =  fav_categories(102, sessions_df, products_df)
    for k in range(len(fv.index.tolist())):
        users_df[fv.index.tolist()[k]] = arr[:, k]
    return users_df
	
def spendings(users_df, sessions_df, products_df):
    spendings = []
    for u_id in users_df['user_id']:
        products_id = sessions_df.loc[(sessions_df['user_id'] == u_id)
                                & (sessions_df['event_type_BUY_PRODUCT'] == 1)][['product_id', 'offered_discount']].to_numpy()       
        spend = 0
        for i in range(products_id.shape[0]):
            if(np.isnan(products_id[i,0])):
                continue
            price = products_df.loc[products_df['product_id'] 
                                 == products_id[i, 0]]['price'].to_numpy()[0] * (100 - products_id[i, 1])/100
            if(not np.isnan(price)):
                spend += price
        spendings.append(spend)
        
    users_df['spendings'] = spendings
    return users_df