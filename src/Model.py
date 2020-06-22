import pandas as pd
import numpy as np
import copy
from scipy.special import softmax
import DataLoad as data
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

def split_data(sessions, day_to_split= '2020-3-12'):
    test = sessions_df.loc[sessions_df['timestamp'] > '2020-3-12']
    train = sessions_df.loc[sessions_df['timestamp'] <= '2020-3-12']
    return train, test


class ClassificationModel:

    def __init__(self, n_estimators=20, max_depth=5, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth,
                                          random_state=self.random_state)

    def train(self, users_df, sessions, products):
        users = copy.deepcopy(users_df)
        users = data.favourite_products(users, sessions, products)
        users = data.spendings(users, sessions, products)
        users = data.discounts_stats(users, sessions)
        users = data.discounts_label(users, sessions)
        users = users.set_index('user_id')
        users = users.drop(['name', 'city', 'street', ], axis=1)
        users = users.fillna(0)
        y_train = users['label']
        X_train = users.drop('label', axis=1)
        self.clf.fit(X_train, y_train)

    def predict(self, users_df, sessions, products):
        users = copy.deepcopy(users_df)
        users = data.favourite_products(users, sessions, products)
        users = data.spendings(users, sessions, products)
        users = data.discounts_stats(users, sessions)
        users = users.set_index('user_id')
        users = users.drop(['name', 'city', 'street', ], axis=1)
        users = users.fillna(0)
        return self.clf.predict(users)


class RegressionModel:
    def __init__(self, classification_model, train_sessions):
        self.reg_list = []
        self.selected_users = []
        self.classification_model = classification_model
        self.train_sessions = train_sessions

    def get_buys(self, uid, cleared_sessions_df):
        drop_list = []
        tmp = cleared_sessions_df.loc[
            (cleared_sessions_df['user_id'] == uid) & (cleared_sessions_df['event_type_BUY_PRODUCT'] == 1)]
        i = -1
        for index, row in tmp.iterrows():
            if index in drop_list:
                continue
            i = i + 1
            count = 1
            sum = row['offered_discount']
            j = i
            tmp2 = tmp[(i + 1):]
            for index2, row2 in tmp2.iterrows():
                j = j + 1
                if row['product_id'] == row2['product_id']:
                    count = count + 1
                    sum = sum + row2['offered_discount']
                    tmp.drop(tmp.index[j], inplace=True)
                    j = j - 1
                    drop_list.append(index2)
            if count > 1:
                tmp.at[index, 'offered_discount'] = sum / count
        return tmp

    def get_product_category(self, pid, products_df):
        tmp = products_df.loc[((products_df['product_id']).isin(pid))]
        tmp = tmp.drop(['product_name', 'price', 'product_id'], axis=1)
        return tmp

    def set_selected_users(self, users_df, test, products_df):
        i = -1
        selection = self.classification_model.predict(users_df, test, products_df)
        for user in users_df['user_id']:
            i = i + 1
            if selection[i] == 1:
                self.selected_users.append(user)

    def train(self, sessions_df, products_df):
        for id in self.selected_users:
            reg = linear_model.LinearRegression()
            user_buys = self.get_buys(id, sessions_df)
            if len(user_buys.index) > 0:
                reg.fit(self.get_product_category(user_buys['product_id'], products_df), user_buys['offered_discount'])
                self.reg_list.append([id, reg])
            else:
                self.reg_list.append([id, 0])

    def find(self, id):
        i = 0
        for user in self.selected_users:
            if id == user:
                return i
            i = i + 1
        return -1

    def predict(self, index):
        # row = sessions_df.loc[sessions_df.index == index]
        row = cleared_test_sessions_df.loc[cleared_test_sessions_df.index == index]
        id = self.find(int(row['user_id']))
        if id == -1:
            return 0
        if self.reg_list[id][1] == 0:
            return 0
        return self.reg_list[id][1].predict(self.get_product_category([row['product_id']]))

    def predict2(self, user_id, product_id):
        id = self.find(user_id)
        if id == -1:
            return 0
        if self.reg_list[id][1] == 0:
            return 0
        return self.reg_list[id][1].predict(self.get_product_category([product_id]))

