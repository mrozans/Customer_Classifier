import pandas as pd
import numpy as np
import DataLoad as data

class BaseModel:
    def __init__(self):
        self.discounts = {}

    def train(self, data, sessions):
        for i in data['product_id'].unique():
            if np.isfinite(i):
                discounts = sessions.loc[(sessions['product_id'] == i)
                                         & (sessions['event_type_BUY_PRODUCT'] == 1)]['offered_discount']
                if len(discounts) != 0:
                    self.discounts[i] = np.average(discounts)
                else:
                    self.discounts[i] = np.random.randint(21)

    def predict(self, product_id):
        try:
            return self.discounts[product_id]
        except:
            return 0