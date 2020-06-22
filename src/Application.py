import BaseModel
import Model
import DataLoad
import sys
if __name__ == "__main__":

    a_b = False
    if len(sys.argv) > 1 and sys.argv[1] == '-A/B':
        a_b = True
        print('A/B test')

    users_df = DataLoad.load_users_data()
    sessions_df = DataLoad.load_sessions_data()
    sessions_df = sessions_df.loc[(~sessions_df['user_id'].isna()) | (~sessions_df['product_id'].isna())]
    products_df = DataLoad.load_products_data()

    classification_model = Model.ClassificationModel()
    classification_model.train(users_df, sessions_df, products_df)

    sessions_df = sessions_df.loc[(~sessions_df['user_id'].isna()) & (~sessions_df['product_id'].isna())]
    model = Model.RegressionModel(classification_model, sessions_df)
    model.set_selected_users(users_df, sessions_df, products_df)
    model.train(sessions_df, products_df)

    base_model = BaseModel.BaseModel()
    base_model.train(products_df, sessions_df)

    print('Model ready')

    while(True):
        user_id = int(input())
        product_id = int(input())
        if not a_b or user_id % 2 == 0:
            print(model.predict2(user_id, product_id, products_df))
        else:
            print(base_model.predict(product_id))