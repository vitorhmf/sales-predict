

# Libs
import pandas as pd
import requests
import json


def load_dataset(store_id):
    # Loading test dataset
    df10 = pd.read_csv('../data/test.csv')
    df_store_raw = pd.read_csv('../data/store.csv')

    # merge test dataset + store
    df_test = pd.merge (df10, df_store_raw, how = 'left', on = 'Store')

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    # remove closed days
    df_test = df_test[df_test['Open'] != 0]
    df_test = df_test[df_test['Open'].notnull()]
    df_test = df_test.drop('Id', axis = 1)

    # convert Dataframe to json
    data = json.dumps (df_test.to_dict(orient='records'))

    return data

def predict (data):

    # API Call
    url = 'https://vitorhmf-rossmann-model.herokuapp.com/rossmann/predict'
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data=data, headers = header)
    print ('Status Code {}'.format(r.status_code))

    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

    return d1

# d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

# for i in range(len (d2)):
#     print('Store Number {} will sell R${:,.2f} in the next 6 weeks '.format(
#         d2.loc[i, 'store'],
#         d2.loc[i, 'prediction']))