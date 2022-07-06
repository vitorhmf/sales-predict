# Libs
import pandas as pd
import requests
import json
import os
from flask import Flask, request, Response


# Info about the Bot
# https://api.telegram.org/bot5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM/getMe

# Get Updates (recebe a msg que sai do telefone)
# https://api.telegram.org/bot5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM/getUpdates

# Webhook (linka a url do telegram para um endpoint)
# https://api.telegram.org/bot5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM/setWebhook?url=https://d68271bca9db03.lhrtunnel.link

# Webhook Heroku
# https://api.telegram.org/bot5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM/setWebhook?url=https://vitorhmf-rossmann-bot.herokuapp.com/

# Send Message (envia a msg para o telegram)
# https://api.telegram.org/bot5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM/sendMessage?chat_id=1024228373&text=Hi Vitor, I am doing good, tks!


# Constants
token = '5380428769:AAENVi9-PEHB_KXxds0V8N_KIHFhLE1RrxM'

# Functions
def send_message (chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(token)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    r = requests.post(url, json={'text': text})  
    print ('Status Code{}'.format(r.status_code))

    return None
   
def load_dataset(store_id):
    # Loading test dataset
    df10 = pd.read_csv('test.csv')
    df_store_raw = pd.read_csv('store.csv')

    # merge test dataset + store
    df_test = pd.merge (df10, df_store_raw, how = 'left', on = 'Store')

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[df_test['Open'].notnull()]
        df_test = df_test.drop('Id', axis = 1)

        # convert Dataframe to json
        data = json.dumps (df_test.to_dict(orient='records'))

    else:
        data = 'error'

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

def parse_message (message):  
    chat_id     = message['message']['chat']['id']
    store_id    = message['message']['text']
    store_id    = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    
    except ValueError:
        store_id = 'error'

    return chat_id, store_id


# API initialize

app = Flask(__name__)
@app.route ('/', methods=['GET', 'POST'])               


def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            # loading data
            data = load_dataset(store_id)

            if data != 'error':
                # prediction
                d1 = predict(data)

                # calculation
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

                # send message
                msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks '.format(
                        d2['store'].values[0],
                        d2['prediction'].values[0])

                send_message (chat_id, msg)
                return Response ('Ok', status=200)

            else: 
                send_message (chat_id, 'Store Not Avaiable')
                return Response ('Ok', status=200)

        else:
            send_message(chat_id, 'Store ID is Wrong')
            return Response ('Ok', status=200)

    else:
        return '<h1> Rossmann Telegram BOT </h1>'


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run (host='0.0.0.0', port=port )