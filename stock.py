import pickle
import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from realtime import RealtimeData
from tensorflow.keras.models import load_model


def get_input():
    d = dt.timedelta(days=30)
    start_date = st.sidebar.text_input("Start Date", dt.date.today() - d)
    end_data = st.sidebar.text_input("End Date", dt.date.today())
    symbol = st.sidebar.selectbox('Choose stock to visualize and predict', stock_list)

    return start_date, end_data, symbol


def get_pred(model, scaler, input_data, days):
    index = list(input_data.index)
    index.reverse()
    pred_data = scaler.transform(input_data.filter(['close']).values)

    predicts = [np.nan for i in range(60)]
    x_test = []
    date = dt.date.today()
    for i in range(60, len(pred_data)):
        x_test.append(pred_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_pred = scaler.inverse_transform(model.predict(x_test))
    predicts.extend([y_pred[i, 0] for i in range(len(y_pred))])

    for i in range(days):
        x_test = np.array(pred_data[-60:]).reshape((1, 60, 1))
        y_pred = scaler.inverse_transform(model.predict(x_test))[0, 0]
        predicts.append(y_pred)
        index.append(pd.Timestamp(date.year, date.month, date.day, 00))
        date = date + dt.timedelta(days=1)

    df = pd.DataFrame({
        'Actual': historical['close'],
        'Predicted': predicts
    }, index=index)

    return df


stock_list = ['TSLA', 'AAPL', 'AMZN']
st.set_page_config(layout='wide')
col1, col2 = st.beta_columns(2)

col1.title('TB Stock Price Visualize and Predicting Site')
img = Image.open('images/frontpage.png')
col1.image(img, use_column_width=False)
start, end, symbol = get_input()
data = RealtimeData(symbol)

col1.markdown("""---""")
col1.markdown("""## {}'s Historical data """.format(symbol))
col1.markdown("""This section contains **1 year** of {}'s historical data.""".format(symbol))
historical = data.get_historical()
col1.dataframe(historical, 1000, 1500)
col1.markdown("""---""")

col2.title('')
col2.header('')
col2.markdown("""## {}'s Real time data """.format(symbol))
col2.markdown("""This section contains {}'s **real time** data with selected interval.""".format(symbol))
expander = col2.beta_expander(label='Display Options')
interval = expander.selectbox('Choose an interval', ('1day', '1min', '1week', '1month'))
col2.plotly_chart(data.get_data(start, end, interval).with_minmax().with_ema().as_plotly_figure(),
                  use_container_width=True)

col2.markdown("""---""")
col2.markdown("""## {}'s Predictive data""".format(symbol))
col2.markdown("""This section displays **predicted** {}'s price from toady to a selected range.""".format(symbol))
days = int(col2.slider('Days to predict', min_value=1, max_value=10, value=5))

model = load_model('models/{}/model.h5'.format(symbol))
scaler = pickle.load(open('models/{}/scaler.pkl'.format(symbol), 'rb'))

result = get_pred(model, scaler, historical, days)

fig = go.Figure()
fig.add_trace(go.Scatter(x=result.index, y=result['Actual'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=result.index, y=result['Predicted'], mode='lines', name='Predicted Price'))
col2.plotly_chart(fig, use_container_width=True)
col2.markdown("""---""")


