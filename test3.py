import streamlit as st
import pandas as pd
import yfinance as yf
import talib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from transformers import pipeline
import plotly.graph_objects as go

# واجهة الأداة
st.title("أداة تحليل الأسواق المالية")

# اختيار الأصل المالي
ticker = st.text_input("رمز الأصل (مثال: AAPL):", "AAPL")
start_date = st.date_input("تاريخ البدء:", pd.to_datetime("2023-01-01"))
end_date = st.date_input("تاريخ النهاية:", pd.to_datetime("today"))

# تحميل البيانات
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if not data.empty:
    st.subheader("البيانات المالية:")
    st.dataframe(data.tail())
    
    try :
      # إضافة مؤشرات فنية
      data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
      data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
      data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
    
      # تحليل نمط الشموع اليابانية
      data['Hammer'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
      data['Engulfing'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
    
    except Exception as e :
      print(f"خطأ في  استخدام TA-Lib: {e}")


    # نموذج LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # عرض الرسم البياني
    st.subheader("رسم تفصيلي للسعر:")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    st.plotly_chart(fig)

    # عرض المؤشرات
    st.subheader("مؤشرات فنية:")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    st.plotly_chart(fig_rsi)

    # كشف الأنماط
    st.subheader("الأنماط المكتشفة:")
    pattern_data = data[['Hammer', 'Engulfing']].dropna()
    st.write(pattern_data)

    # تنبؤات LSTM
   # --- إعداد البيانات لـ LSTM ---

   # تطبيع بيانات الإغلاق
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

   # إعداد بيانات التسلسل
    X, y = [], []
    sequence_len = 60

    for i in range(sequence_len, len(scaled_close)):
        X.append(scaled_close[i-sequence_len:i, 0])
        y.append(scaled_close[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # بناء النموذج (أو تحميله لو كان متدرب)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # تدريب بسيط (قليل لتجربة سريعة)
    with st.spinner("⏳ جاري تدريب النموذج..."):
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # عمل تنبؤ
    predicted_scaled = model.predict(X)
    predicted = scaler.inverse_transform(predicted_scaled)

    # رسم التنبؤ
    st.subheader("📊 تنبؤ LSTM بالسعر:")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=data.index[sequence_len:], y=predicted.flatten(), name="Predicted"))
    fig_pred.add_trace(go.Scatter(x=data.index[sequence_len:], y=data['Close'][sequence_len:], name="Actual"))
    st.plotly_chart(fig_pred, use_container_width=True)


# تشغيل الأداة
if __name__ == "__main__":
    st.write("التطبيق يعمل!")