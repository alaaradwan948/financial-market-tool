import streamlit as st
import pandas as pd
import yfinance as yf
import talib
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from transformers import pipeline
import os

st.title("أداة تحليل الأسواق المالية")

# اختيار الأصل المالي والتواريخ
ticker = st.text_input("رمز الأصل (مثال: AAPL):", "AAPL")
start_date = st.date_input("تاريخ البدء:", pd.to_datetime("2023-01-01"))
end_date = st.date_input("تاريخ النهاية:", pd.to_datetime("today"))

# تحميل البيانات مع استخدام التخزين المؤقت لتقليل التحميل المتكرر
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if not data.empty:
    st.subheader("البيانات المالية:")
    # إمكانية تحديد عدد الصفوف المراد عرضها
    rows = st.slider("حدد عدد الصفوف للعرض:", min_value=5, max_value=50, value=10)
    st.dataframe(data.tail(rows))
    
    # حساب المؤشرات الفنية وتحليل أنماط الشموع اليابانية
    try:
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
        
        # تحليل نمط الشموع (المطرقة والابتلاع)
        data['Hammer'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        data['Engulfing'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء حساب المؤشرات الفنية: {e}")

    # رسم مخطط شموع السعر باستخدام Plotly
    st.subheader("رسم تفصيلي للسعر:")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='السعر'
    ))
    st.plotly_chart(fig, use_container_width=True)

    # رسم مؤشر RSI
    st.subheader("مؤشرات فنية:")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # عرض الأنماط المكتشفة مثل نمط المطرقة ونمط الابتلاع
    st.subheader("الأنماط المكتشفة:")
    pattern_data = data[['Hammer', 'Engulfing']].dropna()
    st.write(pattern_data)

    # قسم توقع السعر باستخدام نموذج LSTM
    st.subheader("📊 تنبؤ LSTM بالسعر:")

    # إمكانية اختيار تدريب نموذج جديد أو استخدام النموذج المحفوظ مسبقاً
    train_model = st.checkbox("تدريب نموذج LSTM جديد")
    model_path = "lstm_model.h5"

    # إعداد بيانات الإغلاق لنموذج LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    sequence_len = 60  # مدة التسلسل المستخدمة

    for i in range(sequence_len, len(scaled_close)):
        X.append(scaled_close[i-sequence_len:i, 0])
        y.append(scaled_close[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # التحقق من خيار التدريب أو تحميل النموذج المحفوظ
    if train_model or not os.path.exists(model_path):
        # بناء نموذج LSTM وتدريبه
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        with st.spinner("⏳ جاري تدريب النموذج..."):
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        
        # حفظ النموذج المدرب للاستخدام في المرات القادمة
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # عمل التنبؤ بالسعر
    predicted_scaled = model.predict(X)
    predicted = scaler.inverse_transform(predicted_scaled)

    # رسم مخطط السعر المتوقع مقابل السعر الفعلي
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=data.index[sequence_len:], 
        y=predicted.flatten(), 
        name="السعر المتوقع"
    ))
    fig_pred.add_trace(go.Scatter(
        x=data.index[sequence_len:], 
        y=data['Close'][sequence_len:], 
        name="السعر الفعلي"
    ))
    st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.error("⚠️ لا توجد بيانات متاحة. يرجى التحقق من الرمز المالي والتواريخ المدخلة.")

if __name__ == "__main__":
    st.write("التطبيق يعمل!")
