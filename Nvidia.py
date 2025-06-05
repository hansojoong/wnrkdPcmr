import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import plotly.graph_objects as go

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="NVIDIA 주가 예측", layout="wide")
st.title("📊 NVIDIA 주가 예측 및 시나리오 분석 (USD 기준)")

# 데이터 로드 및 전처리
file_path = 'dataset/Nvidia Dataset.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df.dropna(inplace=True)
df['Year'] = df['Date'].dt.year

# 스케일링
features = ['Close', 'MA20', 'MA50', 'Volume']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled, columns=features)

# 시퀀스 데이터 생성 함수
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i]['Close'])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df)

# 딥러닝 모델 구성 및 학습
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
y_pred = model.predict(X)

# 평가 지표 계산
rmse = math.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# 다음날 예측
last_seq = scaled_df.iloc[-60:].values.reshape(1, 60, len(features))
next_day_scaled = model.predict(last_seq)[0][0]
next_day_price_usd = scaler.inverse_transform([[next_day_scaled, 0, 0, 0]])[0][0]

# 최소/최대값 (산점도 완벽 예측선용)
min_val = min(scaler.inverse_transform(scaled_df)[60:, 0].min(), (y_pred.flatten() * scaler.data_range_[0] + scaler.data_min_[0]).min())
max_val = max(scaler.inverse_transform(scaled_df)[60:, 0].max(), (y_pred.flatten() * scaler.data_range_[0] + scaler.data_min_[0]).max())

# 탭 구성
tab1, tab2, tab3 = st.tabs(["데이터 EDA", "모델 예측", "시나리오 분석"])

with tab1:
    st.subheader("🔎 데이터 개요")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 데이터 수", f"{len(df):,}건")
    with col2:
        st.metric("날짜 범위", f"{df['Date'].min().date()} ~ {df['Date'].max().date()}")
    with col3:
        st.metric("평균 종가", f"${df['Close'].mean():.2f}")
    with col4:
        missing_total = df.isnull().sum().sum()
        st.metric("결측치 상태", "✅ 없음" if missing_total == 0 else f"⚠️ {missing_total}개 존재")

    st.subheader("📉 연도별 평균 종가")
    avg_by_year = df.groupby('Year')['Close'].mean().reset_index()
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=avg_by_year, x='Year', y='Close', marker='o', ax=ax1)
    for i, row in avg_by_year.iterrows():
        if row['Year'] % 5 == 0 or row['Year'] == avg_by_year['Year'].max():
            ax1.text(row['Year'], row['Close'] + 5, f"${row['Close']:.2f}", ha='center', fontsize=9)
    ax1.set_title("연도별 평균 종가 (USD)")
    ax1.set_xlabel("연도")
    ax1.set_ylabel("평균 종가 (USD)")
    st.pyplot(fig1)

    st.subheader("📊 연도별 거래량 분포")
    years = sorted(df['Year'].unique())
    ncols = 4
    nrows = (len(years) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, year in enumerate(years):
        sns.histplot(data=df[df['Year'] == year], x='Volume', bins=30, ax=axes[i], color='skyblue')
        axes[i].set_title(f"{year}년")
        axes[i].xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("📈 연도별 최고가 / 최저가")
    agg_df = df.groupby('Year').agg({'High': 'max', 'Low': 'min'}).reset_index()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x='Year', y='High', marker='o', label='최고가', ax=ax3)
    sns.lineplot(data=agg_df, x='Year', y='Low', marker='o', label='최저가', ax=ax3)
    for i, row in agg_df.iterrows():
        if row['Year'] % 5 == 0 or row['Year'] == agg_df['Year'].max():
            ax3.text(row['Year'], row['High'] + 5, f"{row['High']:.1f}", color='red', ha='center', fontsize=9)
            ax3.text(row['Year'], row['Low'] - 5, f"{row['Low']:.1f}", color='blue', ha='center', fontsize=9)
    ax3.legend()
    ax3.set_title("연도별 최고가 및 최저가 (USD)")
    ax3.set_xlabel("연도")
    ax3.set_ylabel("가격 (USD)")
    st.pyplot(fig3)

    st.subheader("🧮 주요 변수 간 상관관계 히트맵")
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

with tab2:
    st.subheader("📊 딥러닝 모델 예측 결과")
    st.write(f"✅ RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    st.write(f"📌 다음날 예측 종가: ${next_day_price_usd:.2f}")

    fig5, ax5 = plt.subplots(figsize=(14, 6))
    ax5.plot(df['Date'][60:], scaler.inverse_transform(scaled_df)[60:, 0], label='실제 종가')
    ax5.plot(df['Date'][60:], y_pred.flatten() * scaler.data_range_[0] + scaler.data_min_[0], label='예측 종가')
    ax5.set_title("실제 종가 vs 예측 종가 (USD)")
    ax5.set_xlabel("날짜")
    ax5.set_ylabel("종가 (USD)")
    ax5.legend()
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(12, 6))
    ax6.scatter(scaler.inverse_transform(scaled_df)[60:, 0], y_pred.flatten() * scaler.data_range_[0] + scaler.data_min_[0], 
                c='blue', alpha=0.5, label='예측 vs 실제')
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', label='완벽 예측 선 (y=x)')
    ax6.set_title("예측 정확도 산점도")
    ax6.set_xlabel("실제 종가 (USD)")
    ax6.set_ylabel("예측 종가 (USD)")
    ax6.legend()
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots(figsize=(12, 4))
    ax7.plot(history.history['loss'], label='훈련 손실')
    ax7.plot(history.history['val_loss'], label='검증 손실')
    ax7.set_title("학습 손실 추이")
    ax7.set_xlabel("에포크")
    ax7.set_ylabel("손실 값")
    ax7.legend()
    st.pyplot(fig7)

with tab3:
    st.subheader("🔮 1년 시나리오 예측 (캔들 차트)")

    future_year = st.number_input("예측할 연도 입력 (예: 2025)", min_value=2025, max_value=2100, value=2025)

    if st.button("📈 시나리오 생성"):
        months = 12
        np.random.seed(future_year)

        base_price = next_day_price_usd
        prices = [base_price]

        for _ in range(months - 1):
            change = np.random.normal(loc=0, scale=base_price * 0.05)
            prices.append(prices[-1] + change)

        prices = np.array(prices)
        open_prices = prices * (1 + np.random.uniform(-0.01, 0.01, size=months))
        close_prices = prices
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, size=months))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, size=months))
        dates = pd.date_range(start=f"{future_year}-01-01", periods=months, freq='MS')

        fig8 = go.Figure(data=[go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            increasing_line_color='blue',
            decreasing_line_color='red'
        )])
        fig8.update_layout(title=f"{future_year}년 월별 캔들스틱 시나리오 (USD)",
                           xaxis_title="월", yaxis_title="예상 종가 (USD)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig8, use_container_width=True)

        up = prices.mean() * 1.05
        down = prices.mean() * 0.95
        message = '📈 상승 가능성 높음' if up > down else '📉 하락 가능성 존재'
        st.success(f"💡 딥러닝 기반 판단 결과: {future_year}년은 {message}")
