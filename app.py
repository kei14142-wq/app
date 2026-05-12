import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# Session State Initialization
# ==========================================
if 'daily_run_id' not in st.session_state:
    st.session_state['daily_run_id'] = 0
if 'annual_run_id' not in st.session_state:
    st.session_state['annual_run_id'] = 0
if 'vgi_result' not in st.session_state:
    st.session_state['vgi_result'] = None

# ==========================================
# 1. Data Processing & Profiling Functions
# ==========================================

@st.cache_data
def generate_university_sample():
    """大学の利用パターンを模した、ID付きの1年分擬似データ"""
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    data = []
    id_pool = np.arange(10001, 10501)
    
    for d in range(365):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        is_weekend = current_date.weekday() >= 5
        
        if is_weekend: 
            n_cars = np.random.randint(5, 15)
        elif month in [8, 9, 2, 3]: 
            n_cars = np.random.randint(30, 60)
        else: 
            n_cars = np.random.randint(80, 150)
        
        if n_cars == 0: continue
            
        in_hours = np.concatenate([
            np.random.normal(9.0, 0.8, int(n_cars * 0.6)),
            np.random.normal(13.5, 1.2, n_cars - int(n_cars * 0.6))
        ])
        in_hours = np.clip(in_hours, 0, 23.9)
        stay_durations = np.clip(np.random.normal(6.0, 2.0, n_cars), 1.0, 12.0)
        selected_ids = np.random.choice(id_pool, n_cars, replace=False)
        
        for i in range(n_cars):
            h, m = divmod(int(in_hours[i] * 60), 60)
            in_t = current_date + timedelta(hours=h, minutes=m)
            data.append({
                "car_id": int(selected_ids[i]),
                "in_time": in_t,
                "out_time": in_t + timedelta(hours=stay_durations[i]),
                "stay_duration": float(stay_durations[i])
            })
    return pd.DataFrame(data)

@st.cache_data
def process_and_profile_data(df):
    """実データから統計プロファイルを抽出"""
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['date'] = df['in_time'].dt.date
    df['month'] = df['in_time'].dt.month
    df['weekday'] = df['in_time'].dt.weekday
    df['in_hour'] = df['in_time'].dt.hour
    
    days_count = df.groupby(['month', 'weekday'])['date'].nunique().reset_index(name='num_days')
    arrivals = df.groupby(['month', 'weekday', 'in_hour']).size().reset_index(name='total_cars')
    profile_arrival = pd.merge(arrivals, days_count, on=['month', 'weekday'])
    profile_arrival['avg_cars'] = profile_arrival['total_cars'] / profile_arrival['num_days']
    
    profile_stay = df.groupby(['weekday', 'in_hour'])['stay_duration'].agg(['mean', 'std']).reset_index()
    profile_stay['std'] = profile_stay['std'].fillna(1.0)
    base_capacity = df.groupby('date').size().quantile(0.95) if not df.empty else 100
    
    return df, profile_arrival, profile_stay, base_capacity

def generate_vgi_logs(df_sim):
    """in/out/homeの1時間粒度ログへ変換"""
    logs = []
    for _, row in df_sim.iterrows():
        day_start = row['in_time'].replace(hour=0, minute=0, second=0, microsecond=0)
        in_h, out_h = row['in_time'].hour, row['out_time'].hour
        for h in range(24):
            dt = day_start + timedelta(hours=h)
            status = "in" if in_h <= h < out_h else "out" if h == out_h else "home"
            logs.append({"datetime": dt, "car_id": row['car_id'], "in_out": status})
    return pd.DataFrame(logs)

# ==========================================
# 2. UI Layout
# ==========================================
st.set_page_config(layout="wide", page_title="Parking Intelligence Pro")
st.title("駐車場分析 & VGIシミュレーター")

with st.sidebar:
    st.header("Data Source")
    data_mode = st.radio("Source", ["Sample University", "Upload CSV"])
    if data_mode == "Upload CSV":
        up = st.file_uploader("Upload CSV", type=['csv'])
        if not up: st.stop()
        raw_df = pd.read_csv(up)
    else:
        raw_df = generate_university_sample()

df, p_arr, p_stay, base_cap = process_and_profile_data(raw_df)
available_dates = sorted(df['date'].unique())

tab1, tab2, tab3 = st.tabs(["📊 分析トレンド", "⚙️ VGIシミュレーション", "📂 データ出力"])

# --- Tab 1: 分析 ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("曜日別平均流入数")
        wd_avg = df.groupby('weekday').size() / df.groupby('weekday')['date'].nunique()
        st.plotly_chart(px.bar(wd_avg), use_container_width=True)
    with col2:
        st.subheader("滞在時間の分布")
        st.plotly_chart(px.histogram(df, x="stay_duration"), use_container_width=True)

# --- Tab 2: VGI生成 ---
with tab2:
    st.subheader("1時間粒度シミュレーション")
    c1, c2, c3 = st.columns([2,2,1])
    with c1: sim_dt = st.selectbox("対象日を選択", available_dates)
    with c2: target_c = st.number_input("目標キャパシティ", 10, 2000, int(base_cap))
    with c3:
        st.write(" ")
        if st.button("実行"): st.session_state.daily_run_id += 1

    # シミュレーション実行ロジック
    multiplier = target_c / base_cap if base_cap > 0 else 1.0
    w_idx, month = sim_dt.weekday(), sim_dt.month
    day_arr = p_arr[(p_arr['month'] == month) & (p_arr['weekday'] == w_idx)]
    
    sim_results = []
    for hour in range(24):
        avg = day_arr[day_arr['in_hour'] == hour]['avg_cars'].sum() * multiplier
        n = np.random.poisson(avg)
        stay_info = p_stay[(p_stay['weekday'] == w_idx) & (p_stay['in_hour'] == hour)]
        s_mean = stay_info['mean'].values[0] if not stay_info.empty else 6.0
        for i in range(n):
            in_t = datetime.combine(sim_dt, datetime.min.time()) + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            sim_results.append({
                "car_id": 20000 + i + (hour * 100),
                "in_time": in_t,
                "out_time": in_t + timedelta(hours=s_mean)
            })
    
    if sim_results:
        sim_df = pd.DataFrame(sim_results)
        vgi_logs = generate_vgi_logs(sim_df)
        st.write("VGIログプレビュー (in/out/home)")
        st.dataframe(vgi_logs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vgi_logs[vgi_logs['in_out'] != 'home'].groupby(['datetime', 'in_out']).size().reset_index(name='count'), 
                               x='datetime', y='count', color='in_out', barmode='group'))

# --- Tab 3: 出力 ---
with tab3:
    st.subheader("データエクスポート")
    st.download_button("分析用データを保存", df.to_csv(index=False), "parking_data.csv")
