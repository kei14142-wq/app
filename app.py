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

# ==========================================
# 1. Data Processing & Profiling Functions
# ==========================================

@st.cache_data
def generate_university_sample():
    """大学の利用パターンを模したサンプルデータ（イベントベース）"""
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    days = 365
    data = []
    
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        is_weekend = current_date.weekday() >= 5
        n_cars = np.random.randint(5, 15) if is_weekend else (np.random.randint(30, 60) if month in [8, 9, 2, 3] else np.random.randint(80, 150))
        
        in_hours = np.concatenate([np.random.normal(9.0, 1.0, int(n_cars * 0.6)), np.random.normal(13.5, 1.5, n_cars - int(n_cars * 0.6))])
        in_hours = np.clip(in_hours, 1, 22.0)
        stay_durations = np.clip(np.random.normal(5.0, 2.5, n_cars), 1.0, 15.0)
        
        for i in range(n_cars):
            in_t = current_date + timedelta(hours=int(in_hours[i]))
            out_t = in_t + timedelta(hours=int(stay_durations[i]))
            data.append({"car_id": 10000 + i, "in_time": in_t, "out_time": out_t})
            
    return pd.DataFrame(data)

def convert_events_to_hourly_status(df_events, target_dates, num_ids=1000):
    """イベント（入出庫）を1時間ごとのステータス形式に展開する"""
    if df_events.empty: return pd.DataFrame()
    
    # 日付とIDの全組み合わせを作成 (メモリ節約のため対象日のみ)
    car_ids = np.arange(10001, 10001 + num_ids)
    all_rows = []
    
    for dt in target_dates:
        base_dt = datetime.combine(dt, datetime.min.time())
        day_events = df_events[df_events['in_time'].dt.date == dt]
        
        # 24時間分ループ
        for h in range(24):
            current_hour = base_dt + timedelta(hours=h)
            # 全IDを一旦homeで初期化
            status_map = {cid: "home" for cid in car_ids}
            
            # イベントに基づいて上書き
            for _, ev in day_events.iterrows():
                cid = ev['car_id']
                if cid not in status_map: continue
                
                t_in = ev['in_time'].replace(minute=0, second=0, microsecond=0)
                t_out = ev['out_time'].replace(minute=0, second=0, microsecond=0)
                
                if current_hour == t_in - timedelta(hours=1):
                    status_map[cid] = "out"  # 往路
                elif t_in <= current_hour < t_out:
                    status_map[cid] = "in"   # 滞在
                elif current_hour == t_out:
                    status_map[cid] = "out"  # 復路
            
            for cid, stat in status_map.items():
                all_rows.append({"datetime": current_hour, "car_id": cid, "in_out": stat})
                
    return pd.DataFrame(all_rows)

@st.cache_data
def process_and_profile_data(df):
    """実データのプロファイリング（既存機能の統計用）"""
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['stay_duration'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600.0
    df['date'] = df['in_time'].dt.date
    df['month'] = df['in_time'].dt.month
    df['weekday'] = df['in_time'].dt.weekday
    df['in_hour'] = df['in_time'].dt.hour
    
    days_count = df.groupby(['month', 'weekday'])['in_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='num_days')
    arrivals = df.groupby(['month', 'weekday', 'in_hour']).size().reset_index(name='total_cars')
    profile_arrival = pd.merge(arrivals, days_count, on=['month', 'weekday'])
    profile_arrival['avg_cars'] = profile_arrival['total_cars'] / profile_arrival['num_days']
    
    profile_stay = df.groupby(['weekday', 'in_hour'])['stay_duration'].agg(['mean', 'std']).reset_index()
    base_capacity = df.groupby('date').size().quantile(0.95) if not df.empty else 100.0
    
    return df, profile_arrival, profile_stay, base_capacity

# ==========================================
# 2. Simulation Engines (VGI仕様)
# ==========================================

@st.cache_data
def generate_vgi_annual_simulation(year, num_ids, pareto_alpha, arr_mean, arr_std, dep_mean, dep_std, base_capacity, profile_arrival, run_id):
    np.random.seed(run_id)
    id_weights = (np.random.pareto(pareto_alpha, num_ids) + 1.0)
    id_weights /= id_weights.mean()
    car_ids = np.arange(10001, 10001 + num_ids)
    
    start_date = datetime(year, 1, 1)
    days_in_year = 365
    event_data = []
    
    for d in range(days_in_year):
        cur_date = start_date + timedelta(days=d)
        month, w_idx = cur_date.month, cur_date.weekday()
        
        day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
        base_prob = min(day_arrival['avg_cars'].sum() / num_ids * 2.0, 1.0) if not day_arrival.empty else 0.1
        
        day_probs = np.clip(base_prob * id_weights, 0, 1)
        active_ids = car_ids[np.random.rand(num_ids) < day_probs]
        
        for cid in active_ids:
            in_h = np.clip(np.random.normal(arr_mean, arr_std), 1, 22)
            dep_h = np.clip(np.random.normal(dep_mean, dep_std), in_h + 1, 23)
            event_data.append({
                "car_id": cid, 
                "in_time": datetime.combine(cur_date, datetime.min.time()) + timedelta(hours=int(in_h)),
                "out_time": datetime.combine(cur_date, datetime.min.time()) + timedelta(hours=int(dep_h))
            })
            
    return pd.DataFrame(event_data), car_ids

# ==========================================
# 3. Main Dashboard UI
# ==========================================

st.set_page_config(layout="wide", page_title="VGI Status Dashboard")
st.title("🚗 VGI 状態遷移分析ダッシュボード")

with st.sidebar:
    st.header("データソース")
    data_mode = st.radio("ソース選択", ["サンプル: 大学データ", "CSVアップロード"])
    raw_events = generate_university_sample() if "サンプル" in data_mode else None
    if raw_events is None: st.stop()

df_real, profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_events)
available_dates = sorted(df_real['date'].unique())

# シミュレーション設定 (Tab4の値をグローバルに使用)
st.sidebar.divider()
st.sidebar.header("シミュレーション・パラメータ")
num_ids = st.sidebar.number_input("登録ID数", 100, 10000, 1000)
pareto_a = st.sidebar.slider("頻度の偏り", 1.0, 5.0, 1.5)
arr_m = st.sidebar.slider("出勤ピーク", 6.0, 12.0, 9.0)
dep_m = st.sidebar.slider("退勤ピーク", 15.0, 21.0, 18.0)

if st.sidebar.button("シミュレーション更新"):
    st.session_state['annual_run_id'] += 1

# 年間シミュレーションの実行
sim_events, all_ids = generate_vgi_annual_simulation(2026, num_ids, pareto_a, arr_m, 1.0, dep_m, 1.5, base_capacity, profile_arrival, st.session_state['annual_run_id'])

tab1, tab2, tab3, tab4 = st.tabs(["日別比較", "全体トレンド", "実データ vs シミュレーション", "VGIログ生成"])

# --- Tab 1: 日別比較 ---
with tab1:
    sel_date = st.selectbox("表示日を選択", available_dates)
    status_df = convert_events_to_hourly_status(sim_events, [sel_date], num_ids)
    
    summary = status_df.groupby([status_df['datetime'].dt.hour, 'in_out']).size().reset_index(name='count')
    fig = px.bar(summary, x='datetime', y='count', color='in_out', 
                 title=f"{sel_date} の状態遷移（延べ {num_ids} ID）",
                 labels={'datetime': '時刻', 'count': '車両台数', 'in_out': '状態'},
                 color_discrete_map={'home': '#95a5a6', 'in': '#2980b9', 'out': '#e67e22'})
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 全体トレンド ---
with tab2:
    st.subheader("時間帯別・状態分布トレンド（平均）")
    # 代表的な1週間の平均を算出
    sample_dates = available_dates[:14]
    trend_status = convert_events_to_hourly_status(sim_events, sample_dates, num_ids)
    trend_summary = trend_status.groupby([trend_status['datetime'].dt.hour, 'in_out']).size().reset_index(name='count')
    trend_summary['avg_count'] = trend_summary['count'] / len(sample_dates)
    
    fig_trend = px.area(trend_summary, x='datetime', y='avg_count', color='in_out',
                        color_discrete_map={'home': '#95a5a6', 'in': '#2980b9', 'out': '#e67e22'})
    st.plotly_chart(fig_trend, use_container_width=True)

# --- Tab 3: 実データ vs シミュレーション ---
with tab3:
    st.subheader("1時間粒度での挙動比較")
    comp_date = st.selectbox("比較する日付", available_dates, key="comp_date")
    
    # RealとSimの両方をStatus形式に変換
    real_status = convert_events_to_hourly_status(raw_events, [comp_date], num_ids)
    sim_status = convert_events_to_hourly_status(sim_events, [comp_date], num_ids)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("実データ（サンプル）")
        s_real = real_status.groupby([real_status['datetime'].dt.hour, 'in_out']).size().reset_index(name='count')
        st.plotly_chart(px.bar(s_real, x='datetime', y='count', color='in_out', color_discrete_map={'home': '#95a5a6', 'in': '#2980b9', 'out': '#e67e22'}), use_container_width=True)
    with c2:
        st.write("シミュレーション結果")
        s_sim = sim_status.groupby([sim_status['datetime'].dt.hour, 'in_out']).size().reset_index(name='count')
        st.plotly_chart(px.bar(s_sim, x='datetime', y='count', color='in_out', color_discrete_map={'home': '#95a5a6', 'in': '#2980b9', 'out': '#e67e22'}), use_container_width=True)

# --- Tab 4: VGIログ生成 ---
with tab4:
    st.subheader("VGIシステム用トランザクションログ")
    if st.button("詳細ログをエクスポート用に展開"):
        export_dates = available_dates[:7] # 1週間分をサンプル表示
        df_vgi = convert_events_to_hourly_status(sim_events, export_dates, num_ids)
        st.dataframe(df_vgi.head(100), use_container_width=True)
        
        csv = df_vgi.to_csv(index=False).encode('utf-8')
        st.download_button("VGIフォーマットCSVをダウンロード", csv, "vgi_status_log.csv", "text/csv")
