import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# Session State Initialization
# ==========================================
if 'sim_events' not in st.session_state:
    st.session_state['sim_events'] = pd.DataFrame()
if 'sim_car_ids' not in st.session_state:
    st.session_state['sim_car_ids'] = []

# ==========================================
# 1. Simulation Engine (VGI Logic)
# ==========================================

@st.cache_data
def run_id_based_simulation(year, num_ids, pareto_alpha, arr_mean, arr_std, dep_mean, dep_std, profile_arrival, base_capacity, run_id):
    """IDごとの来訪イベントを生成する (タブ4のパラメータを全タブで共有)"""
    np.random.seed(run_id)
    id_weights = np.random.pareto(pareto_alpha, num_ids) + 1.0 
    id_weights = id_weights / id_weights.mean() 
    car_ids = np.arange(10001, 10001 + num_ids)
    
    start_date = datetime(year, 1, 1)
    days_in_year = (datetime(year, 12, 31) - start_date).days + 1
    event_data = []
    
    daily_macro_factors = {}
    for month in range(1, 13):
        for w_idx in range(7):
            day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
            daily_cars = day_arrival['avg_cars'].sum() if not day_arrival.empty else base_capacity * 0.5
            base_prob = min(daily_cars / max(num_ids, 1) * 2.0, 1.0) 
            daily_macro_factors[(month, w_idx)] = base_prob

    for d in range(days_in_year):
        cur_date = start_date + timedelta(days=d)
        month, w_idx = cur_date.month, cur_date.weekday()
        base_dt = datetime.combine(cur_date, datetime.min.time())
        day_probs = np.clip(daily_macro_factors[(month, w_idx)] * id_weights, 0, 1)
        visits = np.random.rand(num_ids) < day_probs
        active_ids = car_ids[visits]
        
        for cid in active_ids:
            in_h = int(np.clip(np.random.normal(arr_mean, arr_std), 1, 22))
            dep_h = int(np.clip(np.random.normal(dep_mean, dep_std), in_h + 1, 23))
            
            event_data.append({
                "car_id": cid,
                "date": cur_date.date(),
                "in_h": in_h,
                "out_h": dep_h
            })
            
    return pd.DataFrame(event_data), car_ids

def generate_dense_vgi_log(df_events, year, car_ids, target_date=None):
    """1時間ごとの全スロットを home/in/out で埋める"""
    if df_events.empty:
        return pd.DataFrame()

    # 特定の日付が指定された場合はフィルタリング
    work_df = df_events[df_events['date'] == target_date] if target_date else df_events
    
    results = []
    # 処理を高速化するため辞書形式でイベントを保持
    event_dict = {}
    for _, row in work_df.iterrows():
        event_dict[(row['car_id'], row['date'])] = (row['in_h'], row['out_h'])

    dates = [target_date] if target_date else sorted(df_events['date'].unique())
    
    for dt in dates:
        base_dt = datetime.combine(dt, datetime.min.time())
        for cid in car_ids:
            # デフォルトは24時間すべて home
            statuses = ["home"] * 24
            
            if (cid, dt) in event_dict:
                in_h, out_h = event_dict[(cid, dt)]
                # 入庫前の移動 (out)
                if in_h > 0: statuses[in_h - 1] = "out"
                # 滞在 (in)
                for h in range(in_h, out_h):
                    statuses[h] = "in"
                # 退庫の移動 (out)
                statuses[out_h] = "out"
            
            for h in range(24):
                results.append({
                    "datetime": base_dt + timedelta(hours=h),
                    "car_id": cid,
                    "in_out": statuses[h]
                })
                
    return pd.DataFrame(results)

# ==========================================
# 2. Main UI & Tabs Logic
# ==========================================

st.set_page_config(layout="wide", page_title="Parking VGI Dashboard")
st.title("駐車場分析 & VGIシミュレータ")

# サイドバー：シミュレーション設定
with st.sidebar:
    st.header("1. 基本プロファイル（分析元）")
    # ここでは既存の大学サンプルを使用
    raw_df = pd.read_csv("sample_parking.csv") if False else None # ファイルがあれば読み込み
    
    @st.cache_data
    def get_base_profile():
        # 内部でサンプルを生成
        np.random.seed(42)
        data = []
        for d in range(365):
            dt = datetime(2025, 4, 1) + timedelta(days=d)
            n = np.random.randint(50, 100)
            for _ in range(n):
                in_h = np.random.normal(9, 1)
                stay = np.random.normal(5, 2)
                data.append({"in_time": dt + timedelta(hours=in_h), "out_time": dt + timedelta(hours=in_h+stay)})
        df = pd.DataFrame(data)
        df['month'] = df['in_time'].dt.month
        df['weekday'] = df['in_time'].dt.weekday
        df['in_hour'] = df['in_time'].dt.hour
        avg_arrival = df.groupby(['month', 'weekday', 'in_hour']).size().reset_index(name='total')
        avg_arrival['avg_cars'] = avg_arrival['total'] / 52 # 概算
        return avg_arrival, 100.0 # base_capacity
    
    profile_arrival, base_capacity = get_base_profile()

    st.header("2. IDベース生成パラメータ")
    t_yr = st.selectbox("対象年", [2026, 2027])
    num_ids = st.number_input("登録ID数 (1-10000)", 1, 10000, 1000)
    pareto_alpha = st.slider("利用頻度の偏り", 1.0, 5.0, 1.5)
    arr_peak = st.slider("出勤ピーク時間", 6, 12, 9)
    dep_peak = st.slider("退勤ピーク時間", 15, 21, 17)
    
    if st.button("シミュレーション実行・更新", type="primary"):
        st.session_state['sim_events'], st.session_state['sim_car_ids'] = run_id_based_simulation(
            t_yr, num_ids, pareto_alpha, arr_peak, 1.0, dep_peak, 1.5, 
            profile_arrival, base_capacity, 42
        )
        st.success("生成完了")

if st.session_state['sim_events'].empty:
    st.warning("サイドバーの「シミュレーション実行」を押してデータを生成してください。")
    st.stop()

df_ev = st.session_state['sim_events']
car_ids = st.session_state['sim_car_ids']

tab1, tab2, tab3, tab4 = st.tabs(["日別比較", "全体トレンド", "日別シミュレーション表示", "VGIログ生成・DL"])

# --- Tab 1: 日別比較 ---
with tab1:
    st.subheader("IDベース生成結果の日別比較")
    dates = sorted(df_ev['date'].unique())
    sel_dates = st.multiselect("日付選択", dates, default=dates[:2])
    
    if sel_dates:
        fig = go.Figure()
        for i, dt in enumerate(sel_dates):
            # 滞在台数(in)を計算
            day_data = df_ev[df_ev['date'] == dt]
            hourly_counts = [((day_data['in_h'] <= h) & (day_data['out_h'] > h)).sum() for h in range(24)]
            fig.add_trace(go.Scatter(x=[f"{h:02d}:00" for h in range(24)], y=hourly_counts, name=str(dt)))
        
        fig.update_layout(title="滞在台数の推移 (in状態)", xaxis_title="時間", yaxis_title="台数", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 全体トレンド ---
with tab2:
    st.subheader("生成データの月別・曜日別利用傾向")
    c1, c2 = st.columns(2)
    df_ev['month'] = pd.to_datetime(df_ev['date']).dt.month
    df_ev['weekday'] = pd.to_datetime(df_ev['date']).dt.weekday
    
    with c1:
        m_trend = df_ev.groupby('month').size().reset_index(name='count')
        st.plotly_chart(px.bar(m_trend, x='month', y='count', title="月別延べ利用数"), use_container_width=True)
    with c2:
        w_trend = df_ev.groupby('weekday').size().reset_index(name='count')
        w_trend['day'] = w_trend['weekday'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        st.plotly_chart(px.bar(w_trend, x='day', y='count', title="曜日別延べ利用数"), use_container_width=True)

# --- Tab 3: 日別シミュレーション表示 ---
with tab3:
    st.subheader("指定日の全IDステータス推移")
    target_dt = st.selectbox("表示する日付", dates, key="tab3_date")
    
    # プレビュー用に最初の50ID分だけ表示
    sample_vgi = generate_dense_vgi_log(df_ev, t_yr, car_ids[:50], target_date=target_dt)
    
    fig_heat = px.strip(sample_vgi, x="datetime", y="car_id", color="in_out",
                        category_orders={"in_out": ["home", "out", "in"]},
                        title="IDごとの状態遷移 (上位50ID)")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 4: VGIログ生成・DL ---
with tab4:
    st.subheader("VGIシステム用 1時間粒度トランザクションログ")
    st.write(f"設定内容: {num_ids} ID / {t_yr}年 / 1時間ごとに全ステータスを補完")
    
    if st.button("全期間のCSVデータを作成（時間がかかる場合があります）"):
        with st.spinner("CSV生成中..."):
            # 大容量になるため、ここでは直近7日分をサンプルとして生成する例
            # 全期間が必要な場合は car_ids を分割してループ処理することを推奨
            full_vgi = generate_dense_vgi_log(df_ev, t_yr, car_ids)
            
            st.dataframe(full_vgi.head(100), use_container_width=True)
            
            csv = full_vgi.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="VGI形式のCSVをダウンロード",
                data=csv,
                file_name=f"vgi_log_{t_yr}_{num_ids}ids.csv",
                mime='text/csv'
            )
