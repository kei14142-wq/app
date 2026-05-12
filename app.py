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
# 1. 状態遷移ロジックのコア関数
# ==========================================

def get_hourly_status_dense(df_events, target_date, car_ids):
    """
    指定日の24時間分を in, out, home で完全に埋めたDataFrameを返す
    """
    base_dt = datetime.combine(target_date, datetime.min.time())
    hours = [base_dt + timedelta(hours=i) for i in range(24)]
    
    records = []
    # 処理対象の日付のイベントを抽出
    day_events = df_events[df_events['in_time'].dt.date == target_date]
    
    for cid in car_ids:
        # このIDの今日のイベントを取得
        cid_events = day_events[day_events['car_id'] == cid]
        
        # 24時間分のスロットをデフォルト 'home' で作成
        status_map = {h: "home" for h in hours}
        
        for _, row in cid_events.iterrows():
            t_in_start = row['in_time'].replace(minute=0, second=0, microsecond=0)
            t_out_start = row['out_time'].replace(minute=0, second=0, microsecond=0)
            
            # 1. 'in' の埋め（滞在中）
            curr = t_in_start
            while curr <= t_out_start:
                if curr in status_map:
                    status_map[curr] = "in"
                curr += timedelta(hours=1)
            
            # 2. 'out' の埋め（前後1時間。inと重なる場合はin優先）
            t_before = t_in_start - timedelta(hours=1)
            t_after = t_out_start + timedelta(hours=1)
            
            if t_before in status_map and status_map[t_before] != "in":
                status_map[t_before] = "out"
            if t_after in status_map and status_map[t_after] != "in":
                status_map[t_after] = "out"
        
        for h, s in status_map.items():
            records.append({"datetime": h, "car_id": cid, "status": s})
            
    return pd.DataFrame(records)

# ==========================================
# 2. Data Processing & Simulation Engines
# ==========================================

@st.cache_data
def generate_university_sample():
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    data = []
    for d in range(365):
        cur_date = start_date + timedelta(days=d)
        is_weekend = cur_date.weekday() >= 5
        n_cars = np.random.randint(5, 15) if is_weekend else np.random.randint(50, 100)
        
        for i in range(n_cars):
            in_h = np.random.normal(9, 1)
            stay = np.random.normal(6, 2)
            in_t = cur_date + timedelta(hours=int(in_h))
            out_t = in_t + timedelta(hours=max(1, int(stay)))
            data.append({"car_id": 10000 + (i % 500), "in_time": in_t, "out_time": out_t})
    return pd.DataFrame(data)

@st.cache_data
def process_data(df):
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['date'] = df['in_time'].dt.date
    return df

@st.cache_data
def generate_vgi_simulation(year, num_ids, pareto_alpha, arr_mean, arr_std, dep_mean, dep_std, run_id):
    np.random.seed(run_id)
    id_weights = (np.random.pareto(pareto_alpha, num_ids) + 1.0)
    id_weights = id_weights / id_weights.mean()
    car_ids = np.arange(10001, 10001 + num_ids)
    
    start_date = datetime(year, 1, 1)
    event_data = []
    for d in range(365):
        cur_date = start_date + timedelta(days=d)
        # 基本来訪確率（曜日の影響）
        base_prob = 0.1 if cur_date.weekday() >= 5 else 0.4
        day_probs = np.clip(base_prob * id_weights, 0, 1)
        active_ids = car_ids[np.random.rand(num_ids) < day_probs]
        
        for cid in active_ids:
            in_h = np.clip(np.random.normal(arr_mean, arr_std), 0, 22)
            out_h = np.clip(np.random.normal(dep_mean, dep_std), in_h + 1, 23.9)
            event_data.append({
                "car_id": cid,
                "in_time": cur_date + timedelta(hours=int(in_h)),
                "out_time": cur_date + timedelta(hours=int(out_h))
            })
    return pd.DataFrame(event_data), car_ids

# ==========================================
# 3. UI Implementation
# ==========================================

st.set_page_config(layout="wide", page_title="VGI Dense Simulator")
st.title("VGI 1時間粒度ステータス・シミュレータ")

# Sidebar Data Loading
with st.sidebar:
    st.header("1. データソース")
    data_mode = st.radio("ソース選択", ["サンプルデータ", "CSVアップロード"])
    if data_mode == "サンプルデータ":
        raw_df = generate_university_sample()
    else:
        uploaded = st.file_uploader("イベントログ(in_time, out_time, car_id)", type="csv")
        if uploaded: raw_df = pd.read_csv(uploaded)
        else: st.stop()
    
    df = process_data(raw_df)
    unique_ids = sorted(df['car_id'].unique())
    available_dates = sorted(df['date'].unique())

tab1, tab2, tab3, tab4 = st.tabs(["日別遷移プレビュー", "全体統計", "既存シミュ同期", "年間VGI生成"])

# --- Tab 1: 日別遷移プレビュー ---
with tab1:
    st.subheader("1時間ごとのステータス遷移（in/out/home）")
    c1, c2 = st.columns(2)
    with c1: sel_date = st.selectbox("日付選択", available_dates, key="t1_date")
    with c2: sel_id = st.multiselect("ID選択（最大5件）", unique_ids, default=unique_ids[:3])
    
    if sel_id:
        dense_df = get_hourly_status_dense(df, sel_date, sel_id)
        
        # タイムラインチャート
        fig = px.timeline(dense_df, x_start="datetime", 
                          x_end=dense_df["datetime"] + timedelta(hours=1), 
                          y="car_id", color="status",
                          color_discrete_map={"in": "#2ecc71", "out": "#e67e22", "home": "#3498db"},
                          category_orders={"status": ["home", "out", "in"]})
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="時刻", yaxis_title="Car ID", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("データ詳細 (1時間粒度):")
        st.dataframe(dense_df.pivot(index="datetime", columns="car_id", values="status"), use_container_width=True)

# --- Tab 2: 全体統計 ---
with tab2:
    st.subheader("1時間粒度での利用トレンド")
    # 全IDの平均的な1日のステータス分布
    sample_date = available_dates[0]
    all_dense = get_hourly_status_dense(df, sample_date, unique_ids[:100]) # 負荷軽減のため100台サンプル
    hourly_dist = all_dense.groupby([all_dense['datetime'].dt.hour, 'status']).size().reset_index(name='count')
    
    fig_dist = px.bar(hourly_dist, x='datetime', y='count', color='status',
                      title=f"{sample_date} のステータス分布 (Sample 100 IDs)",
                      color_discrete_map={"in": "#2ecc71", "out": "#e67e22", "home": "#3498db"})
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Tab 3: 既存シミュ同期 ---
with tab3:
    st.subheader("既存シミュレーション結果の1時間化")
    st.info("このタブでは既存の入出庫分布に基づくシミュレーションを1時間ステータスに変換して表示します。")
    if st.button("シミュレーション実行 & 1時間化"):
        st.session_state['daily_run_id'] += 1
        # 簡易的な再生成と表示
        sim_events = df.sample(frac=0.5) # 既存分布を模したサンプル
        dense_sim = get_hourly_status_dense(sim_events, available_dates[0], unique_ids[:10])
        st.dataframe(dense_sim.head(24), use_container_width=True)

# --- Tab 4: 年間VGI生成 ---
with tab4:
    st.subheader("高密度年間シミュレーション生成")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            v_num_ids = st.number_input("ID数", 1, 10000, 100)
            v_pareto = st.slider("頻度の偏り", 1.0, 5.0, 1.5)
        with c2:
            v_arr = st.slider("出勤ピーク", 6.0, 12.0, 9.0)
            v_arr_s = st.slider("出勤バラツキ", 0.5, 3.0, 1.0)
        with c3:
            v_dep = st.slider("退勤ピーク", 15.0, 21.0, 18.0)
            v_dep_s = st.slider("退勤バラツキ", 0.5, 3.0, 1.5)
            
        if st.button("年間フルデータ生成 (in/out/home)", type="primary"):
            st.session_state['annual_run_id'] += 1

    if st.session_state['annual_run_id'] > 0:
        events, c_ids = generate_vgi_simulation(2026, v_num_ids, v_pareto, v_arr, v_arr_s, v_dep, v_dep_s, st.session_state['annual_run_id'])
        
        st.success(f"{len(events)} 件のイベントを生成。")
        
        # プレビューとして1日分を密に変換
        preview_dense = get_hourly_status_dense(events, datetime(2026, 1, 1).date(), c_ids[:10])
        st.markdown("#### 1月1日のプレビュー（最初の10台）")
        st.dataframe(preview_dense.pivot(index="datetime", columns="car_id", values="status"), use_container_width=True)
        
        # CSVダウンロード用（全データの変換は重いため、イベントベースでのダウンロードを推奨しつつ、
        # 1時間粒度への変換ロジックを同梱したCSV構成を提示）
        st.info("VGIシステムへ：本CSVの欠損時間はすべて 'home' として処理してください。")
        st.download_button("イベントログをダウンロード", events.to_csv(index=False), "vgi_events.csv")
