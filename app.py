import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# Session State & Config
# ==========================================
st.set_page_config(layout="wide", page_title="Parking Intelligence & VGI Gen")

if 'run_id' not in st.session_state:
    st.session_state['run_id'] = 0

# ==========================================
# 1. ロジック：データ生成・プロファイリング
# ==========================================

@st.cache_data
def generate_university_sample():
    """大学のパターンを模したサンプルデータを生成"""
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    data = []
    for d in range(365):
        current_date = start_date + timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        n_cars = np.random.randint(5, 15) if is_weekend else np.random.randint(80, 150)
        
        in_hours = np.random.normal(9.0, 1.0, n_cars)
        stay_durations = np.clip(np.random.normal(8.0, 2.0, n_cars), 1.0, 14.0)
        
        for i in range(n_cars):
            in_time = current_date + timedelta(hours=int(in_hours[i]), minutes=int((in_hours[i]%1)*60))
            out_time = in_time + timedelta(hours=stay_durations[i])
            data.append({"car_id": 10000 + (i % 500), "in_time": in_time, "out_time": out_time})
    return pd.DataFrame(data)

def process_profile(df):
    """実データから統計プロファイルを抽出"""
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['stay_h'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600
    df['hour'] = df['in_time'].dt.hour
    df['weekday'] = df['in_time'].dt.weekday
    
    arrival_profile = df.groupby(['weekday', 'hour']).size().reset_index(name='count')
    # 1日あたりの平均に正規化
    days_per_wd = df.groupby('weekday')['in_time'].dt.date.nunique()
    arrival_profile['avg_cars'] = arrival_profile.apply(lambda x: x['count'] / days_per_wd[x['weekday']], axis=1)
    
    stay_profile = df.groupby(['weekday', 'hour'])['stay_h'].agg(['mean', 'std']).fillna(1.0).reset_index()
    
    return arrival_profile, stay_profile

def generate_vgi_format(year, total_ids, alpha, arrival_prof, stay_prof, peak_offset=0):
    """VGI用の1時間刻みログを生成 (in/out/home)"""
    np.random.seed(42)
    start_date = datetime(year, 1, 1)
    records = []
    
    # IDごとの頻度（パレート分布）
    raw_weights = (np.random.pareto(alpha, total_ids) + 1)
    visits_per_id = np.clip(np.round((raw_weights / raw_weights.mean()) * 20).astype(int), 1, 365)
    
    user_ids = np.arange(10001, 10001 + total_ids)
    
    progress = st.progress(0)
    for i, user_id in enumerate(user_ids):
        visit_days = np.random.choice(365, visits_per_id[i], replace=False)
        for d in visit_days:
            cur_date = start_date + timedelta(days=int(d))
            wd = cur_date.weekday()
            
            # プロファイルから入庫時間を選択
            possible_hours = arrival_prof[arrival_prof['weekday'] == wd]
            if possible_hours.empty: in_h = 9
            else: in_h = np.random.choice(possible_hours['hour'], p=possible_hours['avg_cars']/possible_hours['avg_cars'].sum())
            
            in_h = int(np.clip(in_h + peak_offset, 0, 23))
            
            # 滞在時間
            s_row = stay_prof[(stay_prof['weekday'] == wd) & (stay_prof['hour'] == in_h)]
            s_mean = s_row['mean'].values[0] if not s_row.empty else 8.0
            stay_h = int(np.clip(np.random.normal(s_mean, 2.0), 1, 24))
            out_h = min(23, in_h + stay_h)
            
            for h in range(24):
                status = "in" if in_h <= h < out_h else ("out" if h == out_h else "home")
                records.append({
                    "datetime": (cur_date + timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00.000"),
                    "car_id": user_id,
                    "in_out": status
                })
        if i % 200 == 0: progress.progress((i+1)/total_ids)
    
    progress.empty()
    return pd.DataFrame(records)

# ==========================================
# 2. UI Layout
# ==========================================
st.title("統合型 駐車場分析 & VGIデータ生成")

with st.sidebar:
    st.header("1. データソース")
    mode = st.radio("ソース選択", ["サンプル(大学)", "CSVアップロード"])
    if mode == "CSVアップロード":
        up = st.file_uploader("入出庫ログ(in_time, out_time列)をアップロード")
        if up: raw_df = pd.read_csv(up)
        else: st.stop()
    else:
        raw_df = generate_university_sample()
    
    st.divider()
    st.header("2. VGI生成パラメータ")
    gen_year = st.selectbox("対象年", [2026, 2027])
    gen_ids = st.number_input("登録車両数", 10, 5000, 500)
    gen_alpha = st.slider("頻度の偏り(Alpha)", 0.5, 3.0, 1.2)
    peak_shift = st.slider("ピーク時間の補正(h)", -3.0, 3.0, 0.0)

# プロファイル抽出
arrival_prof, stay_prof = process_profile(raw_df)

tab_anal, tab_vgi = st.tabs(["実データ分析", "VGIデータ生成"])

# --- Tab 1: 分析 (元のコードの機能) ---
with tab1:
    st.subheader("現在の利用パターン分析")
    col1, col2 = st.columns(2)
    with col1:
        fig_arr = px.bar(arrival_prof, x='hour', y='avg_cars', color='weekday', title="時間帯別・曜日別の平均流入台数")
        st.plotly_chart(fig_arr, use_container_width=True)
    with col2:
        fig_stay = px.box(raw_df, x='weekday', y='stay_h', title="曜日別滞在時間分布")
        st.plotly_chart(fig_stay, use_container_width=True)

# --- Tab 2: VGI生成 (新しい機能) ---
with tab2:
    st.subheader("VGIシミュレーション用データ生成")
    st.info("実データのパターンをベースに、指定したID数と偏りで1年分のログを生成します。")
    
    if st.button("VGIデータ生成開始", type="primary", use_container_width=True):
        with st.spinner("生成中..."):
            vgi_df = generate_vgi_format(gen_year, gen_ids, gen_alpha, arrival_prof, stay_prof, peak_shift)
            st.session_state['vgi_result'] = vgi_df
            
    if 'vgi_result' in st.session_state:
        res = st.session_state['vgi_result']
        st.success(f"生成完了: {len(res):,}レコード")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(res.head(100), use_container_width=True)
        with c2:
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("CSVをダウンロード", csv, f"vgi_sim_{gen_year}.csv", "text/csv", use_container_width=True)
            
            # 検証グラフ
            st.write("### 生成データのチェック")
            check_df = res[res['in_out'].isin(['in', 'out'])].copy()
            check_df['h'] = pd.to_datetime(check_df['datetime']).dt.hour
            fig_check = px.histogram(check_df, x='h', color='in_out', barmode='group', title="生成データの入出庫ピーク")
            st.plotly_chart(fig_check, use_container_width=True)
