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

if 'vgi_result' not in st.session_state:
    st.session_state['vgi_result'] = None

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
        # 土日は少なく、平日は多い
        n_cars = np.random.randint(5, 15) if is_weekend else np.random.randint(80, 150)
        
        in_hours = np.random.normal(9.0, 1.0, n_cars)
        stay_durations = np.clip(np.random.normal(8.0, 2.0, n_cars), 1.0, 14.0)
        
        for i in range(n_cars):
            in_h = int(in_hours[i])
            in_m = int((in_hours[i] % 1) * 60)
            in_time = current_date + timedelta(hours=in_h, minutes=in_m)
            out_time = in_time + timedelta(hours=stay_durations[i])
            # IDをバラつかせる
            data.append({"car_id": 10001 + (i % 500), "in_time": in_time, "out_time": out_time})
    return pd.DataFrame(data)

def process_profile(df):
    """実データから統計プロファイルを抽出"""
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['stay_h'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600
    df['hour'] = df['in_time'].dt.hour
    df['weekday'] = df['in_time'].dt.weekday
    
    # 時間帯別の到着数
    arrival_profile = df.groupby(['weekday', 'hour']).size().reset_index(name='count')
    
    # 曜日ごとのユニークな日付数をカウント (applyを使用してエラー回避)
    days_per_wd = df.groupby('weekday')['in_time'].apply(lambda x: x.dt.date.nunique())
    
    # 1日あたりの平均台数を算出
    arrival_profile['avg_cars'] = arrival_profile.apply(
        lambda x: x['count'] / days_per_wd[x['weekday']] if days_per_wd[x['weekday']] > 0 else 0, axis=1
    )
    
    # 滞在時間の統計
    stay_profile = df.groupby(['weekday', 'hour'])['stay_h'].agg(['mean', 'std']).fillna(1.0).reset_index()
    
    return arrival_profile, stay_profile

def generate_vgi_format(year, total_ids, alpha, arrival_prof, stay_prof, peak_offset=0):
    """VGI用の1時間刻みログを生成 (in/out/home)"""
    np.random.seed(42)
    start_date = datetime(year, 1, 1)
    records = []
    
    # IDごとの頻度（パレート分布）: 1台あたりの平均年20回来場と仮定
    raw_weights = (np.random.pareto(alpha, total_ids) + 1)
    visits_per_id = np.clip(np.round((raw_weights / raw_weights.mean()) * 20).astype(int), 1, 365)
    
    user_ids = np.arange(10001, 10001 + total_ids)
    
    progress = st.progress(0)
    for i, user_id in enumerate(user_ids):
        # 訪問する日をランダム選定
        visit_days = np.random.choice(365, min(visits_per_id[i], 365), replace=False)
        for d in visit_days:
            cur_date = start_date + timedelta(days=int(d))
            wd = cur_date.weekday()
            
            # プロファイルから入庫時間を選択（重み付け抽選）
            possible_hours = arrival_prof[arrival_prof['weekday'] == wd]
            if possible_hours.empty or possible_hours['avg_cars'].sum() == 0:
                in_h = 9
            else:
                in_h = np.random.choice(
                    possible_hours['hour'], 
                    p=possible_hours['avg_cars'] / possible_hours['avg_cars'].sum()
                )
            
            # ピーク補正
            in_h = int(np.clip(in_h + peak_offset, 0, 23))
            
            # 滞在時間をプロファイルから決定
            s_row = stay_prof[(stay_prof['weekday'] == wd) & (stay_prof['hour'] == in_h)]
            s_mean = s_row['mean'].values[0] if not s_row.empty else 8.0
            stay_h = int(np.clip(np.random.normal(s_mean, 2.0), 1, 24))
            out_h = min(23, in_h + stay_h)
            
            # 1日の24時間分のステータス生成
            for h in range(24):
                if h >= in_h and h < out_h:
                    status = "in"
                elif h == out_h:
                    status = "out"
                else:
                    status = "home"
                
                records.append({
                    "datetime": (cur_date + timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00.000"),
                    "car_id": user_id,
                    "in_out": status
                })
        
        if i % 100 == 0:
            progress.progress((i + 1) / total_ids)
    
    progress.empty()
    return pd.DataFrame(records)

# ==========================================
# 2. UI Layout
# ==========================================
st.title("統合型 駐車場分析 & VGIデータ生成")

with st.sidebar:
    st.header("1. データソース設定")
    mode = st.radio("ソース選択", ["サンプル(大学)", "CSVアップロード"])
    
    raw_df = None
    if mode == "CSVアップロード":
        up = st.file_uploader("入出庫ログ(in_time, out_time列)をアップロード", type=['csv'])
        if up:
            raw_df = pd.read_csv(up)
        else:
            st.info("CSVをアップロードしてください。")
            st.stop()
    else:
        raw_df = generate_university_sample()
    
    st.divider()
    st.header("2. VGI生成パラメータ")
    gen_year = st.selectbox("シミュレーション対象年", [2026, 2027], index=0)
    gen_ids = st.number_input("登録車両数 (Unique IDs)", 10, 10000, 1000, step=100)
    gen_alpha = st.slider("頻度の偏り (Alpha)", 0.5, 3.0, 1.2, help="小さいほど一部の車が毎日来ます。")
    peak_shift = st.slider("ピーク時間の補正 (±h)", -3.0, 3.0, 0.0, step=0.5)

# プロファイル抽出の実行
arrival_prof, stay_prof = process_profile(raw_df)

# タブの定義 (変数名を統一)
tab1, tab2 = st.tabs(["📊 実データ分析", "⚙️ VGIデータ生成"])

# --- Tab 1: 分析 ---
with tab1:
    st.subheader("現在の利用パターン分析 (プロファイリング結果)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_arr = px.bar(arrival_prof, x='hour', y='avg_cars', color='weekday', 
                         title="時間帯別の平均流入台数 (曜日別)",
                         labels={'avg_cars': '平均台数', 'hour': '時刻'})
        st.plotly_chart(fig_arr, use_container_width=True)
        
    with col2:
        # 滞在時間の分布を表示
        fig_stay = px.box(raw_df, x='weekday', y='stay_h', points="all",
                          title="曜日別・滞在時間のバラつき",
                          labels={'stay_h': '滞在時間 (h)', 'weekday': '曜日 (0=月)'})
        st.plotly_chart(fig_stay, use_container_width=True)

# --- Tab 2: VGI生成 ---
with tab2:
    st.subheader("VGIシミュレーション用 1時間刻みログ生成")
    st.markdown("""
    左側の「実データ」の傾向を読み取り、指定されたID数で1年分の**in/out/home**ログを作成します。
    ※車が来ない日は出力されません（軽量化済み）。
    """)
    
    if st.button("VGIデータ生成を実行", type="primary", use_container_width=True):
        with st.spinner("データを構築中..."):
            res_df = generate_vgi_format(gen_year, gen_ids, gen_alpha, arrival_prof, stay_prof, peak_shift)
            st.session_state['vgi_result'] = res_df
            
    if st.session_state['vgi_result'] is not None:
        res = st.session_state['vgi_result']
        st.success(f"生成完了！ 総レコード数: {len(res):,}件")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("### 生成データプレビュー")
            st.dataframe(res.head(100), use_container_width=True)
            
        with c2:
            st.write("### ダウンロード")
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSVファイルを保存",
                data=csv,
                file_name=f"vgi_sim_{gen_year}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # 生成データの傾向チェック用
            st.divider()
            st.write("### 生成結果の確認 (in/outのみ)")
            check_df = res[res['in_out'].isin(['in', 'out'])].copy()
            check_df['h'] = pd.to_datetime(check_df['datetime']).dt.hour
            fig_check = px.histogram(check_df, x='h', color='in_out', barmode='group',
                                     title="生成データのピーク分布確認")
            st.plotly_chart(fig_check, use_container_width=True)
