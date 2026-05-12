import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==========================================
# 1. Data Processing & Profiling Functions
# ==========================================

@st.cache_data
def generate_university_sample():
    """大学の利用パターンを模したサンプル（プロファイル作成用）"""
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    days = 365
    data = []
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        n_cars = np.random.randint(5, 15) if is_weekend else np.random.randint(80, 150)
        
        n_morning = int(n_cars * 0.6)
        in_hours = np.concatenate([np.random.normal(9.0, 1.0, n_morning), np.random.normal(13.5, 1.5, n_cars - n_morning)])
        in_hours = np.clip(in_hours, 0, 23.0)
        stay_durations = np.clip(np.random.normal(5.0, 2.5, n_cars), 1.0, 15.0)
        
        for i in range(n_cars):
            in_t = current_date + timedelta(hours=int(in_hours[i]))
            out_t = in_t + timedelta(hours=int(stay_durations[i]))
            data.append({"in_time": in_t, "out_time": out_t})
    return pd.DataFrame(data)

@st.cache_data
def process_and_profile_data(df):
    """生データから到着率と滞在時間の分布プロファイルを抽出"""
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['stay_duration'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600.0
    
    # 到着率プロファイル
    df['month'] = df['in_time'].dt.month
    df['weekday'] = df['in_time'].dt.weekday
    df['in_hour'] = df['in_time'].dt.hour
    
    days_count = df.groupby(['month', 'weekday'])['in_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='num_days')
    arrivals = df.groupby(['month', 'weekday', 'in_hour']).size().reset_index(name='total_cars')
    profile_arrival = pd.merge(arrivals, days_count, on=['month', 'weekday'])
    profile_arrival['avg_cars'] = profile_arrival['total_cars'] / profile_arrival['num_days']
    
    # 滞在時間プロファイル
    profile_stay = df.groupby(['weekday', 'in_hour'])['stay_duration'].agg(['mean', 'std']).reset_index()
    profile_stay['std'] = profile_stay['std'].fillna(1.0)
    
    base_capacity = df.groupby(df['in_time'].dt.date).size().quantile(0.95)
    return profile_arrival, profile_stay, base_capacity

# ==========================================
# 2. Unified Simulation Engine
# ==========================================

def run_simulation(year, num_ids, regular_ratio, target_daily_cars, arr_shift, stay_scale, profile_arrival, profile_stay, progress_bar, status_text):
    np.random.seed(int(time.time()))
    
    # ユーザー頻度モデル
    num_regular = int(num_ids * (regular_ratio / 100.0))
    num_medium = int((num_ids - num_regular) * 0.4)
    num_light = num_ids - num_regular - num_medium
    
    base_probs = np.concatenate([
        np.random.normal(4.5/7.0, 1.0/7.0, num_regular),
        np.random.normal(2.0/7.0, 0.5/7.0, num_medium),
        np.random.normal(0.5/7.0, 0.2/7.0, num_light)
    ])
    base_probs = np.clip(base_probs, 0.02, 0.95)
    np.random.shuffle(base_probs)
    car_ids = np.arange(1, 1 + num_ids)
    
    # スケーリング
    scale_factor = (target_daily_cars / num_ids) / np.mean(base_probs)
    base_probs = np.clip(base_probs * scale_factor, 0.01, 0.98)

    # 季節・曜日の波
    daily_factors = {}
    for month in range(1, 13):
        for w_idx in range(7):
            day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
            daily_factors[(month, w_idx)] = day_arrival['avg_cars'].sum() if not day_arrival.empty else 1.0
    m_val = np.mean(list(daily_factors.values()))
    for k in daily_factors: daily_factors[k] /= m_val

    # イベント生成
    start_date = datetime(year, 1, 1)
    event_data = []
    for d in range(365):
        if d % 20 == 0:
            progress_bar.progress(int((d / 365) * 50))
            status_text.text(f"シミュレーション生成中... {d}/365日")
            
        cur_date = start_date + timedelta(days=d)
        month, w_idx = cur_date.month, cur_date.weekday()
        day_probs = np.clip(base_probs * daily_factors.get((month, w_idx), 1.0), 0, 1)
        active_ids = car_ids[np.random.rand(num_ids) < day_probs]
        
        p_arr = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
        arr_h_dist = p_arr['in_hour'].values if not p_arr.empty else [9]
        arr_p_dist = p_arr['avg_cars'].values / p_arr['avg_cars'].sum() if not p_arr.empty else [1]

        for cid in active_ids:
            base_h = np.random.choice(arr_h_dist, p=arr_p_dist)
            in_h = np.clip(base_h + arr_shift + np.random.normal(0, 0.5), 0, 22)
            
            p_stay = profile_stay[(profile_stay['weekday'] == w_idx) & (profile_stay['in_hour'] == int(base_h))]
            s_mean = p_stay['mean'].values[0] if not p_stay.empty else 5.0
            s_std = p_stay['std'].values[0] if not p_stay.empty else 2.0
            # バラつきを持たせてサンプリング
            duration = max(1.0, np.random.normal(s_mean, s_std) * stay_scale)
            
            in_t = datetime.combine(cur_date, datetime.min.time()) + timedelta(hours=int(in_h))
            out_t = in_t + timedelta(hours=int(duration))
            event_data.append({"car_id": cid, "in_time": in_t, "out_time": out_t, "stay_duration": duration})
            
    df_res = pd.DataFrame(event_data)
    df_res['date'] = df_res['in_time'].dt.date
    df_res['month'] = df_res['in_time'].dt.month
    df_res['weekday'] = df_res['in_time'].dt.weekday
    df_res['in_hour'] = df_res['in_time'].dt.hour
    df_res['out_hour'] = df_res['out_time'].dt.hour
    return df_res, car_ids

def generate_vgi_log(df_events, year, all_car_ids, progress_bar, status_text):
    """イベントからステータスログ(hourly)を展開"""
    times = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='h')
    status_matrix = np.full((len(times), len(all_car_ids)), 'home', dtype=object)
    id_map = {cid: i for i, cid in enumerate(all_car_ids)}
    start_t = times[0]
    
    total = len(df_events)
    for i, (_, row) in enumerate(df_events.iterrows()):
        if i % (total // 10 + 1) == 0:
            progress_bar.progress(60 + int((i / total) * 35))
            status_text.text(f"CSV変換中... {i}/{total}")
            
        cid_idx = id_map[row['car_id']]
        idx_in = int((row['in_time'].floor('h') - start_t).total_seconds() // 3600)
        idx_out = int((row['out_time'].floor('h') - start_t).total_seconds() // 3600)
        
        if 0 <= idx_in < len(times):
            status_matrix[idx_in:min(idx_out, len(times)), cid_idx] = 'in'
        if 0 <= idx_out < len(times):
            status_matrix[idx_out, cid_idx] = 'out'
            
    df_vgi = pd.DataFrame({
        'datetime': np.tile(times, len(all_car_ids)),
        'car_id': np.repeat(all_car_ids, len(times)),
        'in_out': status_matrix.flatten('F')
    })
    return df_vgi

# ==========================================
# 3. Main UI
# ==========================================

st.set_page_config(layout="wide", page_title="Unified VGI Simulator")
st.title("駐車場データ")

# 初期データのロード
raw_sample = generate_university_sample()
prof_arr, prof_stay, base_cap = process_and_profile_data(raw_sample)

with st.sidebar:
    st.header("設定")
    t_yr = st.selectbox("対象年", [2025, 2026, 2027])
    num_ids = st.number_input("登録ID数", 1, 10000, 1000)
    target_avg = st.number_input("平均来訪台数/日", 1, 1000, int(base_cap))
    st.divider()
    reg_ratio = st.slider("定期利用者の割合(%)", 0, 100, 30)
    arr_shift = st.slider("到着時間のシフト", -3.0, 3.0, 0.0, 0.5)
    stay_scale = st.slider("滞在時間の倍率", 0.5, 3.0, 1.0, 0.1)
    
    if st.button("シミュレーションを実行", type="primary", use_container_width=True):
        st.session_state['trigger_sim'] = True

if st.session_state.get('trigger_sim', False):
    pb = st.progress(0); st_txt = st.empty()
    sim_df, ids = run_simulation(t_yr, num_ids, reg_ratio, target_avg, arr_shift, stay_scale, prof_arr, prof_stay, pb, st_txt)
    st.session_state['sim_df'], st.session_state['ids'] = sim_df, ids
    st.session_state['trigger_sim'] = False
    pb.empty(); st_txt.empty(); st.rerun()

if 'sim_df' in st.session_state:
    df_sim = st.session_state['sim_df']
    tab1, tab2, tab3 = st.tabs(["日別詳細分析", "全体トレンド & 分布", "VGIログ生成(CSV)"])

    with tab1:
        st.markdown("### 指定日の挙動（シミュレーション結果より）")
        sel_dates = st.multiselect("日付を選択:", sorted(df_sim['date'].unique()), default=sorted(df_sim['date'].unique())[:2])
        if sel_dates:
            c1, c2 = st.columns(2)
            # 滞在台数計算
            with c1:
                f1 = go.Figure()
                for dt in sel_dates:
                    day_ev = df_sim[df_sim['date'] == dt]
                    # 10分刻みで滞在台数を計算
                    t_range = pd.date_range(start=datetime.combine(dt, datetime.min.time()), periods=144, freq='10min')
                    counts = [((day_ev['in_time'] <= t) & (day_ev['out_time'] > t)).sum() for t in t_range]
                    f1.add_trace(go.Scatter(x=[t.strftime('%H:%M') for t in t_range], y=counts, name=str(dt)))
                f1.update_layout(title="滞在台数の推移", template="plotly_white")
                st.plotly_chart(f1, use_container_width=True)
            with c2:
                f2 = go.Figure()
                for dt in sel_dates:
                    day_ev = df_sim[df_sim['date'] == dt]
                    in_h = day_ev.groupby('in_hour').size()
                    f2.add_trace(go.Bar(x=in_h.index, y=in_h.values, name=f"IN: {dt}"))
                f2.update_layout(title="時間別入庫台数", template="plotly_white", barmode='group')
                st.plotly_chart(f2, use_container_width=True)

    with tab2:
        st.markdown("### 生成データの統計・分布")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**滞在時間の分布（全ID・通年）**")
            f3 = px.histogram(df_sim, x="stay_duration", nbins=50, color_discrete_sequence=['#2980b9'])
            f3.update_layout(xaxis_title="滞在時間 (h)", yaxis_title="サンプル数", template="plotly_white")
            st.plotly_chart(f3, use_container_width=True)
        with c2:
            st.markdown("**曜日別の平均来訪台数**")
            w_avg = df_sim.groupby(['weekday', 'date']).size().reset_index(name='count').groupby('weekday')['count'].mean()
            f4 = px.bar(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y=w_avg.values, color_discrete_sequence=['#7f8c8d'])
            f4.update_layout(yaxis_title="平均台数", template="plotly_white")
            st.plotly_chart(f4, use_container_width=True)

    with tab3:
        st.markdown("### VGIシステム用ログ出力")
        st.warning("大規模データの展開を行うため、ボタン押下後に数秒かかります。")
        if st.button("CSV形式に展開する"):
            pb = st.progress(60); st_txt = st.empty()
            df_vgi = generate_vgi_log(df_sim, t_yr, st.session_state['ids'], pb, st_txt)
            st.success(f"展開完了: {len(df_vgi):,} 行")
            st.dataframe(df_vgi.head(100), use_container_width=True)
            st.download_button("CSVをダウンロード", df_vgi.to_csv(index=False).encode('utf-8'), f"vgi_unified_log_{t_yr}.csv")
            pb.empty(); st_txt.empty()
