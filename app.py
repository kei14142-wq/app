import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. Data Processing & Profiling Functions
# ==========================================

@st.cache_data
def generate_university_sample():
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    days = 365
    data = []
    
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        is_weekend = current_date.weekday() >= 5
        
        if is_weekend: n_cars = np.random.randint(5, 15)
        elif month in [8, 9, 2, 3]: n_cars = np.random.randint(30, 60)
        else: n_cars = np.random.randint(80, 150)
            
        if n_cars == 0: continue
            
        n_morning = int(n_cars * 0.6)
        n_afternoon = n_cars - n_morning
        in_hours = np.concatenate([
            np.random.normal(9.0, 1.0, n_morning),
            np.random.normal(13.5, 1.5, n_afternoon)
        ])
        in_hours = np.clip(in_hours, 0, 23.0)
        stay_durations = np.clip(np.random.normal(5.0, 2.5, n_cars), 1.0, 15.0)
        
        for i in range(n_cars):
            h = int(in_hours[i])
            in_time = current_date + timedelta(hours=h)
            out_time = in_time + timedelta(hours=int(stay_durations[i]))
            
            data.append({
                "in_time": in_time.strftime("%Y-%m-%d %H:%M:%S"), 
                "out_time": out_time.strftime("%Y-%m-%d %H:%M:%S")
            })
    return pd.DataFrame(data)

@st.cache_data
def process_and_profile_data(df):
    df['in_time'] = pd.to_datetime(df['in_time'])
    df['out_time'] = pd.to_datetime(df['out_time'])
    df['stay_duration'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600.0
    
    days_count = df.groupby([df['in_time'].dt.month.rename('month'), df['in_time'].dt.weekday.rename('weekday')])['in_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='num_days')
    arrivals = df.groupby([df['in_time'].dt.month.rename('month'), df['in_time'].dt.weekday.rename('weekday'), df['in_time'].dt.hour.rename('in_hour')]).size().reset_index(name='total_cars')
    profile_arrival = pd.merge(arrivals, days_count, on=['month', 'weekday'])
    profile_arrival['avg_cars'] = profile_arrival['total_cars'] / profile_arrival['num_days']
    
    profile_stay = df.groupby([df['in_time'].dt.weekday.rename('weekday'), df['in_time'].dt.hour.rename('in_hour')])['stay_duration'].agg(['mean', 'std']).reset_index()
    
    daily_counts = df.groupby(df['in_time'].dt.date).size()
    base_capacity = daily_counts.quantile(0.95) if not daily_counts.empty else 21.0
    
    return profile_arrival, profile_stay, base_capacity

def calc_parked_cars(df, target_date, freq='10min'):
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = start_dt + timedelta(hours=23, minutes=50) 
    time_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    if df.empty: return pd.DataFrame({"time_str": [t.strftime('%H:%M') for t in time_range], "parked_cars": 0})
    counts = [( (df['in_time'] <= t) & (df['out_time'] > t) ).sum() for t in time_range]
    counts = np.array(counts) - counts[0]
    return pd.DataFrame({"time_str": [t.strftime('%H:%M') for t in time_range], "parked_cars": counts})

def calc_in_out_hourly(df, target_date):
    hours = pd.DataFrame({'hour': range(24)})
    if df.empty:
        hours['in_count'] = 0
        hours['out_count'] = 0
        hours['time_str'] = hours['hour'].apply(lambda x: f"{x:02d}:00")
        return hours
    in_df = df[df['in_time'].dt.date == target_date].groupby('in_hour').size().reset_index(name='in_count')
    out_df = df[df['out_time'].dt.date == target_date].groupby('out_hour').size().reset_index(name='out_count')
    res = pd.merge(hours, in_df, left_on='hour', right_on='in_hour', how='left')
    res = pd.merge(res, out_df, left_on='hour', right_on='out_hour', how='left').fillna(0)
    res['time_str'] = res['hour'].apply(lambda x: f"{x:02d}:00")
    return res

# ==========================================
# 2. VGI Simulation Engine (Realistic Frequency)
# ==========================================

@st.cache_data
def generate_vgi_simulation(year, num_ids, regular_ratio, target_daily_cars, arr_shift, stay_scale, profile_arrival, profile_stay, run_id):
    np.random.seed(run_id)
    
    # 1. ユーザーを3つの利用頻度グループに分割して現実的な確率を割り当て
    num_regular = int(num_ids * (regular_ratio / 100.0))
    num_medium = int((num_ids - num_regular) * 0.4) # 残りの40%は中頻度（週1~2日）
    num_light = num_ids - num_regular - num_medium  # 残りの60%は低頻度（たまに）

    # 週あたりの来訪日数(平均) / 7 をベース確率とする
    prob_regular = np.random.normal(4.0/7.0, 1.0/7.0, num_regular)
    prob_medium = np.random.normal(1.5/7.0, 0.5/7.0, num_medium)
    prob_light = np.random.normal(0.2/7.0, 0.1/7.0, num_light)

    base_probs = np.concatenate([prob_regular, prob_medium, prob_light])
    base_probs = np.clip(base_probs, 0.01, 0.95) # 誰もが最低1%は来る、最大でも95%
    np.random.shuffle(base_probs) # IDにランダム割り当て

    car_ids = np.arange(1, 1 + num_ids) # IDは1からスタート
    
    # 2. 指定された平均来訪台数(target_daily_cars)に合うように確率全体をスケーリング
    current_mean_prob = np.mean(base_probs)
    target_mean_prob = target_daily_cars / num_ids
    scale_factor = target_mean_prob / current_mean_prob if current_mean_prob > 0 else 1.0
    base_probs = np.clip(base_probs * scale_factor, 0.005, 0.98) # スケール後も0~1に収める

    # 3. 季節・曜日の波ファクターを取得
    daily_factors = {}
    for month in range(1, 13):
        for w_idx in range(7):
            day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
            daily_factors[(month, w_idx)] = day_arrival['avg_cars'].sum() if not day_arrival.empty else 1.0
    mean_factor = np.mean(list(daily_factors.values())) if daily_factors else 1.0
    for k in daily_factors: daily_factors[k] /= mean_factor

    start_date = datetime(year, 1, 1)
    days_in_year = (datetime(year, 12, 31) - start_date).days + 1
    event_data = []

    for d in range(days_in_year):
        cur_date = start_date + timedelta(days=d)
        month, w_idx = cur_date.month, cur_date.weekday()
        base_dt = datetime.combine(cur_date, datetime.min.time())
        
        # この日の各IDの来訪確率 = 個人ベース確率 × 季節曜日の波
        day_probs = np.clip(base_probs * daily_factors.get((month, w_idx), 1.0), 0, 1)
        visits = np.random.rand(num_ids) < day_probs
        active_ids = car_ids[visits]
        
        day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
        arr_probs = day_arrival['avg_cars'].values if not day_arrival.empty else np.array([1])
        arr_hours = day_arrival['in_hour'].values if not day_arrival.empty else np.array([9])
        if arr_probs.sum() > 0: arr_probs = arr_probs / arr_probs.sum()
        
        for cid in active_ids:
            base_in_h = np.random.choice(arr_hours, p=arr_probs)
            in_hour = np.clip(base_in_h + arr_shift, 0, 22.0)
            
            stay_row = profile_stay[(profile_stay['weekday'] == w_idx) & (profile_stay['in_hour'] == int(base_in_h))]
            stay_mean = stay_row['mean'].values[0] if not stay_row.empty else 4.0
            stay_duration = max(1.0, stay_mean * stay_scale)
            
            in_t = base_dt + timedelta(hours=int(in_hour))
            out_t = in_t + timedelta(hours=int(stay_duration))
            
            event_data.append({"car_id": cid, "in_time": in_t, "out_time": out_t})
            
    df_events = pd.DataFrame(event_data)
    if not df_events.empty:
        df_events['stay_duration'] = (df_events['out_time'] - df_events['in_time']).dt.total_seconds() / 3600.0
        df_events['date'] = df_events['in_time'].dt.date
        df_events['month'] = df_events['in_time'].dt.month
        df_events['weekday'] = df_events['in_time'].dt.weekday
        df_events['in_hour'] = df_events['in_time'].dt.hour
        df_events['out_hour'] = df_events['out_time'].dt.hour
    return df_events, car_ids

@st.cache_data
def convert_to_full_vgi_format(df_events, year, all_car_ids):
    times = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='h')
    num_times, num_ids = len(times), len(all_car_ids)
    
    # 全体を 'home' で初期化したステータス行列
    status_matrix = np.full((num_times, num_ids), 'home', dtype=object)
    
    # IDを配列インデックスに変換する辞書 (IDは1からなので注意)
    car_id_to_idx = {cid: i for i, cid in enumerate(all_car_ids)}
    
    start_time = times[0]
    for _, row in df_events.iterrows():
        cid_idx = car_id_to_idx[row['car_id']]
        idx_in = int((row['in_time'].floor('h') - start_time).total_seconds() // 3600)
        idx_out = int((row['out_time'].floor('h') - start_time).total_seconds() // 3600)
        
        # in (滞在：入庫から出庫の1時間前まで)
        if 0 <= idx_in < num_times:
            status_matrix[idx_in:min(idx_out, num_times), cid_idx] = 'in'
            
        # out (退勤：出庫の瞬間のみ)
        if 0 <= idx_out < num_times:
            status_matrix[idx_out, cid_idx] = 'out'

    # 行列を縦持ちのDataFrameに高速変換
    flat_status = status_matrix.flatten('F')
    flat_ids = np.repeat(all_car_ids, num_times)
    flat_times = np.tile(times, num_ids)
    
    return pd.DataFrame({'datetime': flat_times, 'car_id': flat_ids, 'in_out': flat_status})

# ==========================================
# 3. UI
# ==========================================

st.set_page_config(layout="wide", page_title="VGI Simulator V3")
st.title("駐車場")

raw_df = generate_university_sample()
profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_df)

with st.sidebar:
    st.header("シミュレーション設定")
    t_yr = st.selectbox("対象年", [2025, 2026, 2027])
    num_ids = st.number_input("登録ID数 (1-10000)", 1, 10000, 1000)
    target_daily = st.number_input("平均来訪台数/日", 1, 10000, int(base_capacity))
    
    st.divider()
    st.markdown("**ユーザー層と行動パターンの調整**")
    regular_ratio = st.slider("定期利用者(週3~5日)の割合 (%)", 5, 80, 20, help="この値が高いほど、いつも同じ人が来る施設になります。")
    arr_shift = st.slider("到着時間のシフト", -3.0, 3.0, 0.0, 0.5)
    stay_scale = st.slider("滞在時間の倍率", 0.5, 2.0, 1.0, 0.1)
    
    if st.button("シミュレーション実行", type="primary", use_container_width=True):
        st.session_state['sim_run_id'] = st.session_state.get('sim_run_id', 0) + 1

if 'sim_df' not in st.session_state or st.session_state.get('sim_run_id', 0) > 0:
    with st.spinner("シミュレーション実行中..."):
        sim_df, all_ids = generate_vgi_simulation(
            t_yr, num_ids, regular_ratio, target_daily, 
            arr_shift, stay_scale, profile_arrival, profile_stay, 
            st.session_state.get('sim_run_id', 0)
        )
        st.session_state['sim_df'], st.session_state['all_ids'] = sim_df, all_ids
        st.session_state['sim_run_id'] = 0

sim_df, all_ids = st.session_state['sim_df'], st.session_state['all_ids']
available_dates = sorted(sim_df['date'].unique())

tab1, tab2, tab3 = st.tabs(["日別比較", "全体トレンド", "VGIログエクスポート"])
colors = ['#2980b9', '#e67e22', '#27ae60']

with tab1:
    sel_dates = st.multiselect("比較する日付:", available_dates, default=available_dates[:2] if len(available_dates)>1 else available_dates)
    if sel_dates:
        p_data = []
        for i, dt in enumerate(sel_dates):
            day_df = sim_df[sim_df['date'] == dt]
            d_p = calc_parked_cars(sim_df, dt)
            d_io = calc_in_out_hourly(sim_df, dt)
            p_data.append({'dt': dt, 'parked': d_p, 'inout': d_io, 'raw': day_df})
                
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. 滞在台数の推移**")
            f_p = go.Figure()
            for i, d in enumerate(p_data): 
                f_p.add_trace(go.Scatter(x=d['parked']['time_str'], y=d['parked']['parked_cars'], name=str(d['dt']), line=dict(color=colors[i%3])))
            st.plotly_chart(f_p, use_container_width=True)
        with c2:
            st.markdown("**2. 時間別入出庫ダイナミクス**")
            f_io = go.Figure()
            for i, d in enumerate(p_data):
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=d['inout']['in_count'], name=f"IN: {d['dt']}", marker_color=colors[i%3]))
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=-d['inout']['out_count'], name=f"OUT: {d['dt']}", marker_color=colors[i%3], opacity=0.5))
            st.plotly_chart(f_io, use_container_width=True)

with tab2:
    st.markdown("### 月別・曜日別トレンド")
    c1, c2 = st.columns(2)
    with c1:
        m_t = sim_df.groupby('month').size().reset_index(name='t')
        st.plotly_chart(px.bar(m_t, x='month', y='t', title="月別総来訪台数"), use_container_width=True)
    with c2:
        w_t = sim_df.groupby('weekday').size().reset_index(name='t')
        w_t['dw'] = w_t['weekday'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
        st.plotly_chart(px.bar(w_t, x='dw', y='t', title="曜日別総来訪台数"), use_container_width=True)

with tab3:
    st.markdown("### VGIシステム用 状態遷移ログ出力")
    st.info("入庫から出庫前までを `in`、出庫時を `out`、それ以外をすべて `home` で埋めた完全なログを生成します。")
    if st.button("詳細ログを展開してダウンロード準備", type="primary"):
        with st.spinner("全IDの年間推移を展開中..."):
            df_vgi = convert_to_full_vgi_format(sim_df, t_yr, all_ids)
            st.dataframe(df_vgi.head(1000), use_container_width=True)
            st.download_button("VGIフォーマットCSVをダウンロード", df_vgi.to_csv(index=False).encode('utf-8'), f"vgi_log_{t_yr}.csv")
