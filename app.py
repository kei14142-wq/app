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
            
        # 午前ピークと午後ピーク（リアルな大学の波）
        n_morning = int(n_cars * 0.6)
        n_afternoon = n_cars - n_morning
        in_hours = np.concatenate([
            np.random.normal(9.0, 1.0, n_morning),
            np.random.normal(13.5, 1.5, n_afternoon)
        ])
        in_hours = np.clip(in_hours, 0, 23.99)
        stay_durations = np.clip(np.random.normal(5.0, 2.5, n_cars), 0.5, 15.0)
        
        for i in range(n_cars):
            h = int(in_hours[i])
            m = int((in_hours[i] % 1) * 60)
            in_time = current_date + timedelta(hours=h, minutes=m)
            out_time = in_time + timedelta(hours=stay_durations[i])
            
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
    profile_stay['std'] = profile_stay['std'].fillna(1.0)
    
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
# 2. Realistic VGI Simulation Engine
# ==========================================

@st.cache_data
def generate_realistic_vgi_simulation(year, num_ids, pareto_alpha, target_daily_cars, arr_shift, stay_scale, profile_arrival, profile_stay, run_id):
    """リアルな分布をベースにしたシミュレーション"""
    np.random.seed(run_id)
    id_weights = np.random.pareto(pareto_alpha, num_ids) + 1.0 
    id_weights = id_weights / id_weights.mean() 
    car_ids = np.arange(10001, 10001 + num_ids)
    
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
        
        expected_cars_today = target_daily_cars * daily_factors.get((month, w_idx), 1.0)
        day_probs = np.clip((expected_cars_today / num_ids) * id_weights, 0, 1)
        active_ids = car_ids[np.random.rand(num_ids) < day_probs]
        
        # この日のリアルな到着プロファイルを取得
        day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
        arr_probs = day_arrival['avg_cars'].values if not day_arrival.empty else np.array([1])
        arr_hours = day_arrival['in_hour'].values if not day_arrival.empty else np.array([9])
        if arr_probs.sum() > 0: arr_probs = arr_probs / arr_probs.sum()
        
        for cid in active_ids:
            # リアルな到着時間をサンプリングし、UIで指定した分だけシフトする
            base_in_h = np.random.choice(arr_hours, p=arr_probs)
            in_hour = np.clip(base_in_h + arr_shift + np.random.normal(0, 0.5), 0, 22.0)
            
            # その時間のリアルな平均滞在時間をサンプリングし、UIで指定した倍率をかける
            stay_row = profile_stay[(profile_stay['weekday'] == w_idx) & (profile_stay['in_hour'] == int(base_in_h))]
            stay_mean = stay_row['mean'].values[0] if not stay_row.empty else 4.0
            stay_std = stay_row['std'].values[0] if not stay_row.empty else 2.0
            stay_duration = max(0.5, np.random.normal(stay_mean, stay_std) * stay_scale)
            
            in_t = base_dt + timedelta(hours=int(in_hour), minutes=int((in_hour%1)*60))
            out_t = in_t + timedelta(hours=stay_duration)
            
            event_data.append({"car_id": cid, "in_time": in_t, "out_time": out_t})
            
    # Tab 1〜3で使いやすいようにフォーマットを整える
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
    """876万行を高速に生成する行列ベクトル化アプローチ"""
    times = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='h')
    num_times = len(times)
    num_ids = len(all_car_ids)
    
    # 全体を 'home' で初期化した行列 (8760行 × 1000列) を作成
    status_matrix = np.full((num_times, num_ids), 'home', dtype=object)
    car_id_to_idx = {cid: i for i, cid in enumerate(all_car_ids)}
    
    df_events['in_h'] = df_events['in_time'].dt.floor('h')
    df_events['out_h'] = df_events['out_time'].dt.floor('h')
    start_time = times[0]
    
    for _, row in df_events.iterrows():
        cid_idx = car_id_to_idx[row['car_id']]
        idx_in = int((row['in_h'] - start_time).total_seconds() // 3600)
        idx_out = int((row['out_h'] - start_time).total_seconds() // 3600)
        
        # out (往路：入庫の1時間前)
        if 0 <= idx_in - 1 < num_times: status_matrix[idx_in - 1, cid_idx] = 'out'
        
        # in (滞在)
        start_in = max(0, idx_in)
        end_in = min(num_times, idx_out)
        if start_in < end_in: status_matrix[start_in:end_in, cid_idx] = 'in'
            
        # out (復路：出庫の時刻)
        if 0 <= idx_out < num_times: status_matrix[idx_out, cid_idx] = 'out'

    # 行列を縦持ちのDataFrameに変換 (超高速)
    flat_status = status_matrix.flatten('F') # 列ごとに展開
    flat_ids = np.repeat(all_car_ids, num_times)
    flat_times = np.tile(times, num_ids)
    
    return pd.DataFrame({'datetime': flat_times, 'car_id': flat_ids, 'in_out': flat_status})

# ==========================================
# 3. UI Settings & Layout
# ==========================================

st.set_page_config(layout="wide", page_title="Parking & VGI Simulator")
st.title("🚗 駐車場分析 & VGIシミュレータ")

# データのロード（裏でプロファイルを作成）
raw_df = generate_university_sample()
profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_df)

# グローバル設定（サイドバー）
with st.sidebar:
    st.header("シミュレーション設定")
    t_yr = st.selectbox("対象年", [2025, 2026, 2027])
    num_ids = st.number_input("登録ID数 (1-10000)", 1, 10000, 1000, step=100)
    target_daily = st.number_input("1日の平均来訪台数", 10, 10000, int(base_capacity), step=10)
    
    st.divider()
    st.markdown("**行動パターンの調整**")
    pareto_alpha = st.slider("一部のユーザーへの偏り", 1.0, 5.0, 1.5, help="低いほどヘビーユーザーに偏ります")
    arr_shift = st.slider("出勤ピークのシフト (時間)", -3.0, 3.0, 0.0, 0.5, help="元の到着波形を前後にズラします")
    stay_scale = st.slider("滞在時間のスケール (倍率)", 0.5, 2.0, 1.0, 0.1, help="1.0でリアルな滞在時間、1.5で全体的に長居します")
    
    if st.button("シミュレーション実行", type="primary", use_container_width=True):
        st.session_state['sim_run_id'] = st.session_state.get('sim_run_id', 0) + 1

# 初回ロード時に自動でシミュレーションを回す
if 'sim_df' not in st.session_state or st.session_state.get('sim_run_id', 0) > 0:
    with st.spinner("シミュレーション実行中..."):
        sim_df, all_ids = generate_realistic_vgi_simulation(
            t_yr, num_ids, pareto_alpha, target_daily, arr_shift, stay_scale, 
            profile_arrival, profile_stay, st.session_state.get('sim_run_id', 0)
        )
        st.session_state['sim_df'] = sim_df
        st.session_state['all_ids'] = all_ids
        st.session_state['sim_run_id'] = 0 # リセット

# 以降のTab 1〜3はすべて `st.session_state['sim_df']` をもとに描画します
sim_df = st.session_state['sim_df']
available_dates = sorted(sim_df['date'].unique())

tab1, tab2, tab3 = st.tabs(["日別比較 (シミュレーション結果)", "全体トレンド", "VGIログ生成 (CSV出力)"])
colors = ['#2980b9', '#e67e22', '#27ae60']
plotly_template = "plotly_white"

# --- Tab 1: 日別比較 ---
with tab1:
    st.markdown("### 指定日の滞在プロファイル")
    sel_dates = st.multiselect("比較する日付を選択:", available_dates, default=available_dates[:2] if len(available_dates)>1 else available_dates)
    if sel_dates:
        cols = st.columns(len(sel_dates))
        p_data = []
        for i, dt in enumerate(sel_dates):
            day_df = sim_df[sim_df['date'] == dt]
            dow = ['月','火','水','木','金','土','日'][dt.weekday()]
            
            with cols[i]:
                st.markdown(f"<h4 style='color:{colors[i%3]}; text-align:center;'>{dt} ({dow})</h4>", unsafe_allow_html=True)
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("入庫台数", f"{len(day_df)}")
                c_m2.metric("平均滞在時間", f"{day_df['stay_duration'].mean():.1f}h" if not day_df.empty else "0h")
                
            d_p = calc_parked_cars(sim_df, dt)
            d_io = calc_in_out_hourly(sim_df, dt)
            p_data.append({'dt': dt, 'dow': dow, 'parked': d_p, 'inout': d_io, 'raw': day_df})
                
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. 滞在台数の推移**")
            f_p = go.Figure()
            for i, d in enumerate(p_data): 
                f_p.add_trace(go.Scatter(x=d['parked']['time_str'], y=d['parked']['parked_cars'], name=str(d['dt']), line=dict(color=colors[i%3], width=2.5)))
            f_p.update_layout(template=plotly_template, hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(f_p, use_container_width=True)
            
        with c2:
            st.markdown("**2. 時間別入出庫ダイナミクス**")
            f_io = go.Figure()
            for i, d in enumerate(p_data):
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=d['inout']['in_count'], name=f"IN: {d['dt']}", marker_color=colors[i%3], offsetgroup=i))
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=-d['inout']['out_count'], name=f"OUT: {d['dt']}", marker_color=colors[i%3], opacity=0.5, offsetgroup=i))
            f_io.update_layout(template=plotly_template, barmode='group', hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(f_io, use_container_width=True)
            
        st.markdown("**3. 滞在時間の分布**")
        h_cols = st.columns(len(p_data))
        for i, d in enumerate(p_data):
            with h_cols[i]:
                f_h = px.histogram(d['raw'], x='stay_duration', color_discrete_sequence=[colors[i%3]])
                f_h.update_layout(template=plotly_template, xaxis_title="h", yaxis_title="Count", xaxis_range=[0, 24], showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(f_h, use_container_width=True)

# --- Tab 2: 全体トレンド ---
with tab2:
    st.markdown("### 全体トレンド (シミュレーション結果)")
    c1, c2 = st.columns(2)
    with c1:
        m_t = sim_df.groupby('month').size().reset_index(name='t')
        d_t = sim_df.groupby('month')['date'].nunique().reset_index(name='d')
        mt = pd.merge(m_t, d_t, on='month')
        mt['avg'] = mt['t'] / mt['d']
        f_m = px.bar(mt, x='month', y='avg', title="Monthly Average", color_discrete_sequence=['#34495e'])
        f_m.update_layout(template=plotly_template, xaxis=dict(tickmode='linear'))
        st.plotly_chart(f_m, use_container_width=True)
        
    with c2:
        w_t = sim_df.groupby('weekday').size().reset_index(name='t')
        d_w = sim_df.groupby('weekday')['date'].nunique().reset_index(name='d')
        wt = pd.merge(w_t, d_w, on='weekday')
        wt['avg'] = wt['t'] / wt['d']
        wt['dw'] = wt['weekday'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
        f_w = px.bar(wt, x='dw', y='avg', title="Weekday Average", color_discrete_sequence=['#7f8c8d'])
        f_w.update_layout(template=plotly_template)
        st.plotly_chart(f_w, use_container_width=True)

# --- Tab 3: VGIログ生成 ---
with tab3:
    st.markdown("### VGIシステム用 状態遷移ログエクスポート")
    st.info("サイドバーで設定したシミュレーションの全IDの1年分の推移（約876万行）を、指定の `home / in / out` フォーマットで展開します。")
    
    if st.button("詳細ログを展開してダウンロード準備", type="primary"):
        with st.spinner("876万行の行列を展開中... (数秒かかります)"):
            df_vgi = convert_to_full_vgi_format(st.session_state['sim_df'], t_yr, st.session_state['all_ids'])
            
            st.success(f"展開完了: {len(df_vgi):,} 行のデータを生成しました。")
            st.markdown("#### データプレビュー (先頭1000行)")
            st.dataframe(df_vgi.head(1000), use_container_width=True)
            
            # メモリに優しい形式でCSV変換
            st.download_button(
                label="VGIフォーマットCSVをダウンロード", 
                data=df_vgi.to_csv(index=False).encode('utf-8'), 
                file_name=f"vgi_sim_log_{t_yr}.csv", 
                mime="text/csv", 
                use_container_width=True
            )
