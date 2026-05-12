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
    np.random.seed(42)
    start_date = datetime(2025, 4, 1)
    days = 365
    data = []
    
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        is_weekend = current_date.weekday() >= 5
        
        if is_weekend: 
            n_cars = np.random.randint(5, 15)
        elif month in [8, 9, 2, 3]: 
            n_cars = np.random.randint(30, 60)
        else: 
            n_cars = np.random.randint(80, 150)
            
        if n_cars == 0: 
            continue
            
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
    
    df['date'] = df['in_time'].dt.date
    df['month'] = df['in_time'].dt.month
    df['weekday'] = df['in_time'].dt.weekday
    df['in_hour'] = df['in_time'].dt.hour
    df['out_hour'] = df['out_time'].dt.hour
    
    days_count = df.groupby(['month', 'weekday'])['in_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='num_days')
    arrivals = df.groupby(['month', 'weekday', 'in_hour']).size().reset_index(name='total_cars')
    profile_arrival = pd.merge(arrivals, days_count, on=['month', 'weekday'])
    profile_arrival['avg_cars'] = profile_arrival['total_cars'] / profile_arrival['num_days']
    
    profile_stay = df.groupby(['weekday', 'in_hour'])['stay_duration'].agg(['mean', 'std']).reset_index()
    profile_stay['std'] = profile_stay['std'].fillna(1.0)
    
    daily_counts = df.groupby('date').size()
    base_capacity = daily_counts.quantile(0.95) if not daily_counts.empty else 21.0
    
    return df, profile_arrival, profile_stay, base_capacity

def calc_parked_cars(df, target_date, freq='10min'):
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = start_dt + timedelta(hours=23, minutes=50) 
    time_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    
    if df.empty: 
        return pd.DataFrame({"time_str": [t.strftime('%H:%M') for t in time_range], "parked_cars": 0})
        
    counts = []
    for t in time_range:
        mask = (df['in_time'] <= t) & (df['out_time'] > t)
        counts.append(mask.sum())
        
    counts = np.array(counts) - counts[0]
    
    return pd.DataFrame({
        "time_str": [t.strftime('%H:%M') for t in time_range], 
        "parked_cars": counts
    })

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
# 2. Generative Simulation Engines
# ==========================================

@st.cache_data
def generate_daily_simulation(target_date, target_capacity, base_capacity, profile_arrival, profile_stay, run_id):
    """旧仕様のシミュレーション（Tab 3用: 全体トレンド比較用として維持）"""
    multiplier = target_capacity / base_capacity if base_capacity > 0 else 1.0
    target_data = []
    
    w_idx = target_date.weekday()
    month = target_date.month
    base_dt = datetime.combine(target_date, datetime.min.time())
    
    day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == w_idx)]
    if day_arrival.empty: 
        day_arrival = profile_arrival[profile_arrival['weekday'] == w_idx].groupby('in_hour')['avg_cars'].mean().reset_index()
        
    for hour in range(24):
        arr_row = day_arrival[day_arrival['in_hour'] == hour]
        if arr_row.empty: 
            continue
            
        expected_cars = arr_row['avg_cars'].values[0] * multiplier
        n_cars = np.random.poisson(expected_cars)
        
        stay_row = profile_stay[(profile_stay['weekday'] == w_idx) & (profile_stay['in_hour'] == hour)]
        stay_mean = stay_row['mean'].values[0] if not stay_row.empty else 4.0
        stay_std = stay_row['std'].values[0] if not stay_row.empty else 2.0
        
        for _ in range(n_cars):
            in_t = base_dt + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            stay_h = max(0.5, min(np.random.normal(stay_mean, stay_std), 24.0))
            out_t = in_t + timedelta(hours=stay_h)
            
            target_data.append({
                "in_time": in_t, 
                "out_time": out_t, 
                "stay_duration": stay_h
            })
            
    df = pd.DataFrame(target_data)
    if not df.empty:
        df['in_time'] = pd.to_datetime(df['in_time'])
        df['out_time'] = pd.to_datetime(df['out_time'])
    return df

@st.cache_data
def generate_vgi_annual_simulation(year, num_ids, pareto_alpha, arr_peak_mean, arr_peak_std, dep_peak_mean, dep_peak_std, base_capacity, profile_arrival, run_id):
    """IDベース＆ピーク制御を備えた新シミュレーションエンジン（Tab 4用）"""
    np.random.seed(run_id)
    
    # IDプールの生成とパレート分布に基づく来訪確率ウェイトの割り当て
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
            if day_arrival.empty:
                daily_cars = base_capacity * 0.5
            else:
                daily_cars = day_arrival['avg_cars'].sum()
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
            in_hour = np.random.normal(arr_peak_mean, arr_peak_std)
            in_hour = np.clip(in_hour, 0, 22.0)
            
            dep_hour = np.random.normal(dep_peak_mean, dep_peak_std)
            if dep_hour <= in_hour + 1.0:
                dep_hour = in_hour + max(1.0, np.random.normal(2.0, 0.5))
            dep_hour = np.clip(dep_hour, in_hour + 0.5, 23.99)
            
            in_t = base_dt + timedelta(hours=int(in_hour), minutes=int((in_hour%1)*60))
            out_t = base_dt + timedelta(hours=int(dep_hour), minutes=int((dep_hour%1)*60))
            
            event_data.append({
                "car_id": cid,
                "in_time": in_t,
                "out_time": out_t
            })
            
    df_events = pd.DataFrame(event_data)
    return df_events, car_ids

@st.cache_data
def convert_to_vgi_hourly_format(df_events, year, all_car_ids):
    """イベントログをVGI指定の[datetime, car_id, in_out]の1時間ごとの密な形式に変換する"""
    if df_events.empty:
        return pd.DataFrame()
    
    # Pandas 2.2+ 対応： 'H' を 'h' に変更
    df_events['in_h'] = df_events['in_time'].dt.floor('h')
    df_events['out_h'] = df_events['out_time'].dt.floor('h')
    
    records = []
    for _, row in df_events.iterrows():
        cid = row['car_id']
        t_in = row['in_h']
        t_out = row['out_h']
        
        # 出勤移動 (out)
        records.append({"datetime": t_in - timedelta(hours=1), "car_id": cid, "in_out": "out"})
        
        # 施設滞在 (in)
        curr_t = t_in
        while curr_t < t_out:
            records.append({"datetime": curr_t, "car_id": cid, "in_out": "in"})
            curr_t += timedelta(hours=1)
            
        # 退勤移動 (out)
        records.append({"datetime": t_out, "car_id": cid, "in_out": "out"})
        
    df_dense = pd.DataFrame(records).sort_values(['datetime', 'car_id']).reset_index(drop=True)
    return df_dense

# ==========================================
# 3. UI Settings & Layout
# ==========================================

LANG = {
    "日本語": {
        "title": "駐車場分析 & VGIシミュレータ", "data_source": "データソース設定", 
        "select_source": "ソースを選択してください:", "sample_univ": "サンプル: 大学データ", 
        "upload_csv": "CSVをアップロード", "upload_help": "1年分の入出庫ログ(CSV)をアップロード", 
        "load_success": "データの読み込みに成功しました", "est_cap": "推定駐車枠数", 
        "total_rec": "総レコード数", "tab_compare": "日別比較", "tab_trend": "全体トレンド", 
        "tab_daily_sim": "既存シミュレーション維持", "tab_annual_sim": "VGI 年間シミュレーション", 
        "compare_header": "日別プロファイルの比較", "select_dates": "比較する日付を選択:", 
        "metric_arr": "入庫台数", "metric_stay": "平均滞在時間", "graph_parked": "滞在台数の推移", 
        "graph_inout": "時間別入出庫ダイナミクス", "graph_dist": "滞在時間の分布", 
        "regen": "再生成", "target_cap": "目標の駐車台数", "gen_btn": "VGIログを生成する", 
        "download": "CSVをダウンロード"
    }
}

st.set_page_config(layout="wide", page_title="Parking & VGI Simulator")
st.markdown("<style>h1, h2, h3, h4 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }</style>", unsafe_allow_html=True)

T = LANG["日本語"]
st.title(T["title"])

with st.sidebar:
    st.header(T["data_source"])
    data_mode = st.radio(T["select_source"], [T["sample_univ"], T["upload_csv"]])
    raw_df = None
    if data_mode == T["upload_csv"]:
        uploaded_file = st.file_uploader(T["upload_help"], type=['csv'])
        if uploaded_file: 
            raw_df = pd.read_csv(uploaded_file)
        else: 
            st.stop()
    else:
        with st.spinner("Processing..."): 
            raw_df = generate_university_sample()
        st.success(T["load_success"])

df, profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_df)
available_dates = sorted(df['date'].unique())

st.sidebar.divider()
st.sidebar.markdown(f"**{T['est_cap']}:** {int(base_capacity)} cars/day")
st.sidebar.markdown(f"**{T['total_rec']}:** {len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs([T["tab_compare"], T["tab_trend"], T["tab_daily_sim"], T["tab_annual_sim"]])
colors = ['#2980b9', '#e67e22', '#27ae60']
plotly_template = "plotly_white"

# ==========================================
# Tab 1: 日別比較 (完全復元)
# ==========================================
with tab1:
    st.markdown(f"### {T['compare_header']}")
    sel_dates = st.multiselect(T["select_dates"], available_dates, default=available_dates[:2] if len(available_dates)>1 else available_dates)
    if sel_dates:
        cols = st.columns(len(sel_dates))
        p_data = []
        m_p = 0
        
        for i, dt in enumerate(sel_dates):
            day_df = df[df['in_time'].dt.date == dt]
            dow = ['月','火','水','木','金','土','日'][dt.weekday()]
            
            with cols[i]:
                st.markdown(f"<h4 style='color:{colors[i%3]}; text-align:center;'>{dt} ({dow})</h4>", unsafe_allow_html=True)
                c_m1, c_m2 = st.columns(2)
                c_m1.metric(T["metric_arr"], f"{len(day_df)}")
                c_m2.metric(T["metric_stay"], f"{day_df['stay_duration'].mean():.1f}h" if not day_df.empty else "0h")
                
            d_p = calc_parked_cars(df, dt)
            d_io = calc_in_out_hourly(df, dt)
            p_data.append({'dt': dt, 'dow': dow, 'parked': d_p, 'inout': d_io, 'raw': day_df})
            if not d_p.empty: 
                m_p = max(m_p, d_p['parked_cars'].max(), abs(d_p['parked_cars'].min()))
                
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"**1. {T['graph_parked']}**")
            f_p = go.Figure()
            for i, d in enumerate(p_data): 
                f_p.add_trace(go.Scatter(x=d['parked']['time_str'], y=d['parked']['parked_cars'], name=str(d['dt']), line=dict(color=colors[i%3], width=2.5)))
            f_p.update_layout(template=plotly_template, hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(f_p, use_container_width=True)
            
        with c2:
            st.markdown(f"**2. {T['graph_inout']}**")
            f_io = go.Figure()
            for i, d in enumerate(p_data):
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=d['inout']['in_count'], name=f"IN: {d['dt']}", marker_color=colors[i%3], offsetgroup=i))
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=-d['inout']['out_count'], name=f"OUT: {d['dt']}", marker_color=colors[i%3], opacity=0.5, offsetgroup=i))
            f_io.update_layout(template=plotly_template, barmode='group', hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(f_io, use_container_width=True)
            
        st.markdown(f"**3. {T['graph_dist']}**")
        h_cols = st.columns(len(p_data))
        for i, d in enumerate(p_data):
            with h_cols[i]:
                f_h = px.histogram(d['raw'], x='stay_duration', color_discrete_sequence=[colors[i%3]])
                f_h.update_layout(template=plotly_template, xaxis_title="h", yaxis_title="Count", xaxis_range=[0, 24], showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(f_h, use_container_width=True)

# ==========================================
# Tab 2: 全体トレンド (完全復元)
# ==========================================
with tab2:
    st.markdown(f"### {T['tab_trend']}")
    c1, c2 = st.columns(2)
    with c1:
        m_t = df.groupby('month').size().reset_index(name='t')
        d_t = df.groupby('month')['date'].nunique().reset_index(name='d')
        mt = pd.merge(m_t, d_t, on='month')
        mt['avg'] = mt['t'] / mt['d']
        f_m = px.bar(mt, x='month', y='avg', title="Monthly Average", color_discrete_sequence=['#34495e'])
        f_m.update_layout(template=plotly_template, xaxis=dict(tickmode='linear'))
        st.plotly_chart(f_m, use_container_width=True)
        
    with c2:
        w_t = df.groupby('weekday').size().reset_index(name='t')
        d_w = df.groupby('weekday')['date'].nunique().reset_index(name='d')
        wt = pd.merge(w_t, d_w, on='weekday')
        wt['avg'] = wt['t'] / wt['d']
        wt['dw'] = wt['weekday'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
        f_w = px.bar(wt, x='dw', y='avg', title="Weekday Average", color_discrete_sequence=['#7f8c8d'])
        f_w.update_layout(template=plotly_template)
        st.plotly_chart(f_w, use_container_width=True)

# ==========================================
# Tab 3: 日別シミュレーション (完全復元)
# ==========================================
with tab3:
    st.markdown(f"### {T['tab_daily_sim']}")
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: 
        comp_dt = st.selectbox(T["select_dates"], available_dates, key="sb_daily")
    with c_s2: 
        s_cap = st.number_input(T["target_cap"], 10, 5000, int(base_capacity))
    with c_s3: 
        st.write("") 
        if st.button(T["regen"], use_container_width=True): 
            st.session_state['daily_run_id'] += 1
            
    s_df = generate_daily_simulation(comp_dt, s_cap, base_capacity, profile_arrival, profile_stay, st.session_state['daily_run_id'])
    r_df = df[df['in_time'].dt.date == comp_dt]
    
    r_p = calc_parked_cars(r_df, comp_dt)
    s_p = calc_parked_cars(s_df, comp_dt)
    
    st.divider()
    f_c = go.Figure()
    f_c.add_trace(go.Scatter(x=r_p['time_str'], y=r_p['parked_cars'], name='Real', line=dict(color='#7f8c8d', dash='dash')))
    f_c.add_trace(go.Scatter(x=s_p['time_str'], y=s_p['parked_cars'], name='Sim', line=dict(color='#2980b9', width=3)))
    f_c.update_layout(template=plotly_template, xaxis_title="Time", yaxis_title="Cars", hovermode="x unified")
    st.plotly_chart(f_c, use_container_width=True)

# ==========================================
# Tab 4: VGI向け 年間シミュレーション
# ==========================================
with tab4:
    st.markdown(f"### {T['tab_annual_sim']}")
    st.info("総台数の指定を廃止し、**「登録ID数」「頻度の偏り」「出退勤ピーク」**のつまみで各ユーザーの振る舞いを生成します。")
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            t_yr = st.selectbox("対象年", [2026, 2027])
            num_ids = st.number_input("登録ID数 (1-10000)", 1, 10000, 1000, step=100)
        with c2:
            pareto_alpha = st.slider("ユーザー頻度の偏り (パレート係数)", 1.0, 5.0, 1.5, 
                                     help="1.0に近いほど一部のヘビーユーザーに偏り、5.0に近いほど全員が均等に利用します。")
        with c3:
            arr_mean = st.slider("出勤ピーク (平均時刻)", 6.0, 15.0, 9.0, 0.5)
            arr_std = st.slider("出勤ピークの幅 (標準偏差)", 0.5, 3.0, 1.0, 0.1)
        with c4:
            dep_mean = st.slider("退勤ピーク (平均時刻)", 12.0, 22.0, 17.0, 0.5)
            dep_std = st.slider("退勤ピークの幅 (標準偏差)", 0.5, 3.0, 1.5, 0.1)
            
        if st.button(T["gen_btn"], type="primary", use_container_width=True): 
            st.session_state['annual_run_id'] += 1

    if st.session_state['annual_run_id'] > 0:
        with st.spinner("シミュレーション実行中..."):
            df_events, all_ids = generate_vgi_annual_simulation(
                t_yr, num_ids, pareto_alpha, 
                arr_mean, arr_std, dep_mean, dep_std, 
                base_capacity, profile_arrival, st.session_state['annual_run_id']
            )
            
            st.success(f"年間延べ利用回数: {len(df_events):,} 回分のイベントを生成しました。")
            
            df_vgi = convert_to_vgi_hourly_format(df_events, t_yr, all_ids)
            
            st.markdown("#### 生成結果プレビュー (1時間ごとのステータス)")
            st.caption("※ UIのフリーズを防ぐため先頭1000行のみ表示しています。全データはダウンロード可能です。(未表示の時間帯はシミュレータ側で 'home' として扱ってください)")
            st.dataframe(df_vgi.head(1000), use_container_width=True)
            
            st.download_button(
                label="VGIフォーマットをダウンロード (CSV)", 
                data=df_vgi.to_csv(index=False).encode('utf-8'), 
                file_name=f"vgi_sim_log_{t_yr}.csv", 
                mime="text/csv", 
                use_container_width=True
            )
