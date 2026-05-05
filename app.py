import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Session State Initialization ---
if 'daily_run_id' not in st.session_state:
    st.session_state['daily_run_id'] = 0
if 'annual_run_id' not in st.session_state:
    st.session_state['annual_run_id'] = 0

# --- 1. Data Processing Functions ---

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
        in_hours = np.concatenate([np.random.normal(9.0, 1.0, n_morning), np.random.normal(13.5, 1.5, n_afternoon)])
        in_hours = np.clip(in_hours, 0, 23.99)
        stay_durations = np.clip(np.random.normal(5.0, 2.5, n_cars), 0.5, 15.0)
        for i in range(n_cars):
            h, m = int(in_hours[i]), int((in_hours[i] % 1) * 60)
            in_time = current_date + timedelta(hours=h, minutes=m)
            out_time = in_time + timedelta(hours=stay_durations[i])
            data.append({"in_time": in_time.strftime("%Y-%m-%d %H:%M:%S"), "out_time": out_time.strftime("%Y-%m-%d %H:%M:%S")})
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
    time_range = pd.date_range(start=start_dt, end=start_dt + timedelta(days=1), freq=freq)
    if df.empty: return pd.DataFrame({"time_str": [t.strftime('%H:%M') for t in time_range], "parked_cars": 0})
    counts = [( (df['in_time'] <= t) & (df['out_time'] > t) ).sum() for t in time_range]
    return pd.DataFrame({"time_str": [t.strftime('%H:%M') for t in time_range], "parked_cars": counts})

def calc_in_out_hourly(df, target_date):
    hours = pd.DataFrame({'hour': range(24)})
    if df.empty:
        hours['in_count'] = 0; hours['out_count'] = 0; hours['time_str'] = hours['hour'].apply(lambda x: f"{x:02d}:00")
        return hours
    in_df = df[df['in_time'].dt.date == target_date].groupby('in_hour').size().reset_index(name='in_count')
    out_df = df[df['out_time'].dt.date == target_date].groupby('out_hour').size().reset_index(name='out_count')
    res = pd.merge(hours, in_df, left_on='hour', right_on='in_hour', how='left')
    res = pd.merge(res, out_df, left_on='hour', right_on='out_hour', how='left').fillna(0)
    res['time_str'] = res['hour'].apply(lambda x: f"{x:02d}:00")
    return res

@st.cache_data
def generate_daily_simulation(target_date, target_capacity, base_capacity, profile_arrival, profile_stay, run_id):
    multiplier = target_capacity / base_capacity if base_capacity > 0 else 1.0
    target_data = []
    day_arrival = profile_arrival[(profile_arrival['month'] == target_date.month) & (profile_arrival['weekday'] == target_date.weekday())]
    if day_arrival.empty: day_arrival = profile_arrival[profile_arrival['weekday'] == target_date.weekday()].groupby('in_hour')['avg_cars'].mean().reset_index()
    base_dt = datetime.combine(target_date, datetime.min.time())
    for hour in range(24):
        arr_row = day_arrival[day_arrival['in_hour'] == hour]
        if arr_row.empty: continue
        n_cars = np.random.poisson(arr_row['avg_cars'].values[0] * multiplier)
        stay_row = profile_stay[(profile_stay['weekday'] == target_date.weekday()) & (profile_stay['in_hour'] == hour)]
        stay_mean, stay_std = (stay_row['mean'].values[0], stay_row['std'].values[0]) if not stay_row.empty else (4.0, 2.0)
        for _ in range(n_cars):
            in_t = base_dt + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            out_t = in_t + timedelta(hours=max(0.5, min(np.random.normal(stay_mean, stay_std), 24.0)))
            target_data.append({"in_time": in_t, "out_time": out_t, "stay_duration": (out_t - in_t).total_seconds()/3600.0})
    df = pd.DataFrame(target_data)
    if not df.empty:
        df['in_time'] = pd.to_datetime(df['in_time']); df['out_time'] = pd.to_datetime(df['out_time'])
    return df

@st.cache_data
def generate_annual_simulation(year, target_capacity, base_capacity, profile_arrival, profile_stay, run_id):
    multiplier = target_capacity / base_capacity if base_capacity > 0 else 1.0
    start_date = datetime(year, 1, 1)
    days_in_year = (datetime(year, 12, 31) - start_date).days + 1
    target_data = []
    for d in range(days_in_year):
        cur_date = start_date + timedelta(days=d)
        day_arrival = profile_arrival[(profile_arrival['month'] == cur_date.month) & (profile_arrival['weekday'] == cur_date.weekday())]
        if day_arrival.empty: day_arrival = profile_arrival[profile_arrival['weekday'] == cur_date.weekday()].groupby('in_hour')['avg_cars'].mean().reset_index()
        base_dt = datetime.combine(cur_date, datetime.min.time())
        for hour in range(24):
            arr_row = day_arrival[day_arrival['in_hour'] == hour]
            if arr_row.empty: continue
            n_cars = np.random.poisson(arr_row['avg_cars'].values[0] * multiplier)
            stay_row = profile_stay[(profile_stay['weekday'] == cur_date.weekday()) & (profile_stay['in_hour'] == hour)]
            stay_mean, stay_std = (stay_row['mean'].values[0], stay_row['std'].values[0]) if not stay_row.empty else (4.0, 2.0)
            for _ in range(n_cars):
                in_t = base_dt + timedelta(hours=hour, minutes=np.random.randint(0, 60))
                out_t = in_t + timedelta(hours=max(0.5, min(np.random.normal(stay_mean, stay_std), 24.0)))
                target_data.append({"in_time": in_t, "out_time": out_t})
    df = pd.DataFrame(target_data)
    if not df.empty:
        df['in_time'] = pd.to_datetime(df['in_time']); df['out_time'] = pd.to_datetime(df['out_time'])
        df['month'] = df['in_time'].dt.month; df['date'] = df['in_time'].dt.date
    return df

# --- 2. UI Localization ---
LANG = {
    "日本語": {
        "title": "駐車場分析ダッシュボード",
        "data_source": "データソース設定",
        "select_source": "ソースを選択してください:",
        "sample_univ": "サンプル: 大学データ",
        "upload_csv": "CSVをアップロード",
        "upload_help": "1年分の入出庫ログ(CSV)をアップロードしてください",
        "load_success": "データの読み込みに成功しました",
        "est_cap": "推定駐車枠数",
        "total_rec": "総レコード数",
        "tab_compare": "日別比較",
        "tab_trend": "全体トレンド",
        "tab_daily_sim": "実データ vs シミュレーション",
        "tab_annual_sim": "年間シミュレーション",
        "compare_header": "日別プロファイルの比較",
        "select_dates": "比較する日付を選択:",
        "metric_arr": "入庫台数",
        "metric_stay": "平均滞在時間",
        "graph_parked": "滞在台数の推移",
        "graph_inout": "時間別入出庫ダイナミクス",
        "graph_dist": "滞在時間の分布",
        "regen": "再生成",
        "gen_btn": "年間シミュレーションを実行",
        "target_cap": "目標の駐車台数",
        "download": "CSVをダウンロード"
    },
    "English": {
        "title": "Parking Intelligence Dashboard",
        "data_source": "Data Source",
        "select_source": "Select Source:",
        "sample_univ": "Sample: University Data",
        "upload_csv": "Upload CSV",
        "upload_help": "Upload 1-year in/out log (CSV)",
        "load_success": "Data loaded successfully",
        "est_cap": "Estimated Capacity",
        "total_rec": "Total Records",
        "tab_compare": "Compare Profiles",
        "tab_trend": "Seasonal Trends",
        "tab_daily_sim": "Daily Sim vs Real",
        "tab_annual_sim": "Annual Simulation",
        "compare_header": "Daily Profile Comparison",
        "select_dates": "Select dates to compare:",
        "metric_arr": "Arrivals",
        "metric_stay": "Avg Stay",
        "graph_parked": "Parked Vehicles",
        "graph_inout": "Hourly In/Out Dynamics",
        "graph_dist": "Stay Duration Distribution",
        "regen": "Regenerate Sim",
        "gen_btn": "Generate 1-Year Simulation",
        "target_cap": "Target Capacity",
        "download": "Download Data (CSV)"
    }
}

# --- 3. UI Setup ---
st.set_page_config(layout="wide", page_title="Parking Intelligence")
st.markdown("<style>h1, h2, h3, h4 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }</style>", unsafe_allow_html=True)

# Language Selector
sel_lang = st.sidebar.selectbox("Language / 言語", ["日本語", "English"])
T = LANG[sel_lang]

st.title(T["title"])

with st.sidebar:
    st.header(T["data_source"])
    data_mode = st.radio(T["select_source"], [T["sample_univ"], T["upload_csv"]])
    raw_df = None
    if data_mode == T["upload_csv"]:
        uploaded_file = st.file_uploader(T["upload_help"], type=['csv'])
        if uploaded_file: raw_df = pd.read_csv(uploaded_file)
        else: st.stop()
    else:
        with st.spinner("Processing..."): raw_df = generate_university_sample()
        st.success(T["load_success"])

df, profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_df)
available_dates = sorted(df['date'].unique())
st.sidebar.divider()
st.sidebar.markdown(f"**{T['est_cap']}:** {int(base_capacity)} cars/day")
st.sidebar.markdown(f"**{T['total_rec']}:** {len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs([T["tab_compare"], T["tab_trend"], T["tab_daily_sim"], T["tab_annual_sim"]])
colors = ['#2980b9', '#e67e22', '#27ae60']; plotly_template = "plotly_white"

with tab1:
    st.markdown(f"### {T['compare_header']}")
    sel_dates = st.multiselect(T["select_dates"], available_dates, default=available_dates[:2] if len(available_dates)>1 else available_dates)
    if sel_dates:
        cols = st.columns(len(sel_dates)); p_data = []; m_p, m_io = 0, 0
        for i, dt in enumerate(sel_dates):
            day_df = df[df['in_time'].dt.date == dt]
            dow = (['月','火','水','木','金','土','日'] if sel_lang=="日本語" else ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])[dt.weekday()]
            with cols[i]:
                st.markdown(f"<h4 style='color:{colors[i%3]}; text-align:center;'>{dt} ({dow})</h4>", unsafe_allow_html=True)
                c_m1, c_m2 = st.columns(2)
                c_m1.metric(T["metric_arr"], f"{len(day_df)}")
                c_m2.metric(T["metric_stay"], f"{day_df['stay_duration'].mean():.1f}h" if not day_df.empty else "0h")
            d_p = calc_parked_cars(df, dt); d_io = calc_in_out_hourly(df, dt)
            p_data.append({'dt': dt, 'dow': dow, 'parked': d_p, 'inout': d_io, 'raw': day_df})
            if not d_p.empty: m_p = max(m_p, d_p['parked_cars'].max())
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**1. {T['graph_parked']}**")
            f_p = go.Figure()
            for i, d in enumerate(p_data): f_p.add_trace(go.Scatter(x=d['parked']['time_str'], y=d['parked']['parked_cars'], name=str(d['dt']), line=dict(color=colors[i%3], width=2.5)))
            f_p.update_layout(template=plotly_template, yaxis_range=[0, m_p*1.1], hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20)); st.plotly_chart(f_p, use_container_width=True)
        with c2:
            st.markdown(f"**2. {T['graph_inout']}**")
            f_io = go.Figure()
            for i, d in enumerate(p_data):
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=d['inout']['in_count'], name=f"IN: {d['dt']}", marker_color=colors[i%3], offsetgroup=i))
                f_io.add_trace(go.Bar(x=d['inout']['time_str'], y=-d['inout']['out_count'], name=f"OUT: {d['dt']}", marker_color=colors[i%3], opacity=0.5, offsetgroup=i))
            f_io.update_layout(template=plotly_template, barmode='group', hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20)); st.plotly_chart(f_io, use_container_width=True)
        st.markdown(f"**3. {T['graph_dist']}**")
        h_cols = st.columns(len(p_data))
        for i, d in enumerate(p_data):
            with h_cols[i]:
                f_h = px.histogram(d['raw'], x='stay_duration', color_discrete_sequence=[colors[i%3]])
                f_h.update_layout(template=plotly_template, xaxis_title="h", yaxis_title="Count", xaxis_range=[0, 24], showlegend=False, margin=dict(l=10,r=10,t=10,b=10)); st.plotly_chart(f_h, use_container_width=True)

with tab2:
    st.markdown(f"### {T['tab_trend']}")
    c1, c2 = st.columns(2)
    with c1:
        m_t = df.groupby('month').size().reset_index(name='t'); d_t = df.groupby('month')['date'].nunique().reset_index(name='d')
        mt = pd.merge(m_t, d_t, on='month'); mt['avg'] = mt['t']/mt['d']
        f_m = px.bar(mt, x='month', y='avg', title="Monthly Average", color_discrete_sequence=['#34495e']); f_m.update_layout(template=plotly_template, xaxis=dict(tickmode='linear')); st.plotly_chart(f_m, use_container_width=True)
    with c2:
        w_t = df.groupby('weekday').size().reset_index(name='t'); d_w = df.groupby('weekday')['date'].nunique().reset_index(name='d')
        wt = pd.merge(w_t, d_w, on='weekday'); wt['avg'] = wt['t']/wt['d']
        wt['dw'] = wt['weekday'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        f_w = px.bar(wt, x='dw', y='avg', title="Weekday Average", color_discrete_sequence=['#7f8c8d']); f_w.update_layout(template=plotly_template); st.plotly_chart(f_w, use_container_width=True)

with tab3:
    st.markdown(f"### {T['tab_daily_sim']}")
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: comp_dt = st.selectbox(T["select_dates"], available_dates)
    with c_s2: s_cap = st.number_input(T["target_cap"], 10, 5000, int(base_capacity))
    with c_s3: st.write(""); 
        if st.button(T["regen"], use_container_width=True): st.session_state['daily_run_id'] += 1
    s_df = generate_daily_simulation(comp_dt, s_cap, base_capacity, profile_arrival, profile_stay, st.session_state['daily_run_id'])
    r_df = df[df['in_time'].dt.date == comp_dt]
    r_p = calc_parked_cars(r_df, comp_dt); s_p = calc_parked_cars(s_df, comp_dt)
    st.divider()
    f_c = go.Figure()
    f_c.add_trace(go.Scatter(x=r_p['time_str'], y=r_p['parked_cars'], name='Real', line=dict(color='#7f8c8d', dash='dash')))
    f_c.add_trace(go.Scatter(x=s_p['time_str'], y=s_p['parked_cars'], name='Sim', line=dict(color='#2980b9', width=3)))
    f_c.update_layout(template=plotly_template, xaxis_title="Time", yaxis_title="Cars", hovermode="x unified"); st.plotly_chart(f_c, use_container_width=True)

with tab4:
    st.markdown(f"### {T['tab_annual_sim']}")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1,1,2,1])
        with c1: t_yr = st.selectbox("Year", [2026, 2027])
        with c2: st.metric("Base", int(base_capacity))
        with c3: t_cap = st.number_input(T["target_cap"], 10, 5000, int(base_capacity)*2)
        with c4: st.write(""); 
            if st.button("RUN", type="primary", use_container_width=True): st.session_state['annual_run_id'] += 1
    if st.session_state['annual_run_id'] > 0:
        sim_yr = generate_annual_simulation(t_yr, t_cap, base_capacity, profile_arrival, profile_stay, st.session_state['annual_run_id'])
        st.success(f"Generated: {len(sim_yr):,} records")
        st.download_button(T["download"], sim_yr[['in_time', 'out_time']].to_csv(index=False).encode('utf-8'), f"sim_{t_yr}.csv", "text/csv", use_container_width=True)
