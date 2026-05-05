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
        
        in_hours = np.concatenate([
            np.random.normal(9.0, 1.0, n_morning),
            np.random.normal(13.5, 1.5, n_afternoon)
        ])
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
    """【修正】freqを '10T' から '10min' に変更"""
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
    month = target_date.month
    weekday = target_date.weekday()
    
    day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == weekday)]
    if day_arrival.empty:
        day_arrival = profile_arrival[profile_arrival['weekday'] == weekday].groupby('in_hour')['avg_cars'].mean().reset_index()
        
    base_dt = datetime.combine(target_date, datetime.min.time())
    for hour in range(24):
        arr_row = day_arrival[day_arrival['in_hour'] == hour]
        if arr_row.empty: continue
        
        expected = arr_row['avg_cars'].values[0] * multiplier
        n_cars = np.random.poisson(expected)
        if n_cars == 0: continue
        
        stay_row = profile_stay[(profile_stay['weekday'] == weekday) & (profile_stay['in_hour'] == hour)]
        stay_mean = stay_row['mean'].values[0] if not stay_row.empty else 4.0
        stay_std = stay_row['std'].values[0] if not stay_row.empty else 2.0
        
        for _ in range(n_cars):
            in_time = base_dt + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            out_time = in_time + timedelta(hours=max(0.5, min(np.random.normal(stay_mean, stay_std), 24.0)))
            target_data.append({"in_time": in_time, "out_time": out_time, "stay_duration": (out_time - in_time).total_seconds()/3600.0})
            
    df = pd.DataFrame(target_data)
    if not df.empty:
        df['in_time'] = pd.to_datetime(df['in_time'])
        df['out_time'] = pd.to_datetime(df['out_time'])
    return df

@st.cache_data
def generate_annual_simulation(year, target_capacity, base_capacity, profile_arrival, profile_stay, run_id):
    multiplier = target_capacity / base_capacity if base_capacity > 0 else 1.0
    start_date = datetime(year, 1, 1)
    days_in_year = (datetime(year, 12, 31) - start_date).days + 1
    target_data = []
    
    for d in range(days_in_year):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        weekday = current_date.weekday()
        
        day_arrival = profile_arrival[(profile_arrival['month'] == month) & (profile_arrival['weekday'] == weekday)]
        if day_arrival.empty:
            day_arrival = profile_arrival[profile_arrival['weekday'] == weekday].groupby('in_hour')['avg_cars'].mean().reset_index()
            
        base_dt = datetime.combine(current_date, datetime.min.time())
        for hour in range(24):
            arr_row = day_arrival[day_arrival['in_hour'] == hour]
            if arr_row.empty: continue
            
            expected = arr_row['avg_cars'].values[0] * multiplier
            n_cars = np.random.poisson(expected)
            if n_cars == 0: continue
            
            stay_row = profile_stay[(profile_stay['weekday'] == weekday) & (profile_stay['in_hour'] == hour)]
            stay_mean = stay_row['mean'].values[0] if not stay_row.empty else 4.0
            stay_std = stay_row['std'].values[0] if not stay_row.empty else 2.0
            
            for _ in range(n_cars):
                in_time = base_dt + timedelta(hours=hour, minutes=np.random.randint(0, 60))
                out_time = in_time + timedelta(hours=max(0.5, min(np.random.normal(stay_mean, stay_std), 24.0)))
                target_data.append({"in_time": in_time, "out_time": out_time})
                
    df = pd.DataFrame(target_data)
    if not df.empty:
        df['in_time'] = pd.to_datetime(df['in_time'])
        df['out_time'] = pd.to_datetime(df['out_time'])
        df['month'] = df['in_time'].dt.month
        df['date'] = df['in_time'].dt.date
    return df


# --- 2. UI Setup ---
st.set_page_config(layout="wide", page_title="Parking Intelligence Dashboard")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3, h4 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; color: #2c3e50; }
    .stMetric-value { color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

st.title("Parking Intelligence Dashboard")

with st.sidebar:
    st.header("Data Source")
    data_mode = st.radio("Select Source:", ["Sample: University Data", "Upload CSV"])
    
    raw_df = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload 1-Year In/Out Log (CSV)", type=['csv'])
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
    else:
        with st.spinner("Generating sample university data..."):
            raw_df = generate_university_sample()
        st.success("Sample data loaded successfully.")

df, profile_arrival, profile_stay, base_capacity = process_and_profile_data(raw_df)
available_dates = sorted(df['date'].unique())

st.sidebar.divider()
st.sidebar.markdown(f"**Estimated Base Capacity:** Approx. {int(base_capacity)} cars/day")
st.sidebar.markdown(f"**Total Records:** {len(df):,} records")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Profile Comparison", 
    "Seasonal Trends", 
    "Daily Sim vs Real", 
    "Annual Simulation"
])

colors = ['#2980b9', '#e67e22', '#27ae60'] 
color_real = '#7f8c8d'
color_sim = '#2980b9'
plotly_template = "plotly_white"

# ==========================================
# Tab 1: Compare Profiles (Real Data)
# ==========================================
with tab1:
    st.markdown("### Daily Profile Comparison (Real Data)")
    default_dates = [available_dates[0], available_dates[min(10, len(available_dates)-1)]] if len(available_dates) > 1 else available_dates
    selected_dates = st.multiselect("Select dates to compare:", available_dates, default=default_dates)
    
    if selected_dates:
        cols = st.columns(len(selected_dates))
        plot_data = []
        max_parked, max_inout = 0, 0
        
        for i, target_date in enumerate(selected_dates):
            day_df = df[df['in_time'].dt.date == target_date]
            dow_str = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][target_date.weekday()]
            
            with cols[i]:
                st.markdown(f"<h4 style='color:{colors[i%len(colors)]}; text-align:center;'>{target_date} ({dow_str})</h4>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Arrivals", f"{len(day_df)}")
                m2.metric("Avg Stay", f"{day_df['stay_duration'].mean():.1f} h" if not day_df.empty else "0 h")
            
            d_p = calc_parked_cars(df, target_date)
            d_io = calc_in_out_hourly(df, target_date)
            plot_data.append({'date': target_date, 'dow': dow_str, 'parked': d_p, 'inout': d_io, 'raw': day_df})
            if not d_p.empty: max_parked = max(max_parked, d_p['parked_cars'].max())
            if not d_io.empty: max_inout = max(max_inout, d_io['in_count'].max(), d_io['out_count'].max())
        
        st.divider()
        unified_parked_y = [0, max_parked * 1.1] if max_parked > 0 else [0, 10]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Parked Vehicles**")
            fig_p = go.Figure()
            for i, data in enumerate(plot_data):
                fig_p.add_trace(go.Scatter(x=data['parked']['time_str'], y=data['parked']['parked_cars'], mode='lines', name=f"{data['date']}", line=dict(color=colors[i%len(colors)], width=2.5)))
            fig_p.update_layout(template=plotly_template, yaxis_range=unified_parked_y, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.markdown("**2. Hourly In/Out Dynamics**")
            fig_io = go.Figure()
            for i, data in enumerate(plot_data):
                fig_io.add_trace(go.Bar(x=data['inout']['time_str'], y=data['inout']['in_count'], name=f"IN: {data['date']}", marker_color=colors[i%len(colors)], offsetgroup=i))
                fig_io.add_trace(go.Bar(x=data['inout']['time_str'], y=-data['inout']['out_count'], name=f"OUT: {data['date']}", marker_color=colors[i%len(colors)], opacity=0.5, offsetgroup=i))
            fig_io.update_layout(template=plotly_template, barmode='group', margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig_io, use_container_width=True)

        st.markdown("**3. Stay Duration Distribution**")
        hist_cols = st.columns(len(plot_data))
        for i, data in enumerate(plot_data):
            with hist_cols[i]:
                if not data['raw'].empty:
                    fig_h = px.histogram(data['raw'], x='stay_duration', color_discrete_sequence=[colors[i%len(colors)]])
                    fig_h.update_layout(template=plotly_template, xaxis_title="Duration (h)", yaxis_title="Count", xaxis_range=[0, 24], margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig_h, use_container_width=True)

# ==========================================
# Tab 2: Trends
# ==========================================
with tab2:
    st.markdown("### Profile & Cyclicality Analysis (Real Data)")
    col1, col2 = st.columns(2)
    with col1:
        monthly = df.groupby('month').size().reset_index(name='total')
        days = df.groupby('month')['date'].nunique().reset_index(name='days')
        monthly = pd.merge(monthly, days, on='month')
        monthly['avg'] = monthly['total'] / monthly['days']
        fig_m = px.bar(monthly, x='month', y='avg', title="Average Arrivals by Month", color_discrete_sequence=['#34495e'])
        fig_m.update_layout(template=plotly_template, xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        weekly = df.groupby('weekday').size().reset_index(name='total')
        days_w = df.groupby('weekday')['date'].nunique().reset_index(name='days')
        weekly = pd.merge(weekly, days_w, on='weekday')
        weekly['avg'] = weekly['total'] / weekly['days']
        weekly['dow'] = weekly['weekday'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        fig_w = px.bar(weekly, x='dow', y='avg', title="Average Arrivals by Day of Week", color_discrete_sequence=['#7f8c8d'])
        fig_w.update_layout(template=plotly_template)
        st.plotly_chart(fig_w, use_container_width=True)

# ==========================================
# Tab 3: Daily Sim vs Real
# ==========================================
with tab3:
    st.markdown("### Daily Simulation vs. Real Data Comparison")
    st.write("Compare the actual logged data for a specific day against a freshly generated simulation based on the learned profile.")
    
    col_sd1, col_sd2, col_sd3 = st.columns([2, 2, 1])
    with col_sd1:
        compare_date = st.selectbox("Select Date to Compare:", available_dates)
    with col_sd2:
        sim_capacity_daily = st.number_input("Target Capacity for Simulation", min_value=10, max_value=5000, value=int(base_capacity), step=10, key="cap_daily")
    with col_sd3:
        st.write("") 
        if st.button("Regenerate Sim", key="regen_daily", use_container_width=True):
            st.session_state['daily_run_id'] += 1
            
    sim_daily_df = generate_daily_simulation(
        compare_date, sim_capacity_daily, base_capacity, profile_arrival, profile_stay, st.session_state['daily_run_id']
    )
    
    real_daily_df = df[df['in_time'].dt.date == compare_date]
    
    real_parked = calc_parked_cars(real_daily_df, compare_date)
    sim_parked = calc_parked_cars(sim_daily_df, compare_date)
    
    rm1, rm2, sm1, sm2 = st.columns(4)
    rm1.metric("Real: Total Arrivals", f"{len(real_daily_df)}")
    rm2.metric("Real: Avg Stay (h)", f"{real_daily_df['stay_duration'].mean():.1f}" if not real_daily_df.empty else "0.0")
    sm1.metric("Sim: Total Arrivals", f"{len(sim_daily_df)}")
    sm2.metric("Sim: Avg Stay (h)", f"{sim_daily_df['stay_duration'].mean():.1f}" if not sim_daily_df.empty else "0.0")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Parked Vehicles (Real vs. Sim)**")
        fig_comp_p = go.Figure()
        fig_comp_p.add_trace(go.Scatter(x=real_parked['time_str'], y=real_parked['parked_cars'], mode='lines', name='Real Data', line=dict(color=color_real, width=2, dash='dash')))
        fig_comp_p.add_trace(go.Scatter(x=sim_parked['time_str'], y=sim_parked['parked_cars'], mode='lines', name='Simulation', line=dict(color=color_sim, width=3)))
        fig_comp_p.update_layout(template=plotly_template, xaxis_title="Time", yaxis_title="Vehicles", margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig_comp_p, use_container_width=True)
        
    with c2:
        st.markdown("**Inbound Traffic (Real vs. Sim)**")
        real_io = calc_in_out_hourly(real_daily_df, compare_date)
        
        sim_daily_df['in_hour'] = sim_daily_df['in_time'].dt.hour
        sim_daily_df['out_hour'] = sim_daily_df['out_time'].dt.hour
        sim_io = calc_in_out_hourly(sim_daily_df, compare_date)
        
        fig_comp_io = go.Figure()
        fig_comp_io.add_trace(go.Bar(x=real_io['time_str'], y=real_io['in_count'], name='Real IN', marker_color=color_real, opacity=0.6))
        fig_comp_io.add_trace(go.Bar(x=sim_io['time_str'], y=sim_io['in_count'], name='Sim IN', marker_color=color_sim))
        fig_comp_io.update_layout(template=plotly_template, barmode='group', xaxis_title="Time", yaxis_title="Vehicles", margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig_comp_io, use_container_width=True)

# ==========================================
# Tab 4: Annual Simulation
# ==========================================
with tab4:
    st.markdown("### Scale-Out Annual Simulation Generator")
    st.write("Generate a full 1-year virtual log. Compare the generated annual trends with the baseline real data.")
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            target_year = st.selectbox("Target Year", [2026, 2027, 2028, 2029, 2030])
        with c2:
            st.metric("Base Capacity", f"Approx. {int(base_capacity)} cars")
        with c3:
            target_capacity = st.number_input("Target Capacity", min_value=10, max_value=5000, value=int(base_capacity)*2, step=10)
        with c4:
            st.write("") 
            if st.button("Generate", type="primary", key="regen_annual", use_container_width=True):
                st.session_state['annual_run_id'] += 1

    if st.session_state['annual_run_id'] > 0:
        with st.spinner("Processing 1-year simulation..."):
            sim_df = generate_annual_simulation(
                target_year, target_capacity, base_capacity, profile_arrival, profile_stay, st.session_state['annual_run_id']
            )
            
        st.success(f"Simulation Complete. Total records generated: {len(sim_df):,}")
        
        st.markdown("**Monthly Trend Comparison (Real vs. Simulation)**")
        
        real_monthly = df.groupby('month').size().reset_index(name='Real_Total')
        real_days = df.groupby('month')['date'].nunique().reset_index(name='days')
        real_monthly = pd.merge(real_monthly, real_days, on='month')
        real_monthly['Real (Daily Avg)'] = real_monthly['Real_Total'] / real_monthly['days']
        
        sim_monthly = sim_df.groupby('month').size().reset_index(name='Sim_Total')
        sim_days = sim_df.groupby('month')['date'].nunique().reset_index(name='days')
        sim_monthly = pd.merge(sim_monthly, sim_days, on='month')
        sim_monthly['Sim (Daily Avg)'] = sim_monthly['Sim_Total'] / sim_monthly['days']
        
        comp_monthly = pd.merge(real_monthly[['month', 'Real (Daily Avg)']], sim_monthly[['month', 'Sim (Daily Avg)']], on='month', how='outer').fillna(0)
        
        fig_ann_comp = go.Figure()
        fig_ann_comp.add_trace(go.Bar(x=comp_monthly['month'], y=comp_monthly['Real (Daily Avg)'], name='Real Baseline', marker_color=color_real))
        fig_ann_comp.add_trace(go.Bar(x=comp_monthly['month'], y=comp_monthly['Sim (Daily Avg)'], name='Simulation Output', marker_color=color_sim))
        fig_ann_comp.update_layout(template=plotly_template, barmode='group', xaxis=dict(tickmode='linear', title="Month"), yaxis_title="Avg Daily Arrivals")
        st.plotly_chart(fig_ann_comp, use_container_width=True)
        
        st.download_button(
            label="Download Annual Simulation Data (CSV)",
            data=sim_df[['in_time', 'out_time']].to_csv(index=False).encode('utf-8'),
            file_name=f"simulated_parking_{target_year}_{target_capacity}cars.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Click 'Generate' to create the annual simulation and view trend comparisons.")
EOF
