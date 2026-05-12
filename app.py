import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==========================================
# 1. Simulation Engine (Bottom-Up Model)
# ==========================================

def run_simulation(year, num_ids, target_daily_avg, fixed_ratio, 
                   arr_peak_fixed, stay_mean_fixed, 
                   arr_peak_irreg, stay_mean_irreg,
                   monthly_weights, weekly_weights, progress_bar, status_text):
    np.random.seed(int(time.time()))
    
    # 1. IDの分割 (固定層 vs イレギュラー層)
    num_fixed = int(num_ids * (fixed_ratio / 100.0))
    num_irreg = num_ids - num_fixed
    car_ids_fixed = np.arange(1, 1 + num_fixed)
    car_ids_irreg = np.arange(1 + num_fixed, 1 + num_ids)

    # 2. 基本来訪確率の計算
    # 固定層は「平均して週4日」くらい来るベース確率
    prob_fixed_base = np.random.normal(4.0/7.0, 0.5/7.0, num_fixed)
    prob_fixed_base = np.clip(prob_fixed_base, 0.2, 0.95)
    
    # イレギュラー層は「平均して月1〜2回」くらい来るベース確率
    prob_irreg_base = np.random.exponential(scale=0.05, size=num_irreg)
    prob_irreg_base = np.clip(prob_irreg_base, 0.001, 0.1)

    start_date = datetime(year, 1, 1)
    days_in_year = (datetime(year, 12, 31) - start_date).days + 1
    event_data = []

    for d in range(days_in_year):
        if d % 20 == 0:
            progress_bar.progress(int((d / days_in_year) * 50))
            status_text.text(f"シミュレーション実行中... {d}/{days_in_year}日")
            
        cur_date = start_date + timedelta(days=d)
        m_idx = cur_date.month - 1
        w_idx = cur_date.weekday()
        
        # 月と曜日のトレンドを掛け合わせる（固定層はトレンドに強く影響され、イレギュラー層は影響が少ない）
        trend_factor = monthly_weights[m_idx] * weekly_weights[w_idx]
        
        # その日の確率を計算
        day_prob_fixed = np.clip(prob_fixed_base * trend_factor, 0, 1)
        day_prob_irreg = np.clip(prob_irreg_base * (trend_factor * 0.5 + 0.5), 0, 1) # イレギュラーは休日の影響を和らげる
        
        # 来訪ガチャを引く
        active_fixed = car_ids_fixed[np.random.rand(num_fixed) < day_prob_fixed]
        active_irreg = car_ids_irreg[np.random.rand(num_irreg) < day_prob_irreg]
        
        base_dt = datetime.combine(cur_date, datetime.min.time())
        
        # 固定層のイベント生成（朝型・長時間滞在）
        for cid in active_fixed:
            in_h = np.clip(np.random.normal(arr_peak_fixed, 1.0), 0, 22)
            duration = np.clip(np.random.normal(stay_mean_fixed, 1.5), 1.0, 15.0)
            in_t = base_dt + timedelta(hours=int(in_h))
            out_t = in_t + timedelta(hours=int(duration))
            event_data.append({"car_id": cid, "layer": "Fixed", "in_time": in_t, "out_time": out_t, "stay_duration": duration})
            
        # イレギュラー層のイベント生成（バラバラな時間・短時間滞在）
        for cid in active_irreg:
            in_h = np.clip(np.random.normal(arr_peak_irreg, 3.0), 0, 22) # バラつきを大きく
            duration = np.clip(np.random.normal(stay_mean_irreg, 0.5), 0.5, 4.0)
            in_t = base_dt + timedelta(hours=int(in_h))
            out_t = in_t + timedelta(hours=int(duration))
            event_data.append({"car_id": cid, "layer": "Irregular", "in_time": in_t, "out_time": out_t, "stay_duration": duration})
            
    progress_bar.progress(60)
    status_text.text("データ構築中...")
    
    df_res = pd.DataFrame(event_data)
    if not df_res.empty:
        df_res['date'] = df_res['in_time'].dt.date
        df_res['month'] = df_res['in_time'].dt.month
        df_res['weekday'] = df_res['in_time'].dt.weekday
        df_res['in_hour'] = df_res['in_time'].dt.hour
        df_res['out_hour'] = df_res['out_time'].dt.hour
        
    return df_res, np.concatenate([car_ids_fixed, car_ids_irreg])

def generate_vgi_log(df_events, year, all_car_ids, progress_bar, status_text):
    times = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='h')
    status_matrix = np.full((len(times), len(all_car_ids)), 'home', dtype=object)
    id_map = {cid: i for i, cid in enumerate(all_car_ids)}
    start_t = times[0]
    
    total = len(df_events)
    chunk = max(1, total // 10)
    for i, (_, row) in enumerate(df_events.iterrows()):
        if i % chunk == 0:
            progress_bar.progress(60 + int((i / total) * 35))
            status_text.text(f"VGIログ展開中... {i}/{total}")
            
        cid_idx = id_map[row['car_id']]
        idx_in = int((row['in_time'].floor('h') - start_t).total_seconds() // 3600)
        idx_out = int((row['out_time'].floor('h') - start_t).total_seconds() // 3600)
        
        if 0 <= idx_in < len(times):
            status_matrix[idx_in:min(idx_out, len(times)), cid_idx] = 'in'
        if 0 <= idx_out < len(times):
            status_matrix[idx_out, cid_idx] = 'out'
            
    progress_bar.progress(100)
    status_text.text("完了しました！")
    return pd.DataFrame({
        'datetime': np.tile(times, len(all_car_ids)),
        'car_id': np.repeat(all_car_ids, len(times)),
        'in_out': status_matrix.flatten('F')
    })

# ==========================================
# 2. Main UI
# ==========================================

st.set_page_config(layout="wide", page_title="VGI Simulator (Layered Model)")
st.title("🚗 VGIシミュレータ：固定層・イレギュラー層モデル")

with st.sidebar:
    st.header("1. 基本設定")
    t_yr = st.selectbox("対象年", [2025, 2026, 2027])
    num_ids = st.number_input("登録ID数", 100, 10000, 1500)
    target_avg = st.number_input("目標の平均来訪台数/日", 10, 5000, 700)
    
    st.divider()
    st.header("2. ユーザー層の定義")
    fixed_ratio = st.slider("固定層ユーザーの割合 (%)", 10, 90, 80, help="月〜金に安定して来る層。残りは突発的なイレギュラー層になります。")
    
    with st.expander("詳細な行動パターン設定", expanded=False):
        st.markdown("**固定層（長居する人たち）**")
        arr_fixed = st.slider("平均到着時刻", 6.0, 12.0, 9.0)
        stay_fixed = st.slider("平均滞在時間 (h)", 4.0, 12.0, 8.0)
        st.markdown("**イレギュラー層（業者や短時間利用者）**")
        arr_irreg = st.slider("平均到着時刻 (イレギュラー)", 8.0, 18.0, 13.0)
        stay_irreg = st.slider("平均滞在時間 (h) (イレギュラー)", 0.5, 5.0, 2.0)

    st.divider()
    st.header("3. トレンド設定（波の形）")
    with st.expander("曜日トレンド (月〜日)", expanded=False):
        w_weights = []
        days = ['月', '火', '水', '木', '金', '土', '日']
        default_w = [1.2, 1.1, 1.0, 1.0, 0.9, 0.2, 0.1] # 平日メインのデフォルト
        for i, d in enumerate(days):
            w_weights.append(st.slider(f"{d}曜日の重み", 0.0, 2.0, default_w[i], 0.1))

    with st.expander("月別（学校スケジュール）トレンド", expanded=False):
        m_weights = []
        # 大学のデフォルト（2,3,8,9月は休みで人が減る想定）
        default_m = [1.0, 0.5, 0.5, 1.2, 1.1, 1.1, 1.1, 0.4, 0.5, 1.2, 1.1, 1.0]
        for i in range(12):
            m_weights.append(st.slider(f"{i+1}月の重み", 0.0, 2.0, default_m[i], 0.1))

    if st.button("シミュレーションを実行", type="primary", use_container_width=True):
        st.session_state['trigger_sim'] = True

# Simulation Execution
if st.session_state.get('trigger_sim', False):
    pb = st.progress(0); st_txt = st.empty()
    df_sim, ids = run_simulation(
        t_yr, num_ids, target_avg, fixed_ratio, 
        arr_fixed, stay_fixed, arr_irreg, stay_irreg,
        m_weights, w_weights, pb, st_txt
    )
    st.session_state['sim_df'] = df_sim
    st.session_state['ids'] = ids
    st.session_state['trigger_sim'] = False
    pb.empty(); st_txt.empty(); st.rerun()

# Dashboard View
if 'sim_df' in st.session_state:
    df_sim = st.session_state['sim_df']
    
    # 実際の平均台数を計算して補正する情報表示
    real_avg = len(df_sim) / 365
    st.info(f"💡 シミュレーション結果: 年間平均 {real_avg:.1f} 台/日 (目標 {target_avg} 台に対して、設定した月・曜日の波が適用されています)")

    tab1, tab2, tab3 = st.tabs(["分布とトレンド", "日別詳細", "VGIログ(CSV)"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 滞在時間の分布（層による違い）")
            f_stay = px.histogram(df_sim, x="stay_duration", color="layer", nbins=40, 
                                  color_discrete_map={"Fixed": "#2980b9", "Irregular": "#e67e22"},
                                  opacity=0.75, barmode="overlay")
            f_stay.update_layout(xaxis_title="滞在時間 (h)", yaxis_title="件数", template="plotly_white")
            st.plotly_chart(f_stay, use_container_width=True)
        with c2:
            st.markdown("### 曜日別の来訪層の内訳")
            w_layer = df_sim.groupby(['weekday', 'layer']).size().reset_index(name='count')
            w_layer['count'] = w_layer['count'] / 52 # 年間52週で割って1日平均へ
            w_layer['dw'] = w_layer['weekday'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
            f_w = px.bar(w_layer, x='dw', y='count', color='layer', 
                         color_discrete_map={"Fixed": "#2980b9", "Irregular": "#e67e22"})
            f_w.update_layout(yaxis_title="平均来訪台数/日", template="plotly_white")
            st.plotly_chart(f_w, use_container_width=True)
            
        st.markdown("### 月別トレンド（学校スケジュール）")
        m_layer = df_sim.groupby(['month', 'layer']).size().reset_index(name='count')
        m_layer['count'] = m_layer['count'] / 30 # 簡易的に1日平均へ
        f_m = px.bar(m_layer, x='month', y='count', color='layer', 
                     color_discrete_map={"Fixed": "#2980b9", "Irregular": "#e67e22"})
        f_m.update_layout(xaxis=dict(tickmode='linear'), yaxis_title="平均来訪台数/日", template="plotly_white")
        st.plotly_chart(f_m, use_container_width=True)

    with tab2:
        st.markdown("### 指定日の1時間ごとの入出庫")
        sel_date = st.selectbox("日付を選択:", sorted(df_sim['date'].unique()))
        day_df = df_sim[df_sim['date'] == sel_date]
        
        in_h = day_df.groupby(['in_hour', 'layer']).size().reset_index(name='in_count')
        out_h = day_df.groupby(['out_hour', 'layer']).size().reset_index(name='out_count')
        
        f_io = go.Figure()
        for layer, col in [("Fixed", "#2980b9"), ("Irregular", "#e67e22")]:
            in_d = in_h[in_h['layer'] == layer]
            out_d = out_h[out_h['layer'] == layer]
            f_io.add_trace(go.Bar(x=in_d['in_hour'], y=in_d['in_count'], name=f"IN ({layer})", marker_color=col))
            f_io.add_trace(go.Bar(x=out_d['out_hour'], y=-out_d['out_count'], name=f"OUT ({layer})", marker_color=col, opacity=0.5))
            
        f_io.update_layout(barmode='stack', template="plotly_white", xaxis_title="時刻", yaxis_title="台数")
        st.plotly_chart(f_io, use_container_width=True)

    with tab3:
        st.markdown("### VGI用 状態遷移ログ出力")
        if st.button("CSV形式に展開する", type="primary"):
            pb = st.progress(0); st_txt = st.empty()
            df_vgi = generate_vgi_log(df_sim, t_yr, st.session_state['ids'], pb, st_txt)
            st.success("展開完了！")
            st.dataframe(df_vgi.head(1000), use_container_width=True)
            st.download_button("CSVをダウンロード", df_vgi.to_csv(index=False).encode('utf-8'), f"vgi_log_{t_yr}.csv")
            pb.empty(); st_txt.empty()
