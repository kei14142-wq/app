import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 設定とUI
# ==========================================
st.set_page_config(layout="wide", page_title="VGI Data Generator")
st.title("VGIシミュレーション用 駐車場データ生成")

with st.sidebar:
    st.header("1. 基本設定")
    target_year = st.selectbox("対象年", [2026, 2027], index=0)
    # 「延べ」ではなく「何台の車をシミュレートするか」を主役に
    total_ids = st.number_input("固有ID数 (登録車両数)", min_value=10, max_value=10000, value=1000, step=100)
    # 頻度を調整するつまみ
    avg_visits_per_year = st.slider("1台あたりの年間平均来場回数", min_value=1, max_value=200, value=20)
    
    st.header("2. ユーザー分布 (来場頻度の偏り)")
    # Alphaが小さいほど「毎日来る人」と「たまにしか来ない人」の差が激しくなります
    pareto_alpha = st.slider("偏り具合 (Alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

    st.header("3. 時間帯ピークの調整")
    in_peak_hour = st.slider("入庫ピーク (出勤)", min_value=5.0, max_value=12.0, value=8.5, step=0.5)
    stay_mean = st.slider("平均滞在時間 (時間)", min_value=1.0, max_value=15.0, value=9.0, step=0.5)

    if st.button("データ生成を実行", type="primary", use_container_width=True):
        st.session_state['run_sim'] = True

# ==========================================
# 高速化ロジック
# ==========================================
def generate_vgi_data_fast(year, total_ids, avg_visits, alpha, in_peak, stay_mean):
    np.random.seed(42)
    start_date = datetime(year, 1, 1)
    days_in_year = 365

    # 1. 各IDに年間何回来るかを割り振る (パレート分布)
    total_target_visits = total_ids * avg_visits
    raw_weights = (np.random.pareto(alpha, total_ids) + 1)
    visits_per_id = np.round((raw_weights / raw_weights.sum()) * total_target_visits).astype(int)
    # 最大365回に制限
    visits_per_id = np.clip(visits_per_id, 0, 365)

    user_ids = np.arange(10001, 10001 + total_ids)
    records = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 2. 各IDのレコードを生成
    for i, user_id in enumerate(user_ids):
        num_visits = visits_per_id[i]
        if num_visits == 0: continue
            
        # 訪問する日をランダムに決定
        visit_days = np.random.choice(days_in_year, num_visits, replace=False)
        
        for d in visit_days:
            current_date = start_date + timedelta(days=int(d))
            
            # 入庫・滞在時間に少しのランダム性を加える
            in_h = int(np.clip(np.random.normal(in_peak, 1.2), 0, 23))
            s_h = int(np.clip(np.random.normal(stay_mean, 2.0), 1, 24))
            out_h = min(23, in_h + s_h)

            # 車が来た日の24時間分のログを生成
            for h in range(24):
                status = "home"
                if h >= in_h and h < out_h:
                    status = "in"
                elif h == out_h:
                    status = "out"
                
                records.append({
                    "datetime": (current_date + timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00.000"),
                    "car_id": user_id,
                    "in_out": status
                })
        
        # 進捗表示 (負荷軽減のため500件おき)
        if i % 500 == 0:
            progress_bar.progress((i + 1) / total_ids)
            status_text.text(f"処理中: {i+1}/{total_ids} 台目...")

    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(records)

# ==========================================
# 実行と結果表示
# ==========================================
if st.session_state.get('run_sim'):
    with st.spinner("膨大なデータを構築中です..."):
        df_result = generate_vgi_data_fast(target_year, total_ids, avg_visits_per_year, pareto_alpha, in_peak_hour, stay_mean)
    
    st.success(f"生成完了！ 総レコード数: {len(df_result):,} 件")
    
    tab1, tab2 = st.tabs(["データ確認", "統計・分布"])
    
    with tab1:
        st.write("### 生成データ（最初の100件）")
        st.dataframe(df_result.head(100), use_container_width=True)
        
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="CSVをダウンロード",
            data=csv,
            file_name=f"vgi_sim_{target_year}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### ユーザーごとの年間来場頻度")
            freq_df = df_result[df_result['in_out'] == 'out'].groupby('car_id').size().reset_index(name='count')
            fig1 = px.histogram(freq_df, x='count', nbins=30, labels={'count':'年間来場回数', 'y':'人数'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.write("### 時間帯別の入庫(in)・出庫(out)分布")
            # inとoutのみ抽出して時間帯を集計
            move_df = df_result[df_result['in_out'].isin(['in', 'out'])].copy()
            move_df['hour'] = pd.to_datetime(move_df['datetime']).dt.hour
            fig2 = px.histogram(move_df, x='hour', color='in_out', barmode='group', nbins=24)
            st.plotly_chart(fig2, use_container_width=True)
