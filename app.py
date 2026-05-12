import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import scipy.stats as stats

# ==========================================
# 設定とUI
# ==========================================
st.set_page_config(layout="wide", page_title="VGI Data Generator")
st.title("VGIシミュレーション用 駐車場データ生成")

with st.sidebar:
    st.header("1. 基本設定")
    target_year = st.selectbox("対象年", [2026, 2027])
    total_visits = st.number_input("年間延べ利用回数 (Total Visits)", min_value=100, max_value=50000, value=10000, step=1000)
    total_ids = st.number_input("固有ID数 (Unique Users)", min_value=10, max_value=10000, value=1000, step=100)
    
    st.header("2. ユーザー分布 (パレートの法則)")
    # pareto_alpha が小さいほど一部のヘビーユーザーに偏る
    pareto_alpha = st.slider("偏り具合 (Alpha)", min_value=0.5, max_value=3.0, value=1.16, step=0.1, 
                             help="小さいほど一部の人が多く利用し、大きいほど全員が均等に利用します。")

    st.header("3. 時間帯ピークの調整")
    st.markdown("※山の中心時間を設定します。")
    in_peak_hour = st.slider("入庫ピーク (出勤)", min_value=5.0, max_value=12.0, value=9.0, step=0.5)
    stay_mean = st.slider("平均滞在時間 (時間)", min_value=1.0, max_value=15.0, value=8.5, step=0.5)

    if st.button("データ生成を実行", type="primary", use_container_width=True):
        st.session_state['run_sim'] = True
    else:
        if 'run_sim' not in st.session_state:
            st.session_state['run_sim'] = False

# ==========================================
# ロジック
# ==========================================
def generate_vgi_data(year, total_visits, total_ids, alpha, in_peak, stay_mean):
    np.random.seed(42) # 再現性のため
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    days_in_year = (end_date - start_date).days + 1

    # 1. IDごとの利用回数の割り当て (パレート分布)
    # パレート分布からサンプリングし、合計がtotal_visitsになるように正規化して丸める
    raw_weights = (np.random.pareto(alpha, total_ids) + 1)
    visits_per_id = np.round((raw_weights / raw_weights.sum()) * total_visits).astype(int)
    
    # 丸め誤差の調整（足りない分をランダムなIDに足す/引く）
    diff = total_visits - visits_per_id.sum()
    if diff > 0:
        add_idx = np.random.choice(total_ids, diff, replace=True)
        for idx in add_idx: visits_per_id[idx] += 1
    elif diff < 0:
        # 回数が1以上のIDから減らす
        for _ in range(abs(diff)):
            valid_idx = np.where(visits_per_id > 0)[0]
            if len(valid_idx) > 0:
                remove_idx = np.random.choice(valid_idx)
                visits_per_id[remove_idx] -= 1

    # IDのリスト作成 (例: 10001, 10002...)
    user_ids = np.arange(10001, 10001 + total_ids)

    # 2. 個別の訪問レコード(トランザクション)の生成
    records = []
    
    # 処理を軽くするため、事前に入庫時間と滞在時間の分布を作っておく
    # 入庫時間: 正規分布 (平均=in_peak, 標準偏差=1.5)
    in_hours_dist = np.random.normal(in_peak, 1.5, total_visits)
    # 滞在時間: 正規分布 (平均=stay_mean, 標準偏差=2.0)
    stay_hours_dist = np.random.normal(stay_mean, 2.0, total_visits)
    
    # 0〜23の範囲に収める
    in_hours_dist = np.clip(in_hours_dist, 0, 23)
    stay_hours_dist = np.clip(stay_hours_dist, 1, 24)

    visit_counter = 0
    
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 各IDについて処理
    for i, user_id in enumerate(user_ids):
        num_visits = visits_per_id[i]
        if num_visits <= 0: continue
            
        # このユーザーが訪問する日をランダムに選択（重複なし）
        # ※本来は平日/休日の重み付けが必要ですが、今回は単純化
        visit_days = np.random.choice(days_in_year, min(num_visits, days_in_year), replace=False)
        visit_days.sort()

        # 各訪問日の処理
        for d in visit_days:
            current_date = start_date + timedelta(days=int(d))
            
            # 分布から取り出し
            in_h = int(in_hours_dist[visit_counter])
            stay_h = int(stay_hours_dist[visit_counter])
            out_h = min(23, in_h + stay_h) # 日またぎは一旦考慮せず23時でカット
            
            visit_counter += 1

            # 1時間ごとのステータスを生成 (in, out, home)
            for h in range(24):
                dt_str = (current_date + timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00.000")
                
                if h >= in_h and h < out_h:
                    status = "in"
                elif h == out_h:
                    status = "out"
                else:
                    status = "home"
                    
                records.append({
                    "datetime": dt_str,
                    "car_id": user_id,
                    "in_out": status
                })
        
        # UI更新
        if i % 100 == 0:
            progress_bar.progress((i + 1) / total_ids)
            status_text.text(f"Generating data... {i+1}/{total_ids} IDs processed.")

    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(records)
    
    # 日時でソートして返す
    return df.sort_values(['datetime', 'car_id']).reset_index(drop=True)

# ==========================================
# 実行と結果表示
# ==========================================
if st.session_state['run_sim']:
    with st.spinner("データ生成中... (データ量によっては数分かかります)"):
        df_result = generate_vgi_data(target_year, total_visits, total_ids, pareto_alpha, in_peak_hour, stay_mean)
    
    st.success(f"生成完了: {len(df_result):,} レコード")
    
    # タブで表示を分ける
    tab_data, tab_dist = st.tabs(["データプレビュー", "分布確認"])
    
    with tab_data:
        # "home"ばかり表示されると分かりにくいので、特定のIDの動きを見せる
        sample_id = df_result['car_id'].iloc[0]
        st.write(f"サンプルデータ (car_id: {sample_id} の最初の100件)")
        st.dataframe(df_result[df_result['car_id'] == sample_id].head(100))
        
        # ダウンロードボタン
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="VGI用データをダウンロード (CSV)",
            data=csv,
            file_name=f"vgi_sim_data_{target_year}.csv",
            mime="text/csv",
        )

    with tab_dist:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ユーザーごとの年間利用回数 (上位50ID)**")
            # inが記録されている回数から、実際の訪問日数をざっくり計算
            visits_per_user = df_result[df_result['in_out'] == 'out'].groupby('car_id').size().reset_index(name='visits')
            visits_per_user = visits_per_user.sort_values('visits', ascending=False).head(50)
            fig1 = px.bar(visits_per_user, x=range(len(visits_per_user)), y='visits', labels={'x':'Rank', 'visits':'Annual Visits'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("**入庫(in)と出庫(out)の時間帯分布**")
            # 'out'の時間と、その日の最初の'in'の時間を集計
            in_df = df_result[df_result['in_out'] == 'in'].copy()
            in_df['date'] = pd.to_datetime(in_df['datetime']).dt.date
            first_in = in_df.groupby(['date', 'car_id']).first().reset_index()
            first_in['hour'] = pd.to_datetime(first_in['datetime']).dt.hour
            first_in['type'] = 'in'
            
            out_df = df_result[df_result['in_out'] == 'out'].copy()
            out_df['hour'] = pd.to_datetime(out_df['datetime']).dt.hour
            out_df['type'] = 'out'
            
            # 結合してプロット
            dist_df = pd.concat([first_in[['hour', 'type']], out_df[['hour', 'type']]])
            fig2 = px.histogram(dist_df, x='hour', color='type', barmode='group', nbins=24)
            st.plotly_chart(fig2, use_container_width=True)
