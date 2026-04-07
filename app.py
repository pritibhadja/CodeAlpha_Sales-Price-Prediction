from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
data = pd.read_csv("advertising.csv", sep='\t', index_col=0)
data.columns = data.columns.str.strip()

def get_analytics_data():
    tv_total    = float(data['TV'].sum())
    radio_total = float(data['Radio'].sum())
    news_total  = float(data['Newspaper'].sum())
    total_spend = tv_total + radio_total + news_total

    tv_corr    = float(data['TV'].corr(data['Sales']))
    radio_corr = float(data['Radio'].corr(data['Sales']))
    news_corr  = float(data['Newspaper'].corr(data['Sales']))

    corr = data[['TV', 'Radio', 'Newspaper', 'Sales']].corr().round(3)
    corr_dict = corr.to_dict()

    best_channel = max(
        [('TV', tv_corr), ('Radio', radio_corr), ('Newspaper', news_corr)],
        key=lambda x: x[1]
    )

    return {
        'tv': data['TV'].tolist(),
        'radio': data['Radio'].tolist(),
        'news': data['Newspaper'].tolist(),
        'sales': data['Sales'].tolist(),
        'tv_total': round(tv_total, 1),
        'radio_total': round(radio_total, 1),
        'news_total': round(news_total, 1),
        'total_spend': round(total_spend, 1),
        'tv_pct': round(tv_total / total_spend * 100, 1),
        'radio_pct': round(radio_total / total_spend * 100, 1),
        'news_pct': round(news_total / total_spend * 100, 1),
        'tv_corr': round(tv_corr, 3),
        'radio_corr': round(radio_corr, 3),
        'news_corr': round(news_corr, 3),
        'best_channel': best_channel[0],
        'best_corr': round(best_channel[1], 3),
        'corr_matrix': corr_dict,
        'avg_sales': round(float(data['Sales'].mean()), 2),
        'max_sales': round(float(data['Sales'].max()), 2),
        'min_sales': round(float(data['Sales'].min()), 2),
        'sales_trend': data['Sales'].tolist(),
    }

@app.route('/')
def home():
    stats = {
        'avg_sales': round(float(data['Sales'].mean()), 2),
        'max_sales': round(float(data['Sales'].max()), 2),
        'records': len(data),
    }
    return render_template('index.html', stats=stats)

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/analytics')
def analytics():
    d = get_analytics_data()
    return render_template('analytics.html', analytics_json=json.dumps(d),
                           best_channel=d['best_channel'],
                           best_corr=d['best_corr'],
                           avg_sales=d['avg_sales'],
                           max_sales=d['max_sales'],
                           tv_corr=d['tv_corr'],
                           radio_corr=d['radio_corr'],
                           news_corr=d['news_corr'],
                           tv_pct=d['tv_pct'],
                           radio_pct=d['radio_pct'],
                           news_pct=d['news_pct'])

@app.route('/predict', methods=['POST'])
def predict():
    tv    = float(request.form['TV'])
    radio = float(request.form['Radio'])
    news  = float(request.form['Newspaper'])
    prediction = model.predict([[tv, radio, news]])[0]
    result = round(prediction, 2)

    d = get_analytics_data()
    suggestions = []
    if tv < 150 and d['tv_corr'] > 0.7:
        suggestions.append(f"📺 Boost TV to ~150–200 units — TV has {d['tv_corr']} correlation with sales (strongest driver).")
    if radio < 25 and d['radio_corr'] > 0.3:
        suggestions.append(f"📻 Increase Radio to ~25–35 units — Radio shows {d['radio_corr']} correlation with sales.")
    if news > 50:
        suggestions.append(f"📰 Consider cutting Newspaper spend — weakest impact ({d['news_corr']} correlation).")
    if not suggestions:
        suggestions.append("✅ Excellent budget mix! Your allocation is well-optimized for maximum sales.")

    return render_template('predict.html', result=result,
                           tv=tv, radio=radio, news=news,
                           suggestions=suggestions,
                           tv_corr=d['tv_corr'],
                           radio_corr=d['radio_corr'],
                           news_corr=d['news_corr'])

@app.route('/api/input_analytics', methods=['POST'])
def input_analytics():
    """
    Compute all 4 dynamic panels based on the user's exact input values.
    Returns: pie shares, per-channel correlations WITH predicted sales
    contribution, heatmap matrix, and optimizer result — all input-specific.
    """
    body  = request.get_json()
    tv    = float(body.get('tv',    0))
    radio = float(body.get('radio', 0))
    news  = float(body.get('news',  0))

    total = tv + radio + news
    if total == 0:
        return jsonify({'error': 'All values are zero'}), 400

    # ── 1. Pie: actual share of THIS input ──────────────────────────────
    pie = {
        'tv':    round(tv    / total * 100, 1),
        'radio': round(radio / total * 100, 1),
        'news':  round(news  / total * 100, 1),
    }

    # ── 2. Per-channel contribution to predicted sales ──────────────────
    # Use model coefficients (linear model) to show each channel's marginal contribution
    coef = model.coef_   # [tv_coef, radio_coef, news_coef]
    intercept = model.intercept_

    tv_contrib    = round(float(coef[0] * tv),    2)
    radio_contrib = round(float(coef[1] * radio), 2)
    news_contrib  = round(float(coef[2] * news),  2)
    base          = round(float(intercept), 2)
    predicted     = round(float(model.predict([[tv, radio, news]])[0]), 2)

    contrib_total = abs(tv_contrib) + abs(radio_contrib) + abs(news_contrib)
    if contrib_total == 0: contrib_total = 1

    corr_input = {
        'tv':    round(abs(tv_contrib)    / contrib_total, 3),
        'radio': round(abs(radio_contrib) / contrib_total, 3),
        'news':  round(abs(news_contrib)  / contrib_total, 3),
        'tv_contrib':    tv_contrib,
        'radio_contrib': radio_contrib,
        'news_contrib':  news_contrib,
        'base':          base,
        'predicted':     predicted,
        'tv_coef':    round(float(coef[0]), 4),
        'radio_coef': round(float(coef[1]), 4),
        'news_coef':  round(float(coef[2]), 4),
    }

    # ── 3. Heatmap: 4×4 correlation matrix for a "local neighbourhood" ──
    # Build a small synthetic dataset around the user's input (±20% variance)
    # so the heatmap reflects correlations near their operating point
    np.random.seed(42)
    n = 80
    tv_s    = np.clip(np.random.normal(tv,    tv    * 0.2 + 1, n), 0, None)
    radio_s = np.clip(np.random.normal(radio, radio * 0.2 + 1, n), 0, None)
    news_s  = np.clip(np.random.normal(news,  news  * 0.2 + 1, n), 0, None)
    sales_s = model.predict(np.column_stack([tv_s, radio_s, news_s]))

    local_df = pd.DataFrame({'TV': tv_s, 'Radio': radio_s, 'Newspaper': news_s, 'Sales': sales_s})
    heatmap  = local_df.corr().round(3).to_dict()

    # ── 4. Optimizer: best split for this exact total budget ─────────────
    budget   = total
    step     = max(budget / 25, 0.5)
    best_pred, b_tv, b_radio, b_news = 0, tv, radio, news
    for t in np.arange(0, budget + step, step):
        for r in np.arange(0, budget - t + step, step):
            nw = budget - t - r
            if nw < 0: continue
            p = float(model.predict([[t, r, nw]])[0])
            if p > best_pred:
                best_pred, b_tv, b_radio, b_news = p, t, r, nw

    optimizer = {
        'tv':              round(float(b_tv),    1),
        'radio':           round(float(b_radio), 1),
        'newspaper':       round(float(b_news),  1),
        'predicted_sales': round(float(best_pred), 2),
        'current_sales':   predicted,
        'gain':            round(float(best_pred) - predicted, 2),
    }

    return jsonify({
        'pie':        pie,
        'corr_input': corr_input,
        'heatmap':    heatmap,
        'optimizer':  optimizer,
        'tv': tv, 'radio': radio, 'news': news, 'total': round(total, 1),
    })


@app.route('/predict_live', methods=['POST'])
def predict_live():
    """JSON endpoint for live prediction as user types."""
    body = request.get_json()
    tv    = float(body.get('tv', 0))
    radio = float(body.get('radio', 0))
    news  = float(body.get('news', 0))
    prediction = model.predict([[tv, radio, news]])[0]
    return jsonify({'prediction': round(float(prediction), 2)})


@app.route('/api/optimize', methods=['POST'])
def optimize():
    body = request.get_json()
    budget = float(body.get('budget', 300))
    best_pred, best_tv, best_radio, best_news = 0, 0, 0, 0
    step = max(budget / 20, 1)
    for tv in np.arange(0, budget + 1, step):
        for radio in np.arange(0, budget - tv + 1, step):
            news = budget - tv - radio
            if news < 0:
                continue
            pred = model.predict([[tv, radio, news]])[0]
            if pred > best_pred:
                best_pred, best_tv, best_radio, best_news = pred, tv, radio, news
    return jsonify({
        'predicted_sales': round(float(best_pred), 2),
        'tv': round(float(best_tv), 1),
        'radio': round(float(best_radio), 1),
        'newspaper': round(float(best_news), 1),
    })

@app.route('/download_report')
def download_report():
    df = data.copy()
    df['Predicted_Sales'] = model.predict(df[['TV', 'Radio', 'Newspaper']]).round(2)
    df['Error'] = (df['Sales'] - df['Predicted_Sales']).round(2)
    buf = io.StringIO()
    df.to_csv(buf)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'sales_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == "__main__":
    app.run(debug=True)
