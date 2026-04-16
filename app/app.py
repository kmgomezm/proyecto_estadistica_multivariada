import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ames Housing · Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary: #1a1a2e;
    --accent: #e8b86d;
    --accent2: #c44536;
    --surface: #16213e;
    --card: #0f3460;
    --text: #f0f0f0;
    --muted: #8892a4;
    --success: #4caf93;
    --border: rgba(232,184,109,0.2);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: var(--primary); color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label { color: var(--muted) !important; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '🏠';
    position: absolute;
    right: 36px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
}
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.02em;
}
.app-subtitle { color: var(--muted); margin: 4px 0 0 0; font-size: 0.95rem; }

/* Metric cards */
.metric-row { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    flex: 1;
    min-width: 160px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
    border-radius: 4px 0 0 4px;
}
.metric-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: var(--accent); }
.metric-sub { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }

/* Section title */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 28px 0 18px 0;
}

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, var(--card), var(--surface));
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    margin: 24px 0;
}
.prediction-label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.12em; }
.prediction-value {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 900;
    color: var(--accent);
    margin: 8px 0;
    text-shadow: 0 0 40px rgba(232,184,109,0.4);
}
.prediction-range { color: var(--muted); font-size: 0.88rem; }

/* Model table */
.model-table { border-collapse: collapse; width: 100%; }
.model-table th {
    background: var(--card);
    color: var(--accent);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 2px solid var(--border);
}
.model-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-size: 0.9rem;
}
.model-table tr:hover td { background: rgba(232,184,109,0.05); }
.best-row td { background: rgba(232,184,109,0.08) !important; }
.rank-badge {
    background: var(--accent);
    color: var(--primary);
    border-radius: 50%;
    width: 22px; height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--muted); border-radius: 8px; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--accent) !important; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Info box */
.info-box {
    background: rgba(76,175,147,0.1);
    border: 1px solid rgba(76,175,147,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: var(--success);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Generate a realistic Ames Housing dataset sample for demo."""
    np.random.seed(42)
    n = 1460

    neighborhoods = ['CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst',
                     'NWAmes','OldTown','BrkSide','Sawyer','NridgHt','NAmes',
                     'SawyerW','IDOTRR','MeadowV','Edwards','Timber','Gilbert']
    
    data = {
        'MSSubClass': np.random.choice([20,30,40,45,50,60,70,75,80,85,90,120,160,180,190], n),
        'LotFrontage': np.random.normal(70, 24, n).clip(20, 200),
        'LotArea': np.random.lognormal(9.2, 0.5, n).astype(int),
        'OverallQual': np.random.choice(range(1,11), n, p=[0.01,0.02,0.03,0.06,0.14,0.24,0.22,0.17,0.08,0.03]),
        'OverallCond': np.random.choice(range(1,11), n, p=[0.005,0.01,0.01,0.05,0.10,0.55,0.12,0.10,0.05,0.005]),
        'YearBuilt': np.random.randint(1872, 2011, n),
        'YearRemodAdd': np.random.randint(1950, 2011, n),
        'MasVnrArea': np.random.exponential(100, n).clip(0, 1600).astype(int),
        '1stFlrSF': np.random.normal(1163, 386, n).clip(334, 4692).astype(int),
        '2ndFlrSF': np.random.choice([0]*800 + list(np.random.normal(750, 200, 660).astype(int)), n),
        'GrLivArea': np.random.normal(1515, 525, n).clip(334, 5642).astype(int),
        'BsmtFullBath': np.random.choice([0,1,2,3], n, p=[0.43,0.44,0.12,0.01]),
        'FullBath': np.random.choice([0,1,2,3], n, p=[0.01,0.37,0.54,0.08]),
        'HalfBath': np.random.choice([0,1,2], n, p=[0.57,0.39,0.04]),
        'BedroomAbvGr': np.random.choice([0,1,2,3,4,5,6], n, p=[0.003,0.02,0.19,0.52,0.22,0.05,0.017]),
        'TotRmsAbvGrd': np.random.choice(range(2,15), n),
        'Fireplaces': np.random.choice([0,1,2,3], n, p=[0.47,0.40,0.12,0.01]),
        'GarageCars': np.random.choice([0,1,2,3,4], n, p=[0.05,0.22,0.57,0.14,0.02]),
        'GarageArea': np.random.normal(472, 215, n).clip(0, 1418).astype(int),
        'WoodDeckSF': np.random.exponential(60, n).clip(0, 857).astype(int),
        'OpenPorchSF': np.random.exponential(47, n).clip(0, 547).astype(int),
        'Neighborhood': np.random.choice(neighborhoods, n),
        'BldgType': np.random.choice(['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'], n, p=[0.83,0.03,0.04,0.07,0.03]),
        'HouseStyle': np.random.choice(['1Story','2Story','1.5Fin','SLvl','SFoyer'], n, p=[0.50,0.30,0.10,0.06,0.04]),
        'CentralAir': np.random.choice(['Y','N'], n, p=[0.93,0.07]),
        'TotalBsmtSF': np.random.normal(1057, 439, n).clip(0, 6110).astype(int),
        'MoSold': np.random.randint(1, 13, n),
        'YrSold': np.random.choice([2006,2007,2008,2009,2010], n),
    }
    df = pd.DataFrame(data)
    # Generate realistic SalePrice
    df['SalePrice'] = (
        50000
        + df['OverallQual'] * 18000
        + df['GrLivArea'] * 65
        + df['GarageArea'] * 35
        + df['TotalBsmtSF'] * 30
        + df['YearBuilt'] * 200
        + df['FullBath'] * 8000
        + df['Fireplaces'] * 6000
        + np.random.normal(0, 25000, n)
    ).clip(34900, 755000).astype(int)
    return df


@st.cache_resource
def load_models(df):
    """Train all models on the dataset."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except: has_xgb = False
    try:
        from lightgbm import LGBMRegressor
        has_lgbm = True
    except: has_lgbm = False
    try:
        from catboost import CatBoostRegressor
        has_cat = True
    except: has_cat = False

    # Encode categoricals
    df_enc = df.copy()
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.drop('SalePrice', axis=1)
    y = df_enc['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_defs = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=100),
        'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'MLP': MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, random_state=42),
    }
    if has_xgb:
        model_defs['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    if has_lgbm:
        model_defs['LightGBM'] = LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, verbose=-1)
    if has_cat:
        model_defs['CatBoost'] = CatBoostRegressor(iterations=200, depth=6, random_state=42, verbose=0)

    results = {}
    trained = {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        ytr_pred = model.predict(X_train)
        yte_pred = model.predict(X_test)
        results[name] = {
            'train_r2': r2_score(y_train, ytr_pred),
            'test_r2': r2_score(y_test, yte_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, ytr_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, yte_pred)),
            'train_mae': mean_absolute_error(y_train, ytr_pred),
            'test_mae': mean_absolute_error(y_test, yte_pred),
        }
        trained[name] = model

    return trained, results, X_train, X_test, y_train, y_test, list(X.columns), df_enc


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">Ames Housing Predictor</p>
    <p class="app-subtitle">Estadística Multivariada · Modelos de Regresión · Precio de Viviendas</p>
</div>
""", unsafe_allow_html=True)

# Load data & models
with st.spinner("Entrenando modelos..."):
    df = load_data()
    trained_models, results, X_train, X_test, y_train, y_test, feature_cols, df_enc = load_models(df)

best_model_name = max(results, key=lambda k: results[k]['test_r2'])

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predicción", "📊  Resultados de Modelos", "🔍  EDA"])


# ══════════════════════════════════════════════
# TAB 1: PREDICCIÓN
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Predicción de Precio de Vivienda</p>', unsafe_allow_html=True)

    col_model, col_info = st.columns([2, 1])
    with col_model:
        selected_model = st.selectbox(
            "Selecciona el modelo",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name)
        )
    with col_info:
        m = results[selected_model]
        st.markdown(f"""
        <div class="info-box">
            R² Test: <strong>{m['test_r2']:.3f}</strong> &nbsp;|&nbsp;
            RMSE Test: <strong>${m['test_rmse']:,.0f}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Características de la Propiedad")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📐 Tamaño & Estructura**")
        gr_liv = st.number_input("Área habitable (pie²)", 500, 5000, 1500, 50)
        total_bsmt = st.number_input("Área sótano total (pie²)", 0, 3000, 900, 50)
        first_flr = st.number_input("Área 1er piso (pie²)", 300, 4000, 1000, 50)
        second_flr = st.number_input("Área 2do piso (pie²)", 0, 2000, 0, 50)
        lot_area = st.number_input("Tamaño del lote (pie²)", 1000, 50000, 9000, 500)
        garage_area = st.number_input("Área garaje (pie²)", 0, 1500, 450, 25)

    with c2:
        st.markdown("**⭐ Calidad & Condición**")
        overall_qual = st.slider("Calidad general (1-10)", 1, 10, 6)
        overall_cond = st.slider("Condición general (1-10)", 1, 10, 5)
        year_built = st.slider("Año construcción", 1872, 2010, 1995)
        year_remod = st.slider("Año remodelación", 1950, 2010, 2000)
        garage_cars = st.selectbox("Capacidad garaje (autos)", [0,1,2,3,4], index=2)
        fireplaces = st.selectbox("Chimeneas", [0,1,2,3], index=0)

    with c3:
        st.markdown("**🛏️ Habitaciones & Extras**")
        full_bath = st.selectbox("Baños completos", [0,1,2,3], index=2)
        half_bath = st.selectbox("Medios baños", [0,1,2], index=0)
        bedroom = st.selectbox("Habitaciones", [1,2,3,4,5,6], index=2)
        tot_rms = st.selectbox("Habitaciones totales sobre nivel", list(range(2,15)), index=5)
        ms_subclass = st.selectbox("Tipo de construcción", [20,30,60,70,80,90,120,160], index=0)
        mo_sold = st.selectbox("Mes de venta", list(range(1,13)), index=5)

    # Build input dict with defaults for all feature cols
    input_dict = {}
    defaults = {c: df_enc[c].median() for c in feature_cols}
    input_dict.update(defaults)

    # Override with user inputs
    user_vals = {
        'GrLivArea': gr_liv, 'TotalBsmtSF': total_bsmt, '1stFlrSF': first_flr,
        '2ndFlrSF': second_flr, 'LotArea': lot_area, 'GarageArea': garage_area,
        'OverallQual': overall_qual, 'OverallCond': overall_cond,
        'YearBuilt': year_built, 'YearRemodAdd': year_remod,
        'GarageCars': garage_cars, 'Fireplaces': fireplaces,
        'FullBath': full_bath, 'HalfBath': half_bath,
        'BedroomAbvGr': bedroom, 'TotRmsAbvGrd': tot_rms,
        'MSSubClass': ms_subclass, 'MoSold': mo_sold,
    }
    input_dict.update(user_vals)

    X_pred = pd.DataFrame([input_dict])[feature_cols]

    if st.button("🔮 Predecir Precio", use_container_width=True):
        model = trained_models[selected_model]
        price = model.predict(X_pred)[0]
        price_low = price * 0.90
        price_high = price * 1.10

        st.markdown(f"""
        <div class="prediction-box">
            <p class="prediction-label">Precio Estimado · {selected_model}</p>
            <p class="prediction-value">${price:,.0f}</p>
            <p class="prediction-range">Rango estimado: ${price_low:,.0f} — ${price_high:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Comparison across models
        st.markdown('<p class="section-title">Comparación entre modelos</p>', unsafe_allow_html=True)
        all_preds = {n: m.predict(X_pred)[0] for n, m in trained_models.items()}
        fig_cmp = px.bar(
            x=list(all_preds.keys()),
            y=list(all_preds.values()),
            color=list(all_preds.values()),
            color_continuous_scale=[[0,'#0f3460'],[0.5,'#e8b86d'],[1,'#c44536']],
            labels={'x':'Modelo','y':'Precio Predicho ($)'},
        )
        fig_cmp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', showlegend=False,
            coloraxis_showscale=False, height=320,
            yaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
        )
        fig_cmp.add_hline(y=price, line_dash='dash', line_color='#e8b86d',
                          annotation_text=f"Modelo seleccionado: ${price:,.0f}")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2: RESULTADOS DE MODELOS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Métricas por Modelo</p>', unsafe_allow_html=True)

    # Summary KPIs
    best = results[best_model_name]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Mejor Modelo</div>
            <div class="metric-value" style="font-size:1.2rem">{best_model_name}</div>
        </div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">R² Test (mejor)</div>
            <div class="metric-value">{best['test_r2']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RMSE Test (mejor)</div>
            <div class="metric-value">${best['test_rmse']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAE Test (mejor)</div>
            <div class="metric-value">${best['test_mae']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Results table
    st.markdown('<p class="section-title">Tabla Comparativa</p>', unsafe_allow_html=True)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)

    table_html = """<table class="model-table">
    <tr>
        <th>#</th><th>Modelo</th>
        <th>R² Train</th><th>R² Test</th>
        <th>RMSE Train</th><th>RMSE Test</th>
        <th>MAE Train</th><th>MAE Test</th>
    </tr>"""
    for i, (name, m) in enumerate(sorted_models, 1):
        row_class = 'best-row' if name == best_model_name else ''
        badge = f'<span class="rank-badge">{i}</span>'
        ovf = '⚠️' if (m['train_r2'] - m['test_r2']) > 0.15 else ''
        table_html += f"""<tr class="{row_class}">
            <td>{badge}</td>
            <td><strong>{name}</strong> {ovf}</td>
            <td>{m['train_r2']:.4f}</td><td>{m['test_r2']:.4f}</td>
            <td>${m['train_rmse']:,.0f}</td><td>${m['test_rmse']:,.0f}</td>
            <td>${m['train_mae']:,.0f}</td><td>${m['test_mae']:,.0f}</td>
        </tr>"""
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Charts
    st.markdown('<p class="section-title">Visualizaciones</p>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        # R² comparison train vs test
        names = [x[0] for x in sorted_models]
        r2_train = [x[1]['train_r2'] for x in sorted_models]
        r2_test = [x[1]['test_r2'] for x in sorted_models]
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(name='Train', x=names, y=r2_train, marker_color='#4caf93', opacity=0.7))
        fig_r2.add_trace(go.Bar(name='Test', x=names, y=r2_test, marker_color='#e8b86d'))
        fig_r2.update_layout(
            title='R² — Train vs Test', barmode='group',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', height=350,
            yaxis=dict(range=[0,1.05], gridcolor='rgba(255,255,255,0.07)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with ch2:
        # RMSE comparison
        rmse_train = [x[1]['train_rmse'] for x in sorted_models]
        rmse_test = [x[1]['test_rmse'] for x in sorted_models]
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(name='Train', x=names, y=rmse_train, marker_color='#4caf93', opacity=0.7))
        fig_rmse.add_trace(go.Bar(name='Test', x=names, y=rmse_test, marker_color='#c44536'))
        fig_rmse.update_layout(
            title='RMSE — Train vs Test ($)', barmode='group',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', height=350,
            yaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Scatter: predicted vs actual for selected model
    st.markdown('<p class="section-title">Predicho vs Real — Mejor Modelo</p>', unsafe_allow_html=True)
    sel_for_scatter = st.selectbox("Modelo para gráfico", list(trained_models.keys()),
                                    index=list(trained_models.keys()).index(best_model_name),
                                    key="scatter_model")
    y_pred_test = trained_models[sel_for_scatter].predict(X_test)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test, y=y_pred_test, mode='markers',
        marker=dict(color='#e8b86d', size=5, opacity=0.6),
        name='Observaciones'
    ))
    mn, mx = y_test.min(), y_test.max()
    fig_scatter.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                      line=dict(color='#c44536', dash='dash'), name='Línea perfecta'))
    fig_scatter.update_layout(
        title=f'{sel_for_scatter} — Predicho vs Real',
        xaxis_title='Precio Real ($)', yaxis_title='Precio Predicho ($)',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature importance (if available)
    model_fi = trained_models[sel_for_scatter]
    if hasattr(model_fi, 'feature_importances_'):
        fi = pd.Series(model_fi.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation='h',
                        color=fi.values, color_continuous_scale=[[0,'#0f3460'],[1,'#e8b86d']],
                        labels={'x':'Importancia','y':'Variable'})
        fig_fi.update_layout(
            title=f'Top 15 Variables Importantes — {sel_for_scatter}',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', height=420, showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange='reversed'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: EDA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Análisis Exploratorio de Datos</p>', unsafe_allow_html=True)

    # Summary stats
    n_obs = len(df)
    n_feat = df.shape[1] - 1
    price_med = df['SalePrice'].median()
    price_mean = df['SalePrice'].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Observaciones</div>
            <div class="metric-value">{n_obs:,}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Variables</div>
            <div class="metric-value">{n_feat}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Precio Mediano</div>
            <div class="metric-value">${price_med:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Precio Promedio</div>
            <div class="metric-value">${price_mean:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Distribution of SalePrice
    st.markdown('<p class="section-title">Distribución del Precio de Venta</p>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        fig_hist = px.histogram(df, x='SalePrice', nbins=50,
                                color_discrete_sequence=['#e8b86d'])
        fig_hist.update_layout(
            title='Distribución de SalePrice',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', height=320,
            xaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Precio ($)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Frecuencia'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with dc2:
        fig_log = px.histogram(df, x=np.log(df['SalePrice']), nbins=50,
                               color_discrete_sequence=['#4caf93'])
        fig_log.update_layout(
            title='Distribución de log(SalePrice)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f0f0', height=320,
            xaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='log(Precio)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Frecuencia'),
        )
        st.plotly_chart(fig_log, use_container_width=True)

    # Scatter: variable vs SalePrice
    st.markdown('<p class="section-title">Relación entre Variables y SalePrice</p>', unsafe_allow_html=True)
    num_cols = ['GrLivArea','TotalBsmtSF','LotArea','GarageArea','YearBuilt','MasVnrArea','1stFlrSF']
    sel_var = st.selectbox("Variable numérica", num_cols)
    fig_scat = px.scatter(df, x=sel_var, y='SalePrice',
                          color='OverallQual', color_continuous_scale='YlOrRd',
                          opacity=0.6, trendline='ols',
                          labels={'color':'Calidad'})
    fig_scat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=380,
        xaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='SalePrice ($)'),
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    # Box plots by categorical
    st.markdown('<p class="section-title">Precio por Categoría</p>', unsafe_allow_html=True)
    cat_col = st.selectbox("Variable categórica", ['OverallQual','BldgType','HouseStyle','GarageCars','Fireplaces','FullBath'])
    order = sorted(df[cat_col].unique())
    fig_box = px.box(df, x=cat_col, y='SalePrice', color=cat_col,
                     color_discrete_sequence=px.colors.sequential.Plasma,
                     category_orders={cat_col: order})
    fig_box.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=380, showlegend=False,
        xaxis=dict(title=cat_col),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='SalePrice ($)'),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.markdown('<p class="section-title">Mapa de Correlaciones</p>', unsafe_allow_html=True)
    corr_cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','YearBuilt',
                 'GarageArea','1stFlrSF','FullBath','TotRmsAbvGrd','GarageCars',
                 'Fireplaces','LotArea','OverallCond']
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         text_auto='.2f', aspect='auto')
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=500,
    )
    fig_corr.update_traces(textfont_size=9)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Price by neighborhood
    st.markdown('<p class="section-title">Precio Mediano por Vecindario</p>', unsafe_allow_html=True)
    nbh = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).reset_index()
    fig_nbh = px.bar(nbh, x='Neighborhood', y='SalePrice',
                     color='SalePrice', color_continuous_scale=[[0,'#0f3460'],[0.5,'#e8b86d'],[1,'#c44536']])
    fig_nbh.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=360, showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Precio Mediano ($)'),
    )
    st.plotly_chart(fig_nbh, use_container_width=True)

    # Year built trend
    st.markdown('<p class="section-title">Precio Promedio por Año de Construcción</p>', unsafe_allow_html=True)
    yr_trend = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
    fig_yr = px.line(yr_trend, x='YearBuilt', y='SalePrice',
                     color_discrete_sequence=['#e8b86d'])
    fig_yr.update_traces(line_width=2)
    fig_yr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0', height=320,
        xaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Año de Construcción'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Precio Promedio ($)'),
    )
    st.plotly_chart(fig_yr, use_container_width=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#3a4a6b; margin-top:40px; padding:20px;
            border-top:1px solid rgba(232,184,109,0.1); font-size:0.8rem;">
    Ames Housing Predictor · Estadística Multivariada · Datos simulados basados en el dataset original de Ames, Iowa
</div>
""", unsafe_allow_html=True)