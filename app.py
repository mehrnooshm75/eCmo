import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# تنظیمات صفحه
st.set_page_config(page_title="ROC Analysis", layout="wide")
st.title("مقایسه عملکرد مدل هوش مصنوعی و مدل سنتی")

# تابع محاسباتی
def generate_roc_curve(auc):
    # جلوگیری از خطای عددی برای AUC=0.5
    if auc == 0.5:
        return np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    d_prime = norm.ppf(auc) * np.sqrt(2)
    fpr = np.linspace(0, 1, 100)
    tpr = norm.cdf(norm.ppf(fpr) + d_prime)
    return fpr, tpr

# اسلایدرها
col1, col2 = st.columns(2)
with col1:
    auc_ai = st.slider("دقت مدل هوش مصنوعی (AI AUC)", 0.50, 1.00, 0.90)
with col2:
    auc_trad = st.slider("دقت مدل سنتی (Traditional AUC)", 0.50, 1.00, 0.68)

# محاسبات
fpr_ai, tpr_ai = generate_roc_curve(auc_ai)
fpr_trad, tpr_trad = generate_roc_curve(auc_trad)

# ساخت نمودار
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random Guess'))
fig.add_trace(go.Scatter(x=np.linspace(0,1,100), y=tpr_trad, name=f'Traditional: {auc_trad:.2f}', line=dict(color='#e67e22', width=3)))
fig.add_trace(go.Scatter(x=np.linspace(0,1,100), y=tpr_ai, name=f'AI Model: {auc_ai:.2f}', line=dict(color='#1abc9c', width=4)))

fig.update_layout(
    title="نمودار مقایسه‌ای ROC", 
    xaxis_title="1 - Specificity (FPR)", 
    yaxis_title="Sensitivity (TPR)", 
    height=500
)
st.plotly_chart(fig, use_container_width=True)
