import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import arabic_reshaper
from bidi.algorithm import get_display
from ipywidgets import interact, FloatSlider

# تابع کمکی برای اصلاح متون فارسی
def farsi(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# تعریف تابع برای شبیه‌سازی واقعی منحنی ROC براساس AUC
def generate_roc_curve(auc, num_points=100):
    if auc == 0.5:
        return np.linspace(0, 1, num_points), np.linspace(0, 1, num_points)
    d_prime = norm.ppf(auc) * np.sqrt(2)
    fpr = np.linspace(0, 1, num_points)
    tpr = norm.cdf(norm.ppf(fpr) + d_prime)
    tpr[0], tpr[-1] = 0.0, 1.0
    return fpr, tpr

def plot_interactive_roc(auc_ai=0.90, auc_traditional=0.68):
    fpr_ai, tpr_ai = generate_roc_curve(auc_ai)
    fpr_trad, tpr_trad = generate_roc_curve(auc_traditional)
    
    fig = go.Figure()
    
    # اضافه کردن Traceها
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), 
                             name=farsi('Random Guess Line (AUC = 0.50)')))
    fig.add_trace(go.Scatter(x=fpr_trad, y=tpr_trad, mode='lines', line=dict(color='#e67e22', width=3), 
                             name=f'Traditional Scores (AUC = {auc_traditional:.2f})'))
    fig.add_trace(go.Scatter(x=fpr_ai, y=tpr_ai, mode='lines', line=dict(color='#1abc9c', width=4), 
                             name=f'AI / ML Models (AUC = {auc_ai:.2f})'))
    
    fig.update_layout(title=dict(text="Interactive ROC Curves: AI vs Traditional", x=0.5),
                      xaxis_title="1 - Specificity (False Positive Rate)",
                      yaxis_title="Sensitivity (True Positive Rate)",
                      xaxis=dict(range=[-0.02, 1.02], fixedrange=True),
                      yaxis=dict(range=[-0.02, 1.02], fixedrange=True),
                      legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                      hovermode="x unified", margin=dict(l=40, r=40, t=40, b=40), height=500)
    
    # --- این خط جادویی فایل را در پوشه شما می‌سازد ---
    fig.write_html("My_Final_Interactive_Chart.html", include_plotlyjs='cdn')
    
    fig.show(renderer="notebook")

# ایجاد کنترل‌های تعاملی
interact(
    plot_interactive_roc,
    auc_ai=FloatSlider(value=0.90, min=0.50, max=1.00, step=0.01, description=farsi('AI Accuracy'), continuous_update=True),
    auc_traditional=FloatSlider(value=0.68, min=0.50, max=1.00, step=0.01, description=farsi('Tradi Accuracy'), continuous_update=True)
);

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# تنظیمات صفحه
st.set_page_config(page_title="ROC Analysis", layout="wide")
st.title("مقایسه عملکرد مدل هوش مصنوعی و مدل سنتی")

# تابع محاسباتی
def generate_roc_curve(auc):
    d_prime = norm.ppf(auc) * np.sqrt(2)
    fpr = np.linspace(0, 1, 100)
    tpr = norm.cdf(norm.ppf(fpr) + d_prime)
    return fpr, tpr

# اسلایدرها در ستون‌های کناری
col1, col2 = st.columns(2)
with col1:
    auc_ai = st.slider("دقت مدل هوش مصنوعی (AI AUC)", 0.5, 1.0, 0.90)
with col2:
    auc_trad = st.slider("دقت مدل سنتی (Traditional AUC)", 0.5, 1.0, 0.68)

# محاسبات
fpr_ai, tpr_ai = generate_roc_curve(auc_ai)
fpr_trad, tpr_trad = generate_roc_curve(auc_trad)

# ساخت نمودار
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random Guess'))
fig.add_trace(go.Scatter(x=np.linspace(0,1,100), y=tpr_trad, name=f'Traditional: {auc_trad:.2f}', line=dict(color='#e67e22')))
fig.add_trace(go.Scatter(x=np.linspace(0,1,100), y=tpr_ai, name=f'AI Model: {auc_ai:.2f}', line=dict(color='#1abc9c', width=3)))

fig.update_layout(title="Interactive ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=500)
st.plotly_chart(fig, width='stretch')

