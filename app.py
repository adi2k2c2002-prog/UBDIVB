
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide', page_title='Universal Bank - Marketing AI Dashboard')

def load_sample_df(n=1000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({
        'ID': np.arange(1, n+1),
        'Age': np.random.randint(22, 67, size=n),
        'Experience': np.clip(np.random.randint(0, 45, size=n), 0, 44),
        'Income': np.round(np.random.gamma(2.5, 30, size=n)).astype(int),
        'ZIP Code': np.random.choice([94607, 94112, 94117, 94109, 95014], size=n),
        'Family': np.random.choice([1,2,3,4], size=n, p=[0.25,0.35,0.25,0.15]),
        'CCAvg': np.round(np.random.exponential(1.2, size=n),2),
        'Education': np.random.choice([1,2,3], size=n, p=[0.6,0.25,0.15]),
        'Mortgage': np.random.choice([0,50,100,150,200], size=n, p=[0.75,0.08,0.07,0.06,0.04]),
        'Securities Account': np.random.choice([0,1], size=n, p=[0.9,0.1]),
        'CD Account': np.random.choice([0,1], size=n, p=[0.85,0.15]),
        'Online': np.random.choice([0,1], size=n, p=[0.3,0.7]),
        'CreditCard': np.random.choice([0,1], size=n, p=[0.7,0.3])
    })
    logits = (0.03*(df['Income']) + 0.8*(df['Education']==3).astype(int) + 1.2*df['CD Account'] + 1.1*df['CCAvg']) - 40
    probs = 1 / (1 + np.exp(-0.04*(logits)))
    df['Personal Loan'] = (np.random.rand(n) < probs).astype(int)
    df = df[['ID','Personal Loan','Age','Experience','Income','ZIP Code','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']]
    return df

@st.cache_data
def prep_data(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target_col = None
    for c in df.columns:
        if c.lower().replace('_','').replace(' ','') == 'personalloan':
            target_col = c
            break
    if target_col is None:
        raise ValueError('No Personal Loan column found. Expecting "Personal Loan" column.')
    X = df.drop(columns=[target_col, 'ID'], errors='ignore')
    y = df[target_col]
    X_mod = X.drop(columns=[c for c in X.columns if 'zip' in c.lower()], errors='ignore')
    return X_mod, y, df

def train_models(X_train, y_train):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        fitted[name] = m
        try:
            cv_auc = cross_val_score(m, X_train, y_train, cv=cv, scoring='roc_auc')
            cv_scores[name] = cv_auc.mean()
        except Exception:
            cv_scores[name] = None
    return fitted, cv_scores

def compute_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    metrics = {}
    metrics['train_acc'] = accuracy_score(y_train, y_train_pred)
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    metrics['precision'] = precision_score(y_test, y_test_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_test_pred, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_test_pred, zero_division=0)
    metrics['auc'] = roc_auc_score(y_test, y_test_proba)
    metrics['y_test_proba'] = y_test_proba
    metrics['y_test_pred'] = y_test_pred
    metrics['y_train_pred'] = y_train_pred
    return metrics

def plot_roc(models, X_test, y_test):
    fig = go.Figure()
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:,1]
        else:
            proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
    fig.update_layout(title='ROC Curve - All Models', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

def confusion_matrix_plot(cm, title='Confusion Matrix'):
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], text=cm, texttemplate="%{text}", colorscale='Viridis'))
    fig.update_layout(title=title, width=450, height=400)
    return fig

def feature_importance_plot(importances, title='Feature importances'):
    df = importances.reset_index()
    df.columns = ['feature','importance']
    df = df.sort_values('importance', ascending=True)
    fig = go.Figure(go.Bar(x=df['importance'], y=df['feature'], orientation='h'))
    fig.update_layout(title=title, width=700, height=500)
    return fig

st.title('Universal Bank - Marketing AI Dashboard (Streamlit)')
menu = st.sidebar.selectbox('Choose page', ['Dashboard', 'Model Trainer', 'Bulk Predict / Upload'])

st.sidebar.markdown('**Data source**')
uploaded = st.sidebar.file_uploader('Upload UniversalBank CSV (has column "Personal Loan")', type=['csv'])
use_sample = st.sidebar.checkbox('Use sample dataset (if you don\'t upload)', value=True)
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if use_sample:
        df = load_sample_df(2000)
    else:
        st.sidebar.warning('Please upload a CSV or enable sample dataset.')
        st.stop()

if menu == 'Dashboard':
    st.header('Exploratory & Marketing Insights')
    st.markdown('### Dataset snapshot')
    st.dataframe(df.head())
    st.markdown('---')
    st.subheader('1) Income decile vs Personal Loan acceptance rate (by Education level)')
    df['income_decile'] = pd.qcut(df['Income'].rank(method='first'), 10, labels=False) + 1
    pivot = df.groupby(['income_decile','Education'])['Personal Loan'].mean().reset_index()
    fig = px.line(pivot, x='income_decile', y='Personal Loan', color=pivot['Education'].astype(str), markers=True, labels={'Personal Loan':'Acceptance Rate','income_decile':'Income decile','color':'Education'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('2) Education vs Family (stacked) — Acceptance and Count')
    grp = df.groupby(['Education','Family'])['Personal Loan'].agg(['mean','count']).reset_index()
    fig2 = px.bar(grp, x='Education', y='count', color='Family', barmode='group', facet_col='Education', labels={'count':'Count'}, title='Family sizes across education levels (counts)')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('Acceptance rates by Education & Family:')
    st.dataframe(grp.pivot_table(index='Education', columns='Family', values='mean').round(3))

    st.markdown('---')
    st.subheader('3) Heatmap: Income decile vs CCAvg decile — Acceptance rate (2D segmentation)')
    df['ccavg_decile'] = pd.qcut(df['CCAvg'].rank(method='first'), 10, labels=False) + 1
    heat = df.groupby(['income_decile','ccavg_decile'])['Personal Loan'].mean().unstack()
    fig3 = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns.astype(str), y=heat.index.astype(str), colorscale='RdYlGn', colorbar_title='Acceptance Rate'))
    fig3.update_layout(title='Acceptance rate by Income decile vs CCAvg decile', xaxis_title='CCAvg decile', yaxis_title='Income decile', width=800, height=600)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('---')
    st.subheader('4) Predicted propensity distribution by CD Account (requires model training below or fallback)')
    try:
        X_mod, y, _ = prep_data(df)
        quick_model = RandomForestClassifier(n_estimators=100, random_state=42)
        quick_model.fit(X_mod, y)
        proba = quick_model.predict_proba(X_mod)[:,1]
        df['propensity'] = proba
        fig4 = px.box(df, x='CD Account', y='propensity', points='outliers', labels={'propensity':'Predicted propensity','CD Account':'CD Account'})
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.error(f'Could not compute propensity: {e}')

    st.markdown('---')
    st.subheader('5) Lift chart for a quick RandomForest model (Marketing actionable)')
    try:
        X_mod, y, _ = prep_data(df)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_mod, y)
        proba = rf.predict_proba(X_mod)[:,1]
        lift_df = pd.DataFrame({'y': y, 'proba': proba})
        lift_df = lift_df.sort_values('proba', ascending=False).reset_index(drop=True)
        lift_df['cum_resp'] = lift_df['y'].cumsum()
        lift_df['cum_pop_pct'] = (lift_df.index + 1) / len(lift_df)
        lift_df['cum_resp_rate'] = lift_df['cum_resp'] / (lift_df.index + 1)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=lift_df['cum_pop_pct'], y=lift_df['cum_resp_rate'], mode='lines', name='Model capture rate'))
        fig5.add_trace(go.Scatter(x=[0,1], y=[y.mean(), y.mean()], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig5.update_layout(title='Lift / capture chart', xaxis_title='Population captured (fraction)', yaxis_title='Response rate', width=800, height=600)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('Use the lift chart to identify top X% customers to target for campaigns (e.g., top 5-10% by propensity).')
    except Exception as e:
        st.error(f'Could not compute lift: {e}')

elif menu == 'Model Trainer':
    st.header('Train models and generate metrics (Decision Tree, Random Forest, Gradient Boosting)')
    st.markdown('Note: training uses 70/30 split and 5-fold CV on the training set.')
    run = st.button('Train & Evaluate Models')
    if run:
        try:
            X_mod, y, _ = prep_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size=0.3, random_state=42, stratify=y)
            models, cv_scores = train_models(X_train, y_train)
            rows = []
            for name, m in models.items():
                met = compute_metrics(m, X_train, y_train, X_test, y_test)
                rows.append({
                    'Algorithm': name,
                    'Training accuracy': round(met['train_acc'],4),
                    'Testing accuracy': round(met['test_acc'],4),
                    'Precision': round(met['precision'],4),
                    'Recall': round(met['recall'],4),
                    'F1 score': round(met['f1'],4),
                    'AUC': round(met['auc'],4),
                    'CV mean AUC (train 5-fold)': round(cv_scores.get(name, 0) or 0,4)
                })
            results_df = pd.DataFrame(rows).set_index('Algorithm')
            st.dataframe(results_df)
            plot_roc(models, X_test, y_test)
            cols1, cols2 = st.columns(2)
            for name, m in models.items():
                with cols1:
                    st.subheader(f'{name} - Test Confusion Matrix')
                    cm = confusion_matrix(y_test, m.predict(X_test))
                    st.plotly_chart(confusion_matrix_plot(cm, title=f'{name} - Test Confusion Matrix'))
                with cols2:
                    st.subheader(f'{name} - Feature Importance')
                    if hasattr(m, 'feature_importances_'):
                        imp = pd.Series(m.feature_importances_, index=X_mod.columns).sort_values(ascending=False)
                        st.plotly_chart(feature_importance_plot(imp, title=f'{name} - Feature importances'))
                    else:
                        st.info('No feature importance for this model.')
            st.session_state['trained_models'] = models
            st.session_state['X_columns'] = X_mod.columns.tolist()
            st.success('Models trained and stored for session. You can now use the Predict tab.')
        except Exception as e:
            st.error(f'Error during training: {e}')

elif menu == 'Bulk Predict / Upload':
    st.header('Upload a new dataset and predict "Personal Loan" label')
    new_uploaded = st.file_uploader('Upload CSV for prediction', type=['csv'])
    if new_uploaded is not None:
        new_df = pd.read_csv(new_uploaded)
        st.dataframe(new_df.head())
        if 'trained_models' in st.session_state and 'X_columns' in st.session_state:
            models = st.session_state['trained_models']
            cols = st.session_state['X_columns']
        else:
            st.info('No trained models in session — training a quick RandomForest on current data (this may take a moment).')
            X_mod_all, y_all, _ = prep_data(df)
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_mod_all, y_all)
            models = {'Random Forest (quick)': rf}
            cols = X_mod_all.columns.tolist()
        new_df_prep = new_df.copy()
        if 'ID' in new_df_prep.columns:
            new_df_prep = new_df_prep.drop(columns=['ID'])
        if 'Personal Loan' in new_df_prep.columns:
            new_df_prep = new_df_prep.drop(columns=['Personal Loan'])
        new_df_prep = new_df_prep.drop(columns=[c for c in new_df_prep.columns if 'zip' in c.lower()], errors='ignore')
        missing = [c for c in cols if c not in new_df_prep.columns]
        if missing:
            st.warning(f'Missing columns in uploaded file required for prediction: {missing}. Please add them or use a compatible file.')
        else:
            X_new = new_df_prep[cols]
            proba_sum = np.zeros(len(X_new))
            for name,m in models.items():
                proba_sum += m.predict_proba(X_new)[:,1] if hasattr(m, 'predict_proba') else m.decision_function(X_new)
            proba_avg = proba_sum / len(models)
            pred_label = (proba_avg >= 0.5).astype(int)
            out = new_df.copy()
            out['Personal Loan (pred)'] = pred_label
            out['Personal Loan (prob)'] = np.round(proba_avg,4)
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions as CSV', data=csv, file_name='predictions.csv', mime='text/csv')
    else:
        st.info('Upload a CSV to run batch predictions.')
else:
    st.write('Choose a menu option.')

st.sidebar.markdown('---')
st.sidebar.markdown('Built for XYZ Bank — HOD Marketing demo.')
st.sidebar.markdown('Main file: app.py')
