"""FitScore Studio: run with `streamlit run app.py`; evaluate with `python app.py --evaluate`."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ['Height_cm', 'Weight_kg', 'Steps_Per_Day', 'Water_Intake_L', 'Sleep_Hours']
TARGET = 'Fitness_Score'
DATA = Path(__file__).with_name('fitness_score_dataset.csv')
LABELS = ['Height (cm)', 'Weight (kg)', 'Daily steps', 'Water intake (L/day)', 'Sleep (hours/night)']


def clean_data(raw):
    """Keep unknown predictor values missing; never invent target labels."""
    required = FEATURES + [TARGET]
    if not set(required).issubset(raw.columns):
        raise ValueError('Dataset must contain: ' + ', '.join(required))
    df = raw[required].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    for col in FEATURES:
        df.loc[df[col] < 0, col] = np.nan
    for col in ['Height_cm', 'Weight_kg']:
        df.loc[df[col] == 0, col] = np.nan
    df.loc[df.Sleep_Hours > 24, 'Sleep_Hours'] = np.nan
    df.loc[~df[TARGET].between(1, 10), TARGET] = np.nan
    missing = df.isna().sum().to_dict()
    before = len(df)
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    if len(df) < 50 or df[TARGET].nunique() < 2:
        raise ValueError('Need at least 50 labelled rows and a varying target.')
    if df[FEATURES].isna().all().any():
        raise ValueError('A predictor has no observed values; median imputation is impossible.')
    return df, {'raw_rows': before, 'usable_rows': len(df), 'removed_rows': before-len(df), 'missing_or_invalid': missing}


def train_models(df):
    """Select using training-only CV MAE. Reserve 20% for final evaluation."""
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    specs = {
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(random_state=42),
                          {'model__max_depth':[3,5,8,None], 'model__min_samples_leaf':[3,10]}),
        'Random Forest': (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
                          {'model__max_depth':[8,None], 'model__min_samples_leaf':[1,3,8]}),
    }
    searches, records = {}, []
    for name, (model, grid) in specs.items():
        steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
        if name == 'Linear Regression':
            steps += [('scaler', StandardScaler())]
        pipe = Pipeline(steps + [('model', model)])
        search = GridSearchCV(pipe, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
        search.fit(Xtr, ytr)
        searches[name] = search
        records.append({'Model':name, 'CV MAE':-search.best_score_,
                        'CV MAE SD':search.cv_results_['std_test_score'][search.best_index_]})
    comparison = pd.DataFrame(records).sort_values('CV MAE').reset_index(drop=True)
    winner = comparison.iloc[0]['Model']
    evaluated = searches[winner].best_estimator_
    pred = evaluated.predict(Xte)
    baseline = DummyRegressor(strategy='mean').fit(Xtr, ytr).predict(Xte)
    metrics = {'MAE':float(mean_absolute_error(yte,pred)),
               'RMSE':float(np.sqrt(mean_squared_error(yte,pred))),
               'R2':float(r2_score(yte,pred)),
               'Baseline MAE':float(mean_absolute_error(yte,baseline)),
               'train_rows':len(Xtr), 'test_rows':len(Xte)}
    # Evaluation uses the train-only model. Deployment refits the selected recipe on all labelled rows.
    deployed = clone(evaluated).fit(X,y)
    return {'model':deployed, 'winner':winner, 'comparison':comparison, 'metrics':metrics,
            'params':searches[winner].best_params_,
            'test':pd.DataFrame({'Actual':yte.to_numpy(), 'Predicted':pred})}


def main():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    st.set_page_config(page_title='FitScore Studio', page_icon='🌿', layout='wide')
    st.markdown('''<style>
    .stApp {background:linear-gradient(120deg,#f4faf8 0%,#edf3fa 100%);color:#162c3b}
    .block-container {max-width:1200px;padding-top:2rem}
    h1,h2,h3 {color:#163c47!important;letter-spacing:-.03em}
    .hero {background:linear-gradient(110deg,#123c47,#176b66);padding:30px 34px;border-radius:24px;color:white;margin-bottom:24px}
    .hero h1 {color:white!important;font-size:2.6rem;margin:0}
    .hero p {color:#dcf0ea;margin-bottom:0}
    [data-testid="stMetric"] {background:white;border:1px solid #dbe7e4;padding:18px;border-radius:18px}
    .stButton>button {border-radius:12px}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="hero"><small>FITSCORE STUDIO / EVERYDAY WELLNESS</small><h1>A little insight into your everyday.</h1><p>Explore how your daily habits relate to the fitness score in this dataset.</p></div>', unsafe_allow_html=True)
    st.caption('Educational fitness estimate • 1–10 dataset scale • Not a medical assessment or diagnosis')
    if not DATA.exists():
        st.error('Place fitness_score_dataset.csv in the same folder as app.py.'); st.stop()
    try:
        raw = pd.read_csv(DATA)
        df, audit = clean_data(raw)
        # Cache is keyed by the data itself, so changed datasets trigger retraining.
        with st.spinner('Comparing regression pipelines… First launch may take a minute.'):
            result = st.cache_resource(show_spinner=False)(train_models)(df)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(str(exc)); st.stop()
    with st.sidebar:
        st.header('🌿 FitScore Studio')
        st.write('Five inputs. One transparent prediction.')
        st.metric('Selected model', result['winner'])
        st.caption('Selected by lowest five-fold cross-validation MAE, using only the training split.')
        st.divider()
        st.write('Your inputs are used for this session. This app does not write them to a database or file.')
        st.caption('Hosting-provider logging and privacy policies apply separately.')
    predict_tab, model_tab, data_tab = st.tabs(['✦ Your prediction', '◈ Model performance', '▦ Dataset & preprocessing'])
    with predict_tab:
        left, right = st.columns([1.05,1], gap='large')
        with left:
            st.subheader('Tell us about your routine')
            st.write('Use your typical day. Select unknown fields to use the learned training median.')
            unknown = st.multiselect('Values you do not know', FEATURES, format_func=lambda c: LABELS[FEATURES.index(c)])
            with st.form('profile_form'):
                vals = {}
                limits = [(50.,250.,.1),(10.,400.,.1),(0,100000,100),(0.,15.,.1),(0.,24.,.1)]
                for col, label, (lo,hi,step) in zip(FEATURES,LABELS,limits):
                    default = float(df[col].median())
                    if col == 'Steps_Per_Day': default = int(default)
                    val = st.number_input(label,min_value=lo,max_value=hi,value=default,step=step,disabled=col in unknown)
                    vals[col] = np.nan if col in unknown else val
                submit = st.form_submit_button('Calculate my fitness score →',type='primary',width='stretch')
            st.caption('Input limits are entry safeguards, not recommended health targets.')
        with right:
            st.subheader('Your fitness snapshot')
            if submit:
                st.session_state['profile'] = vals
            if 'profile' not in st.session_state:
                st.info('Complete the form and calculate your score to see your snapshot.')
            else:
                profile = st.session_state['profile']
                if all(pd.isna(v) for v in profile.values()):
                    st.warning('Provide at least one known value for a meaningful prediction.')
                else:
                    frame = pd.DataFrame([profile],columns=FEATURES)
                    score = float(result['model'].predict(frame)[0])
                    fig = go.Figure(go.Indicator(mode='gauge+number',value=score,
                        number={'suffix':' / 10','font':{'size':46,'color':'#176b66'}},
                        gauge={'axis':{'range':[1,10]},'bar':{'color':'#176b66'},'bgcolor':'#deebe7',
                               'steps':[{'range':[1,10],'color':'#deebe7'}]}))
                    fig.update_layout(height=290,margin=dict(l=30,r=30,t=40,b=10),paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig,width='stretch')
                    st.caption('Raw regression prediction; no clinical categories are assigned.')
                    if not 1 <= score <= 10: st.warning('Prediction is outside the observed 1–10 target range. Interpret cautiously.')
                    outside = [c for c in FEATURES if pd.notna(profile[c]) and not df[c].min() <= profile[c] <= df[c].max()]
                    if outside: st.warning('Outside observed training-data ranges: '+', '.join(outside)+'. Reliability may be lower.')
                    imputed = [c for c in FEATURES if pd.isna(profile[c])]
                    if imputed: st.info('Median substitution used for: '+', '.join(imputed)+'. This adds uncertainty.')
                    st.metric('Held-out average absolute error',f"{result['metrics']['MAE']:.2f} score points")
                    st.caption('An average test error, not a personal confidence interval. Combinations of inputs can also be unfamiliar to the model.')
                    report = {**profile,'Predicted_Fitness_Score':score,'Model':result['winner']}
                    st.download_button('Download my result',pd.DataFrame([report]).to_csv(index=False),'fitness_prediction.csv','text/csv')
    with model_tab:
        st.subheader('A model chosen by evidence')
        cols = st.columns(3)
        for col,key in zip(cols,['MAE','RMSE','R2']): col.metric('Test '+key,f"{result['metrics'][key]:.3f}")
        st.dataframe(result['comparison'],hide_index=True,width='stretch')
        st.caption('Lower CV MAE is better. SD describes variability across the five validation folds. Test scores above belong only to the selected model.')
        st.write(f"Training: {result['metrics']['train_rows']} rows · Holdout: {result['metrics']['test_rows']} rows · Random seed: 42")
        st.write(f"Mean-only baseline test MAE: {result['metrics']['Baseline MAE']:.3f}")
        fig = px.scatter(result['test'],x='Actual',y='Predicted',color_discrete_sequence=['#176b66'],opacity=.6)
        fig.add_shape(type='line',x0=1,y0=1,x1=10,y1=10,line=dict(color='#dd9b47',dash='dash'))
        fig.update_layout(height=360,title='Held-out predictions vs actual scores',paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,width='stretch')
        st.write('Best parameters:',result['params'] or 'Default linear regression parameters')
        st.info('After evaluation, the winning pipeline is refitted on all labelled rows for app predictions. Test metrics describe the earlier train-only model, not an independent test of that final refit.')
    with data_tab:
        st.subheader('Know the data behind the number')
        st.write(f"{audit['raw_rows']:,} original rows · {len(FEATURES)} numeric predictors · target: {TARGET}")
        st.dataframe(pd.DataFrame({'Missing / invalid':audit['missing_or_invalid']}),width='stretch')
        st.write(f"Removed rows (missing/invalid targets or exact duplicates): {audit['removed_rows']}")
        st.dataframe(df.describe().T,width='stretch')
        st.markdown('''**Preprocessing strategy**
- Numeric conversion; nonnumeric entries, infinities and impossible negatives become missing. Zero height/weight and sleep above 24 hours also become missing.
- Missing predictors: median imputation **inside each fitted pipeline**. Numeric columns need no categorical encoding.
- Linear regression: standard scaling after imputation. Tree models: no scaling required.
- Missing targets: remove those rows; do not impute labels. Exact duplicate rows are removed before splitting.
- Preserve observed extremes; do not silently remove valid unusual values.
- Split 80/20 first, tune and compare with five-fold CV on the training portion, evaluate the winner once on the holdout, then refit on all labelled data.
''')
        st.warning('Dataset provenance, score construction, clinical validation and population coverage were not supplied. This model learns the supplied score and cannot establish overall health, causation or clinical risk.')
        st.dataframe(df.head(20),hide_index=True,width='stretch')


if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        df, audit = clean_data(pd.read_csv(DATA))
        r = train_models(df)
        print(r['comparison'].to_string(index=False))
        print(json.dumps({'winner':r['winner'],'metrics':r['metrics'],'params':r['params'],'audit':audit},indent=2))
    else:
        main()
