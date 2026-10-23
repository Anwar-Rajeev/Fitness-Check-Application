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


HABITS = ['Steps_Per_Day', 'Water_Intake_L', 'Sleep_Hours']
HABIT_NAMES = {'Steps_Per_Day':'Movement', 'Water_Intake_L':'Hydration', 'Sleep_Hours':'Rest & recovery'}
COLORS = {'Steps_Per_Day':'#287d6c', 'Water_Intake_L':'#397bb9', 'Sleep_Hours':'#8262af'}


def goal_snapshot(profile, goals):
    """Compare only measured values to self-selected goals, never imputed values."""
    rows = []
    for c in HABITS:
        v, g = profile[c], goals[c]
        known = pd.notna(v)
        rows.append({'Habit':HABIT_NAMES[c], 'Feature':c, 'Value':v, 'Goal':g,
                     'Known':known, 'Met':bool(known and v >= g),
                     'Progress':min(float(v/g),1.) if known else 0.})
    return rows


def predict_profile(model, profile):
    if all(pd.isna(v) for v in profile.values()):
        raise ValueError('Enter at least one known value before calculating.')
    return float(model.predict(pd.DataFrame([profile], columns=FEATURES))[0])


def main():
    from datetime import date
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    st.set_page_config(page_title='FitScore • Make room for wellbeing',page_icon='🌱',layout='wide')
    st.markdown('''<style>
    .stApp {background:#f7f8f3;color:#20382f}
    .block-container {max-width:1220px;padding-top:2rem;padding-bottom:3rem}
    h1,h2,h3 {color:#203f34!important;letter-spacing:-.025em}
    .hero {position:relative;background:linear-gradient(115deg,#183f33,#2b6b52);border-radius:26px;padding:32px 38px;margin-bottom:18px;color:#fff;overflow:hidden}
    .hero h1 {color:#fff!important;font-size:clamp(2rem,4vw,3.25rem);line-height:1.1;max-width:720px;margin:12px 0}
    .hero p {color:#dcecdf;max-width:660px;font-size:1.05rem;margin-bottom:8px}
    .eyebrow {font-size:.75rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#dae9b6}
    .hero-mark {position:absolute;right:40px;top:24px;font-size:110px;opacity:.15;transform:rotate(-12deg)}
    .pill {display:inline-block;border:1px solid #819e88;border-radius:30px;padding:5px 12px;font-size:.75rem;color:#f5f8ee}
    .note {background:#ecf0e2;border-left:4px solid #799051;border-radius:0 12px 12px 0;padding:14px 18px;color:#35452b;margin:12px 0 22px}
    .habit {background:white;border:1px solid #e0e7dc;border-top:4px solid var(--accent);border-radius:18px;padding:20px;min-height:150px}
    .habit .value {font-size:1.8rem;font-weight:700;color:#203f34;margin:7px 0}
    .habit small {color:#607264}
    .habit .tag {font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;font-weight:700;color:var(--accent)}
    .action {background:#fff;border:1px solid #dee6d7;border-radius:18px;padding:22px;min-height:180px;margin-bottom:12px}
    .action b {color:#244e3d;font-size:1.08rem}
    .action p {color:#566759;line-height:1.6;margin:10px 0 0}
    .stButton>button,.stDownloadButton>button {border-radius:12px;min-height:44px}
    [data-testid="stMetric"] {background:#fff;border:1px solid #e0e7dc;padding:18px;border-radius:16px}
    [data-testid="stSidebar"] {background:#edf1e7}
    [data-baseweb="tab-list"] {gap:12px}
    @media(max-width:650px) {.hero {padding:24px}.hero-mark {display:none}}
    @media(prefers-reduced-motion:no-preference) {.hero {animation:appear .5s ease-out}@keyframes appear {from {opacity:0;transform:translateY(8px)}to {opacity:1;transform:translateY(0)}}}
    </style>''',unsafe_allow_html=True)
    st.markdown('''<div class="hero"><span class="hero-mark">🌱</span><div class="eyebrow">FITSCORE / YOUR DAILY CHECK-IN</div><h1>Small habits.<br>A healthier conversation.</h1><p>Notice your routine. Choose one realistic action. Make a little more room for your wellbeing.</p><span class="pill">Reflect → Explore → Commit</span></div>''',unsafe_allow_html=True)
    st.caption('Educational fitness estimate • Dataset scale 1–10 • Your score is one limited signal, not a medical health assessment.')
    if not DATA.exists():
        st.error('Place fitness_score_dataset.csv beside app.py.');st.stop()
    try:
        df,audit=clean_data(pd.read_csv(DATA))
        with st.spinner('Preparing your workspace and comparing regression models…'):
            result=st.cache_resource(show_spinner=False)(train_models)(df)
    except (OSError,ValueError,pd.errors.ParserError) as exc:
        st.error(str(exc));st.stop()
    st.session_state.setdefault('checkins',[])
    with st.sidebar:
        st.header('🌱 FitScore')
        st.write('Awareness is a useful first step.')
        st.divider()
        st.subheader('Make the goals yours')
        st.caption('Editable example goals, not medical targets. Reaching a number does not certify good health. More water or sleep is not always better.')
        goals={
            'Steps_Per_Day':st.number_input('My daily step goal',min_value=100,max_value=50000,value=7000,step=100,key='goal_steps'),
            'Water_Intake_L':st.number_input('My water tracking goal (L)',min_value=.1,max_value=10.,value=2.,step=.1,key='goal_water'),
            'Sleep_Hours':st.number_input('My sleep goal (hours)',min_value=1.,max_value=12.,value=7.5,step=.5,key='goal_sleep'),
        }
        st.caption('Water needs vary. Follow any fluid guidance from your clinician. Step counts do not measure all types or intensity of physical activity.')
        st.divider()
        st.caption('Session only: no account or database. Download your check-ins before leaving. Hosting-provider policies apply separately.')
    tabs=st.tabs(['☀  My check-in','↗  What if?','✓  My next step','◈  Behind the score'])
    with tabs[0]:
        left,right=st.columns([1,1.05],gap='large')
        with left:
            st.subheader('Start with your real routine')
            st.write('Use a typical day, or today’s values if you are keeping a daily log.')
            unknown=st.multiselect('I do not know these values',FEATURES,format_func=lambda c:LABELS[FEATURES.index(c)],key='unknown_inputs')
            with st.form('checkin_form'):
                a,b=st.columns(2)
                profile={}
                limits=[(50.,250.,.1),(10.,400.,.1),(0,100000,100),(0.,15.,.1),(0.,24.,.1)]
                for i,(c,label,(lo,hi,step)) in enumerate(zip(FEATURES,LABELS,limits)):
                    default=float(df[c].median())
                    if c=='Steps_Per_Day':default=int(default)
                    host=a if i%2==0 else b
                    v=host.number_input(label,min_value=lo,max_value=hi,value=default,step=step,disabled=c in unknown,key='input_'+c)
                    profile[c]=np.nan if c in unknown else v
                submitted=st.form_submit_button('Reveal my snapshot →',type='primary',width='stretch')
            st.caption('Prefilled values are dataset medians: replace them with your own. Entry limits are safeguards, not recommended ranges.')
            if submitted:
                if all(pd.isna(v) for v in profile.values()):
                    st.warning('Enter at least one known value. Your previous snapshot, if any, remains displayed; an entirely imputed profile is not a personal snapshot.')
                else:
                    st.session_state['profile']=profile.copy()
                    st.session_state.pop('scenario_result',None)
                    # Synchronize explorer controls to the newly submitted snapshot.
                    for c in HABITS:
                        st.session_state['what_'+c]=int(profile[c]) if c=='Steps_Per_Day' and pd.notna(profile[c]) else float(profile[c]) if pd.notna(profile[c]) else (int(df[c].median()) if c=='Steps_Per_Day' else float(df[c].median()))
            with st.expander('What does this app actually know?'):
                st.write('It knows five measurements and patterns in 1,000 supplied records. It does not know your age, symptoms, medical history, nutrition, stress, activity intensity or score provenance. A high predicted score cannot rule out a health problem.')
        with right:
            st.subheader('Your fitness snapshot')
            current=st.session_state.get('profile')
            if current is None:
                st.markdown('<div class="action"><b>Your starting point belongs here.</b><p>Enter your routine to reveal a score, see your habits against personal goals, and choose a next step.</p></div>',unsafe_allow_html=True)
                st.info('You do not need a perfect routine to start paying attention to it.')
            else:
                score=predict_profile(result['model'],current)
                fig=go.Figure(go.Indicator(mode='gauge+number',value=score,number={'suffix':' / 10','font':{'size':45,'color':'#245740'}},gauge={'axis':{'range':[1,10],'tickvals':[1,3,5,7,10]},'bar':{'color':'#42785a'},'bgcolor':'#e6eddf','borderwidth':0,'steps':[{'range':[1,10],'color':'#e6eddf'}]}))
                fig.update_layout(height=250,margin=dict(l=25,r=25,t=30,b=10),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#365143'})
                st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
                st.caption('Predicted dataset fitness score. No healthy/unhealthy category is assigned.')
                st.markdown('<div class="note"><b>Let this start a conversation, not end one.</b><br>A score is not a health clearance. The habits you can sustain deserve more attention than a decimal.</div>',unsafe_allow_html=True)
                missing=[c for c in FEATURES if pd.isna(current[c])]
                outside=[c for c in FEATURES if pd.notna(current[c]) and not df[c].min()<=current[c]<=df[c].max()]
                if missing:st.info('Estimated with learned medians for: '+', '.join(missing)+'. These fields are excluded from goal feedback. Missing-input accuracy has not been independently tested.')
                if outside:st.warning('Outside observed data ranges: '+', '.join(outside)+'. The estimate may be less reliable.')
                st.caption(f"Held-out average prediction error: {result['metrics']['MAE']:.2f} points. This is not an individual confidence interval.")
        if current is not None:
            st.divider()
            st.subheader('Three habits worth noticing')
            rows=goal_snapshot(current,goals)
            cols=st.columns(3)
            icons=['🚶','💧','🌙']
            units=['steps','L','h']
            for host,row,icon,unit in zip(cols,rows,icons,units):
                v=row['Value']; value='Unknown' if not row['Known'] else f'{v:,.0f}' if row['Feature']=='Steps_Per_Day' else f'{v:.1f}'
                tag='No measured value' if not row['Known'] else 'Personal goal reached' if row['Met'] else 'Room toward your goal'
                with host:
                    st.markdown(f'<div class="habit" style="--accent:{COLORS[row["Feature"]]}"><div class="tag">{icon} {row["Habit"]}</div><div class="value">{value} <small>{unit if row["Known"] else ""}</small></div><small>Personal goal: {row["Goal"]:g} {unit} · {tag}</small></div>',unsafe_allow_html=True)
                    st.progress(row['Progress'])
                    if row['Known'] and not row['Met']:
                        gap=row['Goal']-v
                        st.caption(f'{gap:,.0f} steps to your selected goal' if row['Feature']=='Steps_Per_Day' else f'{gap:.1f} {unit} to your selected goal')
            st.caption('Progress bars show your own numeric goals, not a health rating. Values above a goal earn no extra credit; more is not automatically healthier.')
            known=sum(r['Known'] for r in rows);met=sum(r['Met'] for r in rows)
            st.write(f'**{met} of {known} measured habits meet your personal goals.** Choose one manageable action in “My next step.”')
    with tabs[1]:
        st.subheader('Explore a different routine')
        st.write('Keep your height and weight fixed. Change one or more habits and compare the model’s outputs.')
        st.info('This is a model sensitivity experiment. It does not predict what will happen to your health, or prove that a habit change causes a score change.')
        current=st.session_state.get('profile')
        if current is None:
            st.write('Complete your first check-in to unlock the explorer.')
        else:
            a,b=st.columns([1,1],gap='large')
            with a:
                scenario=current.copy()
                ranges={'Steps_Per_Day':(0,100000,100),'Water_Intake_L':(0.,15.,.1),'Sleep_Hours':(0.,24.,.1)}
                with st.form('scenario_form'):
                    for c in HABITS:
                        lo,hi,step=ranges[c]
                        default=current[c] if pd.notna(current[c]) else df[c].median()
                        default=int(default) if c=='Steps_Per_Day' else float(default)
                        st.session_state.setdefault('what_'+c,default)
                        scenario[c]=st.slider(LABELS[FEATURES.index(c)],min_value=lo,max_value=hi,step=step,key='what_'+c)
                    compare=st.form_submit_button('Compare this routine',type='primary',width='stretch')
                st.caption('Slider ranges allow exploration; extremes are not recommended targets. Unknown baseline habits start at the learned median.')
                if compare:st.session_state['scenario_result']=scenario.copy()
            with b:
                scenario_saved=st.session_state.get('scenario_result')
                if scenario_saved is None:
                    st.write('Adjust a habit and compare to reveal a side-by-side result.')
                else:
                    base=predict_profile(result['model'],current);sim=predict_profile(result['model'],scenario_saved)
                    ca,cb=st.columns(2);ca.metric('Your snapshot',f'{base:.2f} / 10');cb.metric('Explored routine',f'{sim:.2f} / 10')
                    delta=sim-base
                    st.write(f'**Model output difference: {delta:+.2f} points.**')
                    st.caption('A positive difference is not a promised health benefit. Both estimates have prediction error; the difference has no validated uncertainty interval.')
                    comparison=pd.DataFrame({'Habit':[HABIT_NAMES[c] for c in HABITS], 'Snapshot':[current[c] for c in HABITS],'Explored':[scenario_saved[c] for c in HABITS],'Units':['steps/day','L/day','hours/night']})
                    st.dataframe(comparison,hide_index=True,width='stretch')
                    outside=[c for c in HABITS if not df[c].min()<=scenario_saved[c]<=df[c].max()]
                    if outside:st.warning('Explored values outside training-data ranges: '+', '.join(outside))
            st.divider()
            st.subheader('How sensitive is the model to one habit?')
            selected=st.selectbox('Vary only this habit',HABITS,format_func=lambda c:HABIT_NAMES[c],key='sensitivity_feature')
            grid=np.linspace(float(df[selected].quantile(.05)),float(df[selected].quantile(.95)),35)
            sweep=pd.DataFrame([current.copy() for _ in grid],columns=FEATURES);sweep[selected]=grid
            line=pd.DataFrame({LABELS[FEATURES.index(selected)]:grid,'Predicted score':result['model'].predict(sweep)})
            fig=px.line(line,x=line.columns[0],y='Predicted score',color_discrete_sequence=[COLORS[selected]])
            fig.update_layout(height=280,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',margin=dict(l=15,r=15,t=15,b=15))
            fig.update_yaxes(range=[1,10]);st.plotly_chart(fig,width='stretch')
            st.caption('Only this input varies; other snapshot inputs stay fixed, with missing values imputed. The curve spans the dataset’s 5th–95th percentiles, not recommended health limits. Feature combinations may still be unfamiliar to the model. This is not a causal effect or a health recommendation.')
    with tabs[2]:
        st.subheader('Turn awareness into one small action')
        st.write('Pick something practical enough to repeat. You can care about your health without chasing the highest model score.')
        cols=st.columns(3)
        messages=[('🚶 Make space for movement','Some activity is better than none. Choose movement you enjoy and can manage. A step count misses cycling, swimming, strength work and activity intensity.'),('🌙 Protect time for rest','CDC recommends at least 7 hours for adults aged 18–60. Sleep quality matters too. A consistent bedtime and wake time can make rest a deliberate part of your routine.'),('💧 Notice your water routine','Log what you actually drink before choosing a goal. This app’s example water goal is not a universal prescription. Personal needs and any medical fluid limits take priority.')]
        for host,(title,body) in zip(cols,messages):
            host.markdown(f'<div class="action"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
        st.markdown('General adult guidance: [CDC — sleep](https://www.cdc.gov/sleep/about/index.html) · [WHO — physical activity](https://www.who.int/news-room/fact-sheets/detail/physical-activity). Adapt activity to your circumstances; these notes are separate from model predictions.')
        st.caption('This app is designed for adult education and reflection. It cannot personalize medical advice or assess symptoms. Persistent health concerns deserve professional assessment regardless of your score.')
        st.divider()
        chosen=st.selectbox('My focus for the next 7 days',['Movement','Rest & recovery','Hydration awareness','Another habit'],key='plan_focus')
        defaults={'Movement':'Take one manageable movement break during my day.','Rest & recovery':'Set aside a consistent time to wind down before bed.','Hydration awareness':'Keep an honest record of my usual water intake.','Another habit':'Choose one small routine I can repeat.'}
        with st.form('commitment_form'):
            action=st.text_input('The action I choose',value=defaults[chosen],max_chars=200,key='action_'+chosen)
            cue=st.text_input('When will I do it?',placeholder='For example: after lunch, or before getting ready for bed',max_chars=160)
            committed=st.form_submit_button('Save my commitment',type='primary')
        if committed:
            if not action.strip():st.warning('Write one specific action before saving.')
            else:st.session_state['commitment']={'Focus':chosen,'Action':action.strip(),'When':cue.strip() or 'Not specified'}
        if 'commitment' in st.session_state:
            p=st.session_state['commitment'];st.success('Your commitment is ready. Start small and revisit it honestly.')
            st.write('**Your action:** '+p['Action']);st.write('**Your cue:** '+p['When'])
            st.download_button('Download my action plan',pd.DataFrame([p]).to_csv(index=False),'my_wellbeing_plan.csv','text/csv',key='plan_download')
        st.divider()
        st.subheader('Build a record you can reflect on')
        st.caption('No invented streaks or history. Only check-ins you explicitly save appear here. Data lasts for this browser session and may clear on reload.')
        log_date=st.date_input('Date of this check-in',value=date.today(),max_value=date.today(),key='log_date')
        if st.button('Save current snapshot to my log',key='save_log'):
            current=st.session_state.get('profile')
            if current is None:st.warning('Complete a check-in first.')
            else:
                entry={'Date':str(log_date),**current,'Predicted_Fitness_Score':predict_profile(result['model'],current)}
                logs=st.session_state['checkins'];existing=[i for i,e in enumerate(logs) if e['Date']==str(log_date)]
                if existing:logs[existing[0]]=entry
                else:logs.append(entry)
                st.success('Snapshot saved. Saving the same date updates that date’s entry.')
        if st.session_state['checkins']:
            history=pd.DataFrame(st.session_state['checkins']).sort_values('Date')
            st.dataframe(history,hide_index=True,width='stretch')
            if len(history)>1:
                metric=st.selectbox('Review a habit over time',HABITS,format_func=lambda c:LABELS[FEATURES.index(c)],key='history_metric')
                st.line_chart(history.set_index('Date')[[metric]],color=COLORS[metric])
                st.caption('Self-reported saved values; gaps remain missing. This is not measured health improvement.')
            st.download_button('Download all check-ins',history.to_csv(index=False),'my_fitness_checkins.csv','text/csv',key='history_download')
            if st.button('Clear my session log',key='clear_log'):
                st.session_state['checkins']=[];st.rerun()
    with tabs[3]:
        st.subheader('A transparent model, with clear limits')
        cols=st.columns(3)
        for host,k in zip(cols,['MAE','RMSE','R2']):host.metric('Held-out '+k,f"{result['metrics'][k]:.3f}")
        st.caption('MAE and RMSE are errors in score points; lower is better. R² measures fit relative to a mean baseline and is not percentage accuracy.')
        st.dataframe(result['comparison'],hide_index=True,width='stretch')
        st.write(f"Selected: **{result['winner']}** by lowest training-only five-fold CV MAE. Training split: {result['metrics']['train_rows']} rows. Held-out test: {result['metrics']['test_rows']} rows. Mean-only baseline test MAE: {result['metrics']['Baseline MAE']:.3f}.")
        fig=px.scatter(result['test'],x='Actual',y='Predicted',color_discrete_sequence=['#42785a'],opacity=.6)
        fig.add_shape(type='line',x0=1,y0=1,x1=10,y1=10,line=dict(color='#b28b43',dash='dash'))
        fig.update_layout(height=310,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',title='Actual vs predicted • untouched holdout')
        st.plotly_chart(fig,width='stretch')
        with st.expander('Preprocessing and fair model comparison'):
            st.markdown('''- Numeric conversion and basic impossible-value checks; preserve unusual but valid observations.
- Remove exact duplicates and rows with missing/invalid target scores. Never impute targets.
- Median imputation for predictors inside each training pipeline. Linear Regression also uses standard scaling; tree models do not need it.
- Split 80/20 before learning preprocessing; compare and tune models using five training-only CV folds, random seed 42.
- Evaluate the winner once on the untouched holdout; refit its pipeline on all labelled rows for app predictions.
- Unknown profile fields use the final pipeline’s learned medians. They are excluded from personal-goal comparisons.
''')
            st.write('Winning parameters:',result['params'] or 'Default Linear Regression parameters')
            st.caption('Displayed holdout metrics describe the train-only model, not an independent evaluation of the final all-data refit.')
        with st.expander('Dataset quality and observed ranges'):
            st.write(f"{audit['raw_rows']} source rows; {audit['usable_rows']} usable rows; {audit['removed_rows']} removed rows.")
            st.dataframe(pd.DataFrame({'Missing or invalid':audit['missing_or_invalid']}),width='stretch')
            st.dataframe(df.describe().T,width='stretch')
            st.dataframe(df.head(15),hide_index=True,width='stretch')
        st.warning('The dataset’s origin, score formula, population coverage and clinical validation were not supplied. The model estimates that supplied score. It cannot certify health, diagnose illness or demonstrate that changing a habit will cause a health improvement.')
    st.divider()
    st.caption('🌱 Look after the person behind the score. Reflect on your habits, make realistic choices, and seek appropriate care for health concerns.')


if __name__=='__main__':
    if '--evaluate' in sys.argv:
        df,audit=clean_data(pd.read_csv(DATA));r=train_models(df)
        print(r['comparison'].to_string(index=False))
        print(json.dumps({'winner':r['winner'],'metrics':r['metrics'],'params':r['params'],'audit':audit},indent=2))
    else:
        main()
