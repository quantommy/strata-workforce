"""Cohort, model and occupational helpers shared by the Streamlit dashboard."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modeling import INPUT_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parent
RISK_ORDER = ['Low','Medium','High']
RISK_COLORS = {'Low':'#27B39D','Medium':'#F2B44B','High':'#E96B82'}
LABELS = {'Job_Title':'Job role','Industry':'Industry','Company_Size':'Company size','Location':'Location',
          'AI_Adoption_Level':'AI adoption','Required_Skills':'Required skill','Salary_USD':'Listed salary (USD)',
          'Remote_Friendly':'Remote-friendly','Job_Growth_Projection':'Job outlook'}

DEPARTMENTS = {'Data Scientist':'Data & research','AI Researcher':'Data & research',
    'Software Engineer':'Engineering','Cybersecurity Analyst':'Security',
    'UX Designer':'Product & design','Product Manager':'Product & design',
    'HR Manager':'People','Marketing Specialist':'Marketing','Sales Manager':'Sales',
    'Operations Manager':'Operations'}
# Authored exploration checklists, deliberately separate from model features.
ROLE_SKILLS = {
    'Data Scientist':['Python','Data Analysis','Machine Learning'],
    'AI Researcher':['Machine Learning','Python','Data Analysis'],
    'Software Engineer':['Python','JavaScript','Communication'],
    'Cybersecurity Analyst':['Cybersecurity','Python','Data Analysis'],
    'UX Designer':['UX/UI Design','Communication','Data Analysis'],
    'Product Manager':['Project Management','Communication','Data Analysis'],
    'HR Manager':['Communication','Project Management'],
    'Marketing Specialist':['Marketing','Communication','Data Analysis'],
    'Sales Manager':['Sales','Communication','Project Management'],
    'Operations Manager':['Project Management','Data Analysis','Communication'],
}


def load_bundle():
    folder = ROOT
    metadata=json.loads((folder/'model_metadata.json').read_text(encoding='utf-8'))
    assert metadata['pretrained_weights_used'] is False
    assert hashlib.sha256((folder/'enriched_risk_model.joblib').read_bytes()).hexdigest()==metadata['model_sha256']
    assert hashlib.sha256((ROOT/'ai_job_market_insights.csv').read_bytes()).hexdigest()==metadata['source_sha256']
    return joblib.load(folder/'enriched_risk_model.joblib'),metadata


def load_profiles(model):
    data=pd.read_csv(ROOT/'ai_job_market_insights.csv')
    data['Department']=data.Job_Title.map(DEPARTMENTS)
    return score_profiles(model,data)


def score_profiles(model,data):
    data=data.copy()
    with threadpool_limits(limits=2): scores=model.predict_proba(data[INPUT_COLUMNS])
    data['MLP_Estimate']=model.classes_[scores.argmax(axis=1)]
    for i,label in enumerate(model.classes_): data[f'Score_{label}']=scores[:,i]
    data['Source_Row']=np.arange(len(data))
    data['Profile_ID']=[f'P{i+1:03d}' for i in range(len(data))]
    return data


def company_profiles(model,frame):
    """Validate uploaded employee rows before they reach inference or analytics."""
    missing=set(INPUT_COLUMNS+['Employee_ID','Department'])-set(frame.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if frame.empty: raise ValueError('The file has no employee rows.')
    if len(frame)>10000: raise ValueError('Please use up to 10,000 employee rows per file.')
    frame=frame.copy()
    required=INPUT_COLUMNS+['Employee_ID','Department']
    if frame[required].isna().any().any(): raise ValueError('Required fields contain missing values.')
    for col in required:
        if col!='Salary_USD': frame[col]=frame[col].astype(str).str.strip()
    if (frame[['Employee_ID','Department']]=='').any().any(): raise ValueError('Employee IDs and departments must be filled in.')
    if frame.Employee_ID.duplicated().any(): raise ValueError('Employee_ID must be unique: one row per employee.')
    reference=pd.read_csv(ROOT/'ai_job_market_insights.csv')
    for col in INPUT_COLUMNS:
        if col=='Salary_USD': continue
        unknown=set(frame[col])-set(reference[col])
        if unknown: raise ValueError(f'{col} has unsupported categories: '+', '.join(sorted(unknown)))
    frame['Salary_USD']=pd.to_numeric(frame.Salary_USD,errors='coerce')
    if not np.isfinite(frame.Salary_USD).all() or (frame.Salary_USD<=0).any():
        raise ValueError('Salary_USD must contain positive, finite numbers.')
    if 'Automation_Risk' in frame and not frame.Automation_Risk.isin(RISK_ORDER).all():
        raise ValueError('Optional Automation_Risk labels must be Low, Medium or High.')
    return score_profiles(model,frame)


def department_summary(view,flagged=('High',)):
    work=view.assign(Flagged=view.View_Risk.isin(flagged),High=view.View_Risk.eq('High'))
    result=work.groupby('Department').agg(Profiles=('Profile_ID','size'),
        Flagged=('Flagged','sum'),High=('High','sum'),Median_salary=('Salary_USD','median'))
    result['Flagged_share']=result.Flagged/result.Profiles
    return result.reset_index().sort_values(['Flagged','Flagged_share'],ascending=False)


def career_options(reference,inputs,known_skills,weights):
    """Transparent preference ranking; no inferred probabilities or outcomes."""
    weights={name:max(0.,float(weights.get(name,0))) for name in ['Salary','Skill fit','Outlook','Remote']}
    total=sum(weights.values())
    if not total: raise ValueError('Give at least one factor a weight above zero.')
    known=set(known_skills)
    rows=[]
    for role,checklist in ROLE_SKILLS.items():
        if role==inputs['Job_Title']: continue
        group=reference.loc[reference.Job_Title==role]
        local=group.loc[(group.Industry==inputs['Industry']) & (group.Location==inputs['Location'])]
        matched=len(local)>=5
        context=local if matched else group
        salary=float(context.Salary_USD.median())
        rows.append({'Role':role,'Median salary':salary,'Salary change':salary-float(inputs['Salary_USD']),
            'Skill fit':100*len(known.intersection(checklist))/len(checklist),
            'Outlook':100*((context.Job_Growth_Projection=='Growth').mean()+.5*(context.Job_Growth_Projection=='Stable').mean()),
            'Remote':100*(context.Remote_Friendly=='Yes').mean(),
            'Growth share':100*(context.Job_Growth_Projection=='Growth').mean(),
            'Evidence rows':len(context),'Salary scope':'Same industry & city' if matched else 'All demo profiles for role',
            'Skills to build':[s for s in checklist if s not in known]})
    result=pd.DataFrame(rows)
    minimum,maximum=result['Median salary'].min(),result['Median salary'].max()
    result['Salary']=100*(result['Median salary']-minimum)/(maximum-minimum) if maximum>minimum else 50.
    for factor,weight in weights.items(): result[f'{factor} contribution']=result[factor]*weight/total
    result['Score']=sum(result[f'{factor} contribution'] for factor in weights)
    return result.sort_values(['Score','Role'],ascending=[False,True]).reset_index(drop=True)


def filter_profiles(data, filters, focus=(), basis='Dataset labels', search=''):
    risk_field=TARGET if basis=='Dataset labels' else 'MLP_Estimate'
    output=data.copy()
    for field in ['Industry','Job_Title','Location','Company_Size','AI_Adoption_Level','Job_Growth_Projection']:
        values=filters.get(field,[])
        if values: output=output.loc[output[field].isin(values)]
    if filters.get('Risk'): output=output.loc[output[risk_field].isin(filters['Risk'])]
    remote=filters.get('Remote_Friendly','Any')
    if remote!='Any': output=output.loc[output.Remote_Friendly==remote]
    salary=filters.get('Salary_USD')
    if salary: output=output.loc[output.Salary_USD.between(*salary)]
    if focus: output=output.loc[output.Industry.isin(focus)]
    if search.strip():
        columns=['Profile_ID','Job_Title','Industry','Location','Required_Skills']
        text=output[columns].fillna('').astype(str).agg(' '.join,axis=1)
        output=output.loc[text.str.contains(search.strip(),case=False,regex=False)]
    return output.assign(View_Risk=output[risk_field])


def risk_counts(data):
    return data.View_Risk.value_counts().reindex(RISK_ORDER,fill_value=0)


def cohort_metrics(data):
    n=len(data)
    return {'profiles':n,'high_count':int((data.View_Risk=='High').sum()),
            'high_share':float((data.View_Risk=='High').mean()) if n else 0,
            'median_salary':float(data.Salary_USD.median()) if n else 0,
            'remote_share':float((data.Remote_Friendly=='Yes').mean()) if n else 0}


def predict_profile(model, inputs):
    with threadpool_limits(limits=2): scores=model.predict_proba(pd.DataFrame([inputs],columns=INPUT_COLUMNS))[0]
    return {'inputs':inputs,'label':str(model.classes_[scores.argmax()]),
            'scores':{str(label):float(score) for label,score in zip(model.classes_,scores)}}


def pathway_candidates(model, role):
    if role not in model.role_vectors: return []
    features=list(model.skill_scaler.feature_names_in_)
    idx=[i for i,f in enumerate(features) if f.endswith(' [IM]')]
    source=model.role_vectors[role][idx]
    result=[]
    for candidate,vector in model.role_vectors.items():
        if candidate==role: continue
        target=vector[idx]
        similarity=float(source@target/(np.linalg.norm(source)*np.linalg.norm(target)))
        priorities=sorted([{'skill':features[i].replace(' [IM]',''),'importance':float(vector[i]),
                            'difference':float(vector[i]-model.role_vectors[role][i])} for i in idx],
                          key=lambda x:x['difference'],reverse=True)
        result.append({'role':candidate,'similarity':similarity,'priorities':priorities})
    return sorted(result,key=lambda x:x['similarity'],reverse=True)


def csv_bytes(data):
    return data.drop(columns=['View_Risk'],errors='ignore').to_csv(index=False).encode('utf-8-sig')
