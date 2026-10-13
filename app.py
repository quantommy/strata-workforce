"""Strata: company risk analytics and a connected career workspace."""
import html
import json
import io
import hashlib
import pandas as pd
import plotly.graph_objects as go
import altair as alt
import streamlit as st
from dashboard_data import (ROOT, RISK_ORDER, RISK_COLORS, LABELS, ROLE_SKILLS,
    load_bundle, load_profiles, company_profiles, filter_profiles, risk_counts,
    predict_profile, department_summary, career_options, csv_bytes, pathway_candidates)
from modeling import INPUT_COLUMNS
from learning_plans import PLANS

st.set_page_config(page_title='Strata · Workforce intelligence',
    page_icon=str(ROOT/'strata-icon.svg'),layout='wide',initial_sidebar_state='auto')
st.html('<style>'+(ROOT/'dashboard.css').read_text(encoding='utf-8')+'</style>')

@st.cache_resource
def bundle(): return load_bundle()

@st.cache_data
def profiles(): return load_profiles(bundle()[0])

@st.cache_data
def upload_profiles(raw): return company_profiles(bundle()[0],pd.read_csv(io.BytesIO(raw),dtype={'Employee_ID':str}))

model,metadata=bundle()
demo=profiles()
PAGES=['Company overview','Career workspace']
if st.session_state.get('page') not in PAGES: st.session_state.page=PAGES[0]
st.session_state.setdefault('department_focus',[])
st.session_state.setdefault('chart_revision',0)

def navigate(page): st.session_state.page=page
def clear_focus():
    st.session_state.department_focus=[]
    st.session_state.chart_revision+=1
def reset_filters():
    for field in ['Industry','Location','Company_Size','Job_Title']:
        st.session_state['filter_'+field]=[]
    st.session_state.saved_scope={}
    clear_focus()
def use_demo():
    st.session_state.pop('company_roster_bytes',None)
    st.session_state.upload_revision=st.session_state.get('upload_revision',0)+1
    clear_focus()
def load_profile(record):
    st.session_state.page='Career workspace'
    for field in INPUT_COLUMNS: st.session_state['lab_'+field]=record[field]
    st.session_state.known_skills=[record['Required_Skills']]
    st.session_state.submitted_skills=[record['Required_Skills']]
    st.session_state.profile_result=predict_profile(model,{f:record[f] for f in INPUT_COLUMNS})
    st.session_state.profile_origin=str(record.get('Employee_ID',record['Profile_ID']))

def heading(title,subtitle,eyebrow):
    left,right=st.columns([4,1],vertical_alignment='center')
    with left:
        st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div><h1 class="page-title">{html.escape(title)}</h1><p class="page-subtitle">{html.escape(subtitle)}</p>')
    with right: st.html('<div class="top-chip"><span class="chip-dot"></span>ENRICHED MLP · LIVE</div>')
def section(title,note=''):
    st.html(f'<div class="section-title">{html.escape(title)}</div><div class="section-note">{html.escape(note)}</div>')
def metric(label,value,foot,color=''):
    st.html(f'<div class="metric-card"><div class="metric-top">{html.escape(label)}</div><div class="metric-value" style="color:{color}">{value}</div><div class="metric-foot">{html.escape(foot)}</div></div>')
def plot(fig,height=285,key=None):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI',size=11,color='#56637A'),margin=dict(l=8,r=8,t=20,b=12),
        legend=dict(orientation='h',y=1.13,x=0),hoverlabel=dict(bgcolor='#26354D',font_color='white'),
        xaxis=dict(showgrid=True,gridcolor='#D8DFE9',zeroline=False,automargin=True),
        yaxis=dict(showgrid=False,zeroline=False,automargin=True))
    st.plotly_chart(fig,width='stretch',theme=None,key=key,config={'displayModeBar':False})
def evidence_note():
    st.caption('Enriched MLP trained from scratch · validation macro F1 0.350. Scores are uncalibrated class scores, not probabilities of job loss. Department totals describe profiles assigned a risk class.')
    with st.expander('Evidence, data and model details'):
        st.write('The supplied 500-row dataset contains job profiles, no employer or employee identifiers, and no department field. Demo department groups are derived from job titles. The risk labels have no documented measurement method. Charts do not measure observed job losses.')
        st.write('The model uses all nine original inputs plus title-derived occupational descriptors for six supported roles. Four ambiguous titles have this branch masked. Its three randomly initialized models were fitted on 400 development records; the 100 reserved records were excluded from new fitting. Displaying estimates across the dataset includes training records.')
        st.write('Career scores are adjustable exploration heuristics. Skill checklists and learning steps are authored suggestions. Salary, outlook and remote context comes from the small demo dataset, not current market research; its salary pay period is unspecified.')
        st.markdown('Occupational enrichment: [AIOE source](https://github.com/AIOE-Data/AIOE) · [O*NET database archive](https://www.onetcenter.org/db_releases.html). O*NET 24.2 (2020), USDOL/ETA, CC BY 4.0; local mappings and aggregations are modifications.')
        st.download_button('Download model card',json.dumps(metadata,indent=2).encode(),'strata_model_card.json','application/json')

with st.sidebar:
    a,b=st.columns([1,3],vertical_alignment='center')
    with a: st.image(str(ROOT/'strata-icon.svg'),width=49)
    with b: st.html('<div class="brand-name">strata</div><div class="brand-note">WORKFORCE INTELLIGENCE</div>')
    st.html('<div class="rail-label">YOUR WORKSPACE</div>')
    for page,icon in zip(PAGES,[':material/donut_large:',':material/route:']):
        st.button(page,icon=icon,type='primary' if st.session_state.page==page else 'secondary',
            width='stretch',on_click=navigate,args=(page,),key='nav_'+page)
    st.html('<div class="rail-note"><span class="chip-dot"></span>Connected insights<br><small>Understand the exposure.<br>Explore the next move.</small></div>')
    filters={}
    if st.session_state.page=='Company overview':
        st.html('<div class="rail-label">SCOPE YOUR COMPANY VIEW</div>')
        # Upload-specific categories are validated against the same input schema.
        for field in ['Industry','Location','Company_Size']:
            st.session_state.setdefault('filter_'+field,st.session_state.get('saved_scope',{}).get(field,[]))
            filters[field]=st.multiselect(LABELS[field],sorted(demo[field].unique()),placeholder='All',key='filter_'+field)
        with st.expander('Role filter'):
            st.session_state.setdefault('filter_Job_Title',st.session_state.get('saved_scope',{}).get('Job_Title',[]))
            filters['Job_Title']=st.multiselect('Job role',sorted(demo.Job_Title.unique()),key='filter_Job_Title')
        st.session_state.saved_scope=filters.copy()
        st.button('Reset filters',icon=':material/restart_alt:',width='stretch',on_click=reset_filters)
    else:
        st.caption('One profile. Ranked possibilities. A practical learning plan.')
        st.html('<div class="rail-label">HOW IT WORKS</div><div class="rail-steps">01 &nbsp; Tell us where you are<br>02 &nbsp; Set what matters to you<br>03 &nbsp; Explore a path forward</div>')
    st.html('<div class="rail-footer">STRATA / TRACK A<br>Enriched MLP from scratch<br>Interactive workforce portal</div>')

def overview():
    heading('See the risk. Find the focus.','A company-wide view of exposure, department priorities and the people behind the numbers.','STRATA / COMPANY OVERVIEW')
    with st.expander('Company data · use your own employee roster'):
        st.caption('The default is a demo of 500 job profiles, not a real company. Upload one row per employee with a unique Employee_ID, actual Department, and the nine model inputs. Optional Automation_Risk labels allow a source-label view.')
        st.session_state.setdefault('company_name',st.session_state.get('saved_company_name','Your company'))
        st.text_input('Company name',key='company_name')
        st.session_state.saved_company_name=st.session_state.company_name
        uploaded=st.file_uploader('Employee roster (CSV)',type=['csv'],key='company_upload_'+str(st.session_state.get('upload_revision',0)))
        if uploaded is not None: st.session_state.company_roster_bytes=uploaded.getvalue()
        if st.session_state.get('company_roster_bytes') is not None:
            st.caption('Your uploaded roster stays active when you switch between views.')
            st.button('Return to demo data',on_click=use_demo)
        template=demo[INPUT_COLUMNS].head(3).copy()
        template.insert(0,'Department',demo.Department.head(3))
        template.insert(0,'Employee_ID',['EMP001','EMP002','EMP003'])
        st.download_button('Download example roster',template.to_csv(index=False).encode(),'strata_roster_example.csv','text/csv')
    source=demo
    actual=False
    raw=st.session_state.get('company_roster_bytes')
    if raw is not None:
        try:
            source=upload_profiles(raw)
            actual=True
        except (ValueError,TypeError,pd.errors.ParserError,UnicodeDecodeError) as error:
            st.error('Roster could not be used: '+str(error))
            return
    identity=hashlib.sha256(raw).hexdigest() if actual else 'demo'
    if st.session_state.get('cohort_identity')!=identity:
        st.session_state.cohort_identity=identity
        clear_focus()
    if actual:
        st.html(f'<div class="context-banner"><strong>{html.escape(st.session_state.company_name or "Your company")}</strong>Uploaded roster · {len(source):,} unique employees · departments supplied in your file</div>')
    else:
        st.html('<div class="context-banner"><strong>DEMO COMPANY VIEW</strong>500 job profiles · department groups derived from titles · no actual employee headcount supplied</div>')
    a,b,c=st.columns([2,2,1.5],vertical_alignment='bottom')
    options=['MLP estimates']+(['Dataset labels'] if 'Automation_Risk' in source else [])
    with a:
        # A single available basis still renders a stable selector.
        if st.session_state.get('risk_basis') not in options:
            saved=st.session_state.get('saved_risk_basis',options[0])
            st.session_state.risk_basis=saved if saved in options else options[0]
        basis=st.segmented_control('Risk basis',options,selection_mode='single',key='risk_basis') or options[0]
        st.session_state.saved_risk_basis=basis
    with b:
        flag=st.selectbox('Flag for attention',['High only','Medium + High'],key='attention')
    with c:
        if st.session_state.department_focus:
            st.button('Clear department focus',on_click=clear_focus,width='stretch')
        else: st.caption('Click a department to focus.')
    flagged=['High'] if flag=='High only' else ['Medium','High']
    base=filter_profiles(source,filters,basis=basis)
    view=base.loc[base.Department.isin(st.session_state.department_focus)] if st.session_state.department_focus else base
    if st.session_state.department_focus: st.caption('Department focus: '+', '.join(st.session_state.department_focus))
    if view.empty:
        st.info('No profiles match this scope. Clear department focus or reset filters.')
        return
    n=len(view)
    exposed=int(view.View_Risk.isin(flagged).sum())
    units='employees' if actual else 'demo profiles'
    departments=department_summary(view,flagged)
    for col,(label,value,foot,color) in zip(st.columns(4),[
        ('WORKFORCE IN VIEW',f'{n:,}',units+' in current scope',''),
        ('FLAGGED FOR ATTENTION',f'{exposed:,}',f'{exposed/n:.1%} of scope · {flag.lower()}','#B64E70'),
        ('DEPARTMENTS WITH FLAGS',str(int((departments.Flagged>0).sum())),f'of {len(departments)} departments in scope',''),
        ('LARGEST FLAGGED GROUP',str(departments.iloc[0].Flagged),str(departments.iloc[0].Department) if exposed else 'None in scope','#6C5CE7')]):
        with col: metric(label,value,foot,color)
    st.space('small')
    left,right=st.columns([1.65,1])
    with left,st.container(key='department_card'):
        section('Where should you focus?','Department distribution · click a bar to filter the whole overview.')
        counts=pd.crosstab(base.Department,base.View_Risk).reindex(columns=RISK_ORDER,fill_value=0)
        long=counts.rename_axis('Department').reset_index().melt(id_vars='Department',var_name='Risk',value_name='Profiles')
        long['Risk_order']=long.Risk.map({r:i for i,r in enumerate(RISK_ORDER)})
        selection=alt.selection_point(name='department_selector',fields=['Department'],on='click',toggle=False,clear=False)
        chart=alt.Chart(long).mark_bar(size=17,cornerRadiusEnd=4).encode(
            y=alt.Y('Department:N',sort=counts.sum(axis=1).sort_values(ascending=False).index.tolist(),title=None,axis=alt.Axis(labelLimit=160)),
            x=alt.X('Profiles:Q',stack='zero',title=units.capitalize()),
            color=alt.Color('Risk:N',scale=alt.Scale(domain=RISK_ORDER,range=[RISK_COLORS[r] for r in RISK_ORDER]),legend=alt.Legend(orient='top',title=None)),
            order='Risk_order:Q',opacity=alt.condition(selection,alt.value(1),alt.value(.28)),
            tooltip=['Department:N','Risk:N','Profiles:Q']).add_params(selection).properties(height=285).configure_view(stroke=None).configure_axis(labelColor='#56637A',titleColor='#56637A',gridColor='#D8DFE9').configure_legend(labelColor='#56637A')
        chart_key='departments_'+str(st.session_state.chart_revision)
        event=st.altair_chart(chart,width='stretch',theme=None,key=chart_key,on_select='rerun',selection_mode=['department_selector'])
        selected=event.selection.department_selector
        if selected:
            signature=(chart_key,json.dumps(selected,sort_keys=True))
            if st.session_state.get('last_department_selection')!=signature:
                st.session_state.last_department_selection=signature
                st.session_state.department_focus=[item['Department'] for item in selected]
                st.rerun()
    with right,st.container(key='risk_card'):
        section('The overall risk picture',basis+' · within the selected scope')
        counts=risk_counts(view)
        fig=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.78,sort=False,textinfo='none',
            marker=dict(colors=[RISK_COLORS[r] for r in counts.index]),hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'))
        fig.update_layout(showlegend=False,annotations=[dict(text=f'<b>{exposed/n:.0%}</b><br>flagged',x=.5,y=.5,font=dict(size=24,color='#25354D'),showarrow=False)])
        plot(fig,235)
        st.html('<div class="risk-legend">'+''.join(f'<div><i style="background:{RISK_COLORS[r]}"></i>{r}<b>{counts[r]} · {counts[r]/n:.0%}</b></div>' for r in RISK_ORDER)+'</div>')
    st.space('small')
    with st.container(key='priority_card'):
        section('Department priorities','Compare scale and concentration. Flags are assigned classes, not measured automation events.')
        table=departments.rename(columns={'Profiles':units.capitalize(),'Flagged':'Flagged','High':'High class','Flagged_share':'Flagged share','Median_salary':'Median listed salary'})
        st.dataframe(table,hide_index=True,width='stretch',column_config={
            'Flagged share':st.column_config.ProgressColumn('Flagged share',min_value=0,max_value=1,format='percent'),
            'Median listed salary':st.column_config.NumberColumn(format='$%.0f')})
    st.space('small')
    left,right=st.columns(2)
    with left,st.container(key='role_card'):
        section('Risk within each role','Share of each role assigned to each class · hover for counts.')
        counts=pd.crosstab(view.Job_Title,view.View_Risk).reindex(columns=RISK_ORDER,fill_value=0)
        fig=go.Figure()
        for risk in RISK_ORDER:
            fig.add_bar(y=counts.index,x=counts[risk]/counts.sum(axis=1)*100,name=risk,orientation='h',
                marker_color=RISK_COLORS[risk],customdata=counts[risk],hovertemplate='%{y}<br>'+risk+': %{customdata} profiles · %{x:.1f}%<extra></extra>')
        fig.update_layout(barmode='stack');fig.update_xaxes(range=[0,100],ticksuffix='%')
        plot(fig,330)
    with right,st.container(key='location_card'):
        section('Where are the flags concentrated?','Flagged counts by location in this scope.')
        locations=view.assign(Flagged=view.View_Risk.isin(flagged)).groupby('Location').agg(Total=('Profile_ID','size'),Flagged=('Flagged','sum')).sort_values('Flagged')
        fig=go.Figure(go.Bar(x=locations.Flagged,y=locations.index,orientation='h',marker_color='#897AE9',
            customdata=locations.Total,hovertemplate='%{y}<br>%{x} flagged / %{customdata} profiles<extra></extra>'))
        plot(fig,330)
    left,right=st.columns(2)
    with left,st.container(key='adoption_card'):
        section('AI adoption and risk','Counts by adoption category; this shows association.')
        grid=pd.crosstab(view.AI_Adoption_Level,view.View_Risk).reindex(index=RISK_ORDER,columns=RISK_ORDER,fill_value=0)
        fig=go.Figure(go.Heatmap(x=grid.columns,y=grid.index,z=grid.values,text=grid.values,texttemplate='%{text}',
            colorscale=[[0,'#E1E5EF'],[1,'#7767D5']],showscale=False,hovertemplate='Adoption %{y}<br>Risk %{x}: %{z}<extra></extra>'))
        fig.update_xaxes(title='Risk class');fig.update_yaxes(title='AI adoption')
        plot(fig,240)
    with right,st.container(key='skills_card'):
        section('Skills represented in flagged profiles','One named skill per source profile; this is demand context.')
        skills=view.loc[view.View_Risk.isin(flagged),'Required_Skills'].value_counts().head(6).sort_values()
        if skills.empty: st.info('No profiles are flagged in this scope.')
        else:
            fig=go.Figure(go.Bar(x=skills.values,y=skills.index,orientation='h',marker_color='#56A99C',hovertemplate='%{y}: %{x} flagged profiles<extra></extra>'))
            plot(fig,240)
    with st.expander('Inspect profiles and export this scope'):
        search=st.text_input('Search employee ID, role, location or skill',key='profile_search')
        records=filter_profiles(view,{},basis=basis,search=search)
        if actual and search.strip():
            # Employee IDs are additional to the common profile search fields.
            records=view.loc[view.Employee_ID.astype(str).str.contains(search.strip(),case=False,regex=False) | view.Profile_ID.isin(records.Profile_ID)]
        columns=(['Employee_ID'] if actual else ['Profile_ID'])+['Department','Job_Title','Industry','Location','Salary_USD','View_Risk','MLP_Estimate','Required_Skills']
        if records.empty: st.info('No profiles match this search.')
        else:
            event=st.dataframe(records[columns].rename(columns={'View_Risk':'Selected risk basis'}),hide_index=True,
                width='stretch',on_select='rerun',selection_mode='single-row',key='roster_'+identity,
                column_config={'Salary_USD':st.column_config.NumberColumn(format='$%.0f')})
            if event.selection.rows:
                record=records.iloc[event.selection.rows[0]].to_dict()
                st.button('Explore this profile in Career workspace',icon=':material/route:',on_click=load_profile,args=(record,))
        st.download_button('Export filtered workforce',csv_bytes(records),'strata_workforce.csv','text/csv')
    evidence_note()

DEFAULT={'Job_Title':'Software Engineer','Industry':'Technology','Company_Size':'Medium','Location':'London',
    'AI_Adoption_Level':'Medium','Required_Skills':'Python','Salary_USD':92000.,'Remote_Friendly':'Yes','Job_Growth_Projection':'Growth'}

def career():
    heading('Your next chapter, weighted your way.','Your current profile, career options and learning steps — connected in one place.','STRATA / CAREER WORKSPACE')
    last_inputs=st.session_state.get('profile_result',{}).get('inputs',DEFAULT)
    for field,value in last_inputs.items(): st.session_state.setdefault('lab_'+field,value)
    st.session_state.setdefault('known_skills',st.session_state.get('submitted_skills',[st.session_state.lab_Required_Skills]))
    st.session_state.setdefault('submitted_skills',list(st.session_state.known_skills))
    if 'profile_result' not in st.session_state:
        st.session_state.profile_result=predict_profile(model,DEFAULT.copy())
    left,right=st.columns([1.65,1])
    with left,st.container(key='profile_form'):
        section('01 / Where are you now?','Submit your profile to update both your estimate and your options.')
        with st.form('career_profile'):
            pairs=[('Job_Title','Industry'),('Location','Company_Size'),('AI_Adoption_Level','Required_Skills'),('Remote_Friendly','Job_Growth_Projection')]
            for pair in pairs:
                for col,field in zip(st.columns(2),pair):
                    with col: st.selectbox('Primary skill' if field=='Required_Skills' else LABELS[field],sorted(demo[field].unique()),key='lab_'+field)
            st.number_input('Current listed salary (USD)',min_value=1.,max_value=100000000.,step=1000.,format='%.0f',key='lab_Salary_USD')
            st.multiselect('Skills you already have',sorted(PLANS),key='known_skills',help='Self-reported skills are used for the authored career checklist, not added as model features.')
            submitted=st.form_submit_button('Find my options',type='primary',width='stretch',icon=':material/arrow_forward:')
        if submitted:
            inputs={f:st.session_state['lab_'+f] for f in INPUT_COLUMNS}
            st.session_state.profile_result=predict_profile(model,inputs)
            st.session_state.submitted_skills=list(st.session_state.known_skills)
        st.caption('Estimates and career options use the last submitted profile. Salary values have no stated pay period in the demo dataset.')
    result=st.session_state.profile_result
    inputs=result['inputs']
    with right,st.container(key='profile_result'):
        section('Your current risk estimate','Enriched MLP · trained from scratch')
        color=RISK_COLORS[result['label']]
        st.html(f'<div class="prediction-card"><div class="eyebrow">{html.escape(inputs["Job_Title"])}</div><div class="prediction-label" style="color:{color}">{result["label"]}</div><div class="prediction-detail">Highest-scoring class · exploratory estimate</div></div>')
        for risk in RISK_ORDER:
            score=result['scores'][risk]*100
            st.html(f'<div class="score-row"><span>{risk}</span><b>{score:.1f}%</b></div><div class="score-track"><div style="width:{score}%;background:{RISK_COLORS[risk]}"></div></div>')
        st.caption('Uncalibrated class scores. These do not measure your probability of losing a job.')
        if inputs['Job_Title'] not in model.role_vectors:
            st.caption('This role has an ambiguous occupational mapping. Its external branch is masked; the original profile fields still power the estimate.')
        st.button('Save profile as comparison baseline',key='save_baseline',
            on_click=lambda:st.session_state.update(baseline_profile=st.session_state.profile_result),width='stretch')
        st.download_button('Download assessment',json.dumps({'model':metadata['name'],'job_loss_probability':False,
            'assessment':result,'self_reported_skills':st.session_state.submitted_skills},indent=2).encode(),'strata_assessment.json','application/json',width='stretch')
    baseline=st.session_state.get('baseline_profile')
    if baseline:
        with st.expander('Compare with saved profile',expanded=True):
            changes=[{'Input':LABELS[f],'Saved':str(baseline['inputs'][f]),'Now':str(inputs[f])} for f in INPUT_COLUMNS if baseline['inputs'][f]!=inputs[f]]
            if changes: st.dataframe(pd.DataFrame(changes),hide_index=True,width='stretch')
            else: st.caption('Your submitted profile matches the saved baseline.')
            delta=(result['scores']['High']-baseline['scores']['High'])*100
            st.caption(f'High-class score changed by {delta:+.1f} percentage points. This measures model sensitivity, not a causal effect.')
    st.space('small')
    with st.container(key='weights_card'):
        section('02 / What matters in your next move?','Weights are relative. Change them to reorder your options immediately.')
        weights={}
        for col,(factor,default) in zip(st.columns(4),[('Salary',35),('Skill fit',35),('Outlook',20),('Remote',10)]):
            with col:
                st.session_state.setdefault('weight_'+factor,st.session_state.get('saved_weights',{}).get(factor,default))
                weights[factor]=st.slider(factor,0,100,step=5,key='weight_'+factor)
        st.session_state.saved_weights=weights.copy()
    if sum(weights.values())==0:
        st.info('Give at least one factor a weight above zero to rank your options.')
        evidence_note()
        return
    options=career_options(demo,inputs,st.session_state.submitted_skills,weights)
    section('03 / Your ranked possibilities','Nine alternative roles · preference scores out of 100 · exploration, not a validated career forecast')
    for col,(index,row) in zip(st.columns(3),options.head(3).iterrows()):
        with col:
            missing=', '.join(row['Skills to build'][:2]) or 'All checklist skills selected'
            st.html(f'<div class="pathway"><div class="pathway-tag">#{index+1} / MATCH TO YOUR PRIORITIES</div><div class="pathway-role">{html.escape(row.Role)}</div><div class="match-score">{row.Score:.1f}<small>/100</small></div><div class="pathway-detail">Median listed salary <b>${row["Median salary"]:,.0f}</b><br>Vs your input <b>{row["Salary change"]:+,.0f} USD</b><br>Skills to explore: {html.escape(missing)}</div></div>')
    with st.container(key='ranking_card'):
        section('Compare every option','Salary is scaled across the nine options. Skill fit is overlap with an authored role checklist.')
        display=options[['Role','Score','Median salary','Salary change','Skill fit','Growth share','Remote','Evidence rows','Salary scope']].copy()
        st.dataframe(display,hide_index=True,width='stretch',column_config={
            'Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.1f'),
            'Median salary':st.column_config.NumberColumn(format='$%.0f'),
            'Salary change':st.column_config.NumberColumn(format='$%.0f'),
            **{f:st.column_config.NumberColumn(format='%.1f%%') for f in ['Skill fit','Growth share','Remote']}})
        with st.expander('How the ranking is weighted'):
            st.write('Final score = the weighted average of salary, skill fit, outlook and remote scores. Salary uses relative median salary; skill fit is the share of an authored checklist you self-report. Outlook = Growth share + half the Stable share. Remote = the share marked remote-friendly. Risk estimates are shown separately because validation performance is modest.')
            st.caption('Context uses the same role, industry and city only when at least five demo profiles match; otherwise it uses all demo profiles for that role. These small samples are illustrative. Salary differences are comparisons with your input, not promised raises.')
            st.dataframe(options[['Role']+[f+' contribution' for f in weights]],hide_index=True,width='stretch')
        st.download_button('Export ranked options',options.to_csv(index=False).encode(),'strata_career_options.csv','text/csv')
    st.space('small')
    with st.container(key='learning_plan'):
        section('04 / Make a move you can act on','Select an option to see its skill checklist and a practical workflow.')
        names=options.Role.tolist()
        if st.session_state.get('career_target') not in names: st.session_state.career_target=names[0]
        target=st.selectbox('Explore an option',names,key='career_target')
        chosen=options.loc[options.Role==target].iloc[0]
        missing=chosen['Skills to build']
        a,b=st.columns([1,1.5])
        with a:
            st.markdown('**Skill checklist · '+target+'**')
            for skill in ROLE_SKILLS[target]:
                st.write(('✓ Already selected · ' if skill in st.session_state.submitted_skills else '○ Build next · ')+skill)
            st.caption('Authored starting checklist; self-reported overlap does not establish proficiency or hiring eligibility.')
        with b:
            st.markdown('**Why this option ranks here**')
            normalized={f:round(w/sum(weights.values())*100) for f,w in weights.items()}
            st.write(f'Preference score {chosen.Score:.1f}/100. Skill overlap {chosen["Skill fit"]:.0f}%. Median listed salary ${chosen["Median salary"]:,.0f}, supported by {chosen["Evidence rows"]} demo profiles.')
            st.caption('Active weights: '+ ' · '.join(f'{f} {w}%' for f,w in normalized.items()))
        skill_options=missing+[s for s in ROLE_SKILLS[target] if s not in missing]
        target_key='plan_skill_'+target
        skill=st.selectbox('Skill to develop',skill_options,key=target_key)
        steps=PLANS[skill]
        for col,(number,title,body) in zip(st.columns(3),[
            ('01','Build the foundation',steps[0]),('02','Practice on a project',steps[1]),('03','Show your evidence',steps[2])]):
            with col: st.html(f'<div class="plan-step"><div class="number">STEP {number}</div><h4>{title}</h4><p>{html.escape(body)}</p></div>')
        content=f'# Strata career workflow\n\nFrom: {inputs["Job_Title"]}\nExplore: {target}\nSkill focus: {skill}\nPreference score: {chosen.Score:.1f}/100\n\n'+ '\n\n'.join(f'## {i+1}. {title}\n{body}' for i,(title,body) in enumerate(zip(['Foundation','Practice','Evidence'],steps)))+'\n\nAuthored exploration plan. Demo salary and outlook data are illustrative, not validated career outcomes.'
        st.download_button('Download this workflow',content.encode(),'strata_career_workflow.md','text/markdown',icon=':material/download:')
        adjacent=next((c for c in pathway_candidates(model,inputs['Job_Title']) if c['role']==target),None)
        if adjacent:
            with st.expander('Compare occupational skill emphasis'):
                priorities=pd.DataFrame(adjacent['priorities']).head(6)
                st.dataframe(priorities.rename(columns={'skill':'Broad skill','importance':'Target importance / 5','difference':'Target minus current'}),hide_index=True,width='stretch')
                st.caption('O*NET 24.2 (2020) role descriptors. This compares occupations; it does not measure personal skill gaps.')
    evidence_note()

{'Company overview':overview,'Career workspace':career}[st.session_state.page]()
st.html('<div class="footer"><span>strata / Clarity for what comes next.</span><span>Enriched MLP from scratch · Exploratory workforce intelligence</span></div>')
