"""Separate exposure-index benchmark and source-backed role lookup."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from fetch_exposure_sources import fetch_sources

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'external_exposure'
RAW = ROOT / 'data' / 'external'

# Analyst-reviewed proposals. No row labels, salary, location or model predictions
# were used to choose these codes. Exact role details are absent from the CSV.
ROLE_MAP = {
    'HR Manager': ('role_equivalent', ['11-3121.00'], 'Human Resources Managers: common abbreviation.'),
    'Cybersecurity Analyst': ('role_equivalent', ['15-1122.00'], 'Information Security Analysts: plausible role equivalent.'),
    'Sales Manager': ('role_equivalent', ['11-2022.00'], 'Sales Managers: singular/plural match.'),
    'Marketing Specialist': ('role_equivalent', ['13-1161.00'], 'Source occupation explicitly includes marketing specialists.'),
    'Operations Manager': ('role_equivalent', ['11-1021.00'], 'General and Operations Managers: broad role equivalent.'),
    'Software Engineer': ('candidate_range', ['15-1132.00', '15-1133.00'], 'Application and systems developers; specialization is unspecified.'),
    'AI Researcher': ('provisional_proxy', ['15-1111.00'], 'Computer research scientist is a broad proxy, not an AI-specific occupation.'),
    'Data Scientist': ('historical_alias_proxy', ['15-1111.00'], 'Data Scientist occurs as a 2020 alternate title of computer research scientist; modern role scope differs.'),
    'UX Designer': ('ambiguous_candidates', ['15-1134.00', '15-1199.01'], 'UX alternate titles occur under web developers and software QA; duties are needed to resolve this.'),
    'Product Manager': ('unresolved', ['11-2021.00', '13-1022.00', '41-9031.00'], 'Product Manager is an alternate title in multiple occupations; do not assign a single exposure score.'),
}


def load_sources():
    fetch_sources()  # Cached sources are verified by hash before use.
    score = pd.read_excel(RAW / 'Language Modeling AIOE and AIIE.xlsx', sheet_name='LM AIOE')
    score = score.rename(columns={'SOC Code': 'soc_code', 'Occupation Title': 'occupation',
                                 'Language Modeling AIOE': 'exposure_score'})
    assert score.soc_code.is_unique and score.exposure_score.notna().all()
    score['exposure_percentile'] = score.exposure_score.rank(method='average', pct=True) * 100
    occupations = pd.read_csv(RAW / 'Occupation Data.txt', sep='\t')
    tasks = pd.read_csv(RAW / 'Task Statements.txt', sep='\t')
    skills = pd.read_csv(RAW / 'Skills.txt', sep='\t')
    usable = skills.loc[(skills['Recommend Suppress'] != 'Y') & skills['Scale ID'].isin(['IM', 'LV'])].copy()
    usable['soc_code'] = usable['O*NET-SOC Code'].str[:7]
    usable['feature'] = usable['Element Name'] + ' [' + usable['Scale ID'] + ']'
    # Equal-weight mean across O*NET specialties within a six-digit SOC. This is
    # a documented aggregation, not a count of independent occupational samples.
    wide = usable.pivot_table(index='soc_code', columns='feature', values='Data Value', aggfunc='mean')
    merged = score.merge(wide.reset_index(), on='soc_code', how='inner', validate='one_to_one')
    return score, occupations, tasks, skills, wide, merged


def build_role_lookup(score, occupations, tasks, skills):
    candidates, roles, context = [], [], []
    for role, (status, codes, reason) in ROLE_MAP.items():
        role_scores = []
        for code in codes:
            occ = occupations.loc[occupations['O*NET-SOC Code'] == code]
            source = score.loc[score.soc_code == code[:7]]
            assert len(occ) == len(source) == 1, (role, code)
            source = source.iloc[0]
            candidates.append({'Job_Title': role, 'mapping_status': status, 'onet_code': code,
                               'onet_title': occ.iloc[0].Title, 'soc_code': code[:7],
                               'source_occupation': source.occupation,
                               'exposure_score': source.exposure_score,
                               'exposure_percentile': source.exposure_percentile, 'reason': reason})
            role_scores.append(source)
            for _, t in tasks.loc[tasks['O*NET-SOC Code'] == code].iterrows():
                context.append({'Job_Title': role, 'onet_code': code, 'kind': 'task',
                                'name': t.Task, 'importance': np.nan, 'date': t.Date})
            s = skills.loc[(skills['O*NET-SOC Code'] == code) & (skills['Scale ID'] == 'IM') &
                           (skills['Recommend Suppress'] != 'Y')].nlargest(8, 'Data Value')
            for _, t in s.iterrows():
                context.append({'Job_Title': role, 'onet_code': code, 'kind': 'skill',
                                'name': t['Element Name'], 'importance': t['Data Value'], 'date': t.Date})
        accepted = status in ['role_equivalent', 'candidate_range']
        roles.append({'Job_Title': role, 'mapping_status': status,
                      'supported_lookup': accepted, 'candidate_count': len(codes),
                      'exposure_score_low': min(x.exposure_score for x in role_scores) if accepted else np.nan,
                      'exposure_score_high': max(x.exposure_score for x in role_scores) if accepted else np.nan,
                      'exposure_percentile_low': min(x.exposure_percentile for x in role_scores) if accepted else np.nan,
                      'exposure_percentile_high': max(x.exposure_percentile for x in role_scores) if accepted else np.nan,
                      'reason': reason})
    return pd.DataFrame(roles), pd.DataFrame(candidates), pd.DataFrame(context)


def get_role_exposure(job_title):
    """Read dashboard-ready evidence, with unresolved titles left unscored."""
    if not isinstance(job_title, str) or not job_title.strip():
        raise ValueError('Provide a nonempty job title.')
    roles = pd.read_csv(OUT / 'role_exposure_lookup.csv')
    row = roles.loc[roles.Job_Title.str.casefold() == job_title.strip().casefold()]
    if row.empty:
        return {'Job_Title': job_title, 'mapping_status': 'unknown_title', 'supported_lookup': False,
                'exposure_score_low': None, 'exposure_score_high': None,
                'exposure_percentile_low': None, 'exposure_percentile_high': None,
                'reason': 'No reviewed occupation mapping is available.', 'candidates': [], 'tasks_and_skills': []}
    result = json.loads(row.to_json(orient='records'))[0]
    for filename, key in [('role_mapping_candidates.csv', 'candidates'), ('role_tasks_and_skills.csv', 'tasks_and_skills')]:
        data = pd.read_csv(OUT / filename)
        result[key] = json.loads(data.loc[data.Job_Title == result['Job_Title']].to_json(orient='records'))
    result['measure'] = 'Historical language-model occupational exposure'
    result['percentile_reference'] = '774 US occupations; percentile is a relative rank, not job-loss probability'
    return result


def benchmark(merged, features):
    X = merged.loc[:, features]
    y = merged.exposure_score.to_numpy()
    groups = merged.soc_code.str[:2].to_numpy()
    def scaled(model):
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
    models = {
        'Mean baseline': make_pipeline(SimpleImputer(strategy='median'), DummyRegressor(strategy='mean')),
        'Ridge skills': scaled(Ridge(alpha=10)),
        'Extra trees skills': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=2)),
        'MLP skills (3-seed ensemble)': VotingRegressor([(f'seed_{seed}', scaled(MLPRegressor(
            hidden_layer_sizes=(32,), alpha=0.1, solver='adam', max_iter=1500,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=40,
            random_state=seed))) for seed in [42, 7, 123]], n_jobs=1),
    }
    splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    fold_records, warning_records, predictions = [], [], []
    split_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        assert not set(groups[train]) & set(groups[test])
        split_rows.extend({'soc_code': merged.iloc[i].soc_code, 'major_group': groups[i], 'fold': fold} for i in test)
    for name, template in models.items():
        print('Benchmarking', name, flush=True)
        for fold, (train, test) in enumerate(splits, 1):
            estimator = clone(template)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                estimator.fit(X.iloc[train], y[train])
            warning_records.extend({'model': name, 'fold': fold, 'category': w.category.__name__,
                                    'message': str(w.message)} for w in caught)
            pred = estimator.predict(X.iloc[test])
            fold_records.append({'model': name, 'fold': fold, 'n_test': len(test),
                                 'mae': mean_absolute_error(y[test], pred),
                                 'rmse': root_mean_squared_error(y[test], pred),
                                 'r2': r2_score(y[test], pred),
                                 'train_r2': r2_score(y[train], estimator.predict(X.iloc[train]))})
            predictions.extend({'model': name, 'soc_code': merged.iloc[i].soc_code,
                                'occupation': merged.iloc[i].occupation, 'major_group': groups[i],
                                'fold': fold, 'actual_exposure': y[i], 'predicted_exposure': p}
                               for i, p in zip(test, pred))
    folds, pred = pd.DataFrame(fold_records), pd.DataFrame(predictions)
    comparison = []
    for name, g in pred.groupby('model', sort=False):
        comparison.append({'model': name, 'oof_mae': mean_absolute_error(g.actual_exposure, g.predicted_exposure),
                           'oof_rmse': root_mean_squared_error(g.actual_exposure, g.predicted_exposure),
                           'oof_r2': r2_score(g.actual_exposure, g.predicted_exposure),
                           'fold_mae_sd': folds.loc[folds.model == name, 'mae'].std()})
    comparison = pd.DataFrame(comparison).sort_values('oof_mae')
    winner = comparison.loc[comparison.model != 'Mean baseline'].iloc[0].model
    estimator = clone(models[winner]).fit(X, y)
    joblib.dump(estimator, OUT / 'exposure_index_model.joblib')
    pd.DataFrame(split_rows).to_csv(OUT / 'occupation_family_split.csv', index=False)
    folds.to_csv(OUT / 'cv_fold_metrics.csv', index=False)
    pred.to_csv(OUT / 'out_of_fold_predictions.csv', index=False)
    pd.DataFrame(warning_records, columns=['model', 'fold', 'category', 'message']).to_csv(OUT / 'fit_warnings.csv', index=False)
    comparison.to_csv(OUT / 'model_comparison.csv', index=False)
    return comparison, winner, len(set(groups))


def run_analysis():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / 'ai_job_market_insights.csv', ROOT / 'artifacts' / 'automation_risk_pipeline.joblib',
                 ROOT / 'artifacts' / 'model_metadata.json', ROOT / 'artifacts' / 'test_predictions.csv']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    score, occupations, tasks, skills, wide, merged = load_sources()
    features = sorted(wide.columns)
    score.to_csv(OUT / 'occupation_exposure_lookup.csv', index=False)
    merged.to_csv(OUT / 'training_occupations.csv', index=False)
    score.loc[~score.soc_code.isin(merged.soc_code)].to_csv(OUT / 'occupations_without_skill_features.csv', index=False)
    comparison, winner, n_groups = benchmark(merged, features)
    roles, candidates, context = build_role_lookup(score, occupations, tasks, skills)
    roles.to_csv(OUT / 'role_exposure_lookup.csv', index=False)
    candidates.to_csv(OUT / 'role_mapping_candidates.csv', index=False)
    context.to_csv(OUT / 'role_tasks_and_skills.csv', index=False)
    profiles = pd.read_csv(ROOT / 'ai_job_market_insights.csv')
    assert set(profiles.Job_Title) == set(ROLE_MAP)
    # Profile export joins only inputs. The original target is deliberately absent.
    enriched = profiles.drop(columns=['Automation_Risk']).assign(source_row=np.arange(len(profiles)))
    enriched = enriched.merge(roles, on='Job_Title', how='left', validate='many_to_one')
    enriched.to_csv(OUT / 'profile_exposure_context.csv', index=False)
    # Compatibility audit uses only the previously defined development rows.
    split = json.loads((ROOT / 'artifacts' / 'split_manifest.json').read_text(encoding='utf-8'))
    dev = profiles.iloc[split['development_row_indices']].merge(roles, on='Job_Title', validate='many_to_one')
    accepted = dev.loc[dev.supported_lookup].copy()
    accepted['exposure_midpoint'] = (accepted.exposure_score_low + accepted.exposure_score_high) / 2
    ordinal = accepted.Automation_Risk.map({'Low': 0, 'Medium': 1, 'High': 2})
    rho = float(spearmanr(accepted.exposure_midpoint, ordinal).statistic)
    role_labels = pd.crosstab(dev.Job_Title, dev.Automation_Risk).reindex(columns=['Low', 'Medium', 'High'], fill_value=0)
    role_labels.to_csv(OUT / 'development_label_counts_by_role.csv')
    metadata = {'target': 'Felten–Raj–Seamans language-model occupational exposure index (2023)',
                'target_is_job_loss_probability': False, 'percentile_is_probability': False,
                'source_occupations': len(score), 'training_occupations': len(merged),
                'skill_features': features, 'occupation_major_groups': n_groups,
                'evaluation': '5-fold GroupKFold by two-digit SOC major group. Pooled out-of-fold metrics; fixed configurations, exploratory model comparison.',
                'selected_model': winner, 'selection_metric': 'pooled out-of-fold MAE',
                'independent_post_selection_test': False, 'refit_on_all_external_occupations': True,
                'feature_aggregation': 'Mean nonsuppressed skill importance/level across specialties of each 2010 SOC occupation.',
                'target_construction_overlap': 'AIOE itself uses O*NET occupational abilities. Skills are different but related descriptors; predicting this constructed index does not validate automation outcomes.',
                'onet_version': '24.2 (February 2020), historical SOC compatibility; not a contemporary exposure forecast',
                'geographic_scope': 'US occupation definitions; worldwide challenge locations are not validated',
                'supported_role_titles': int(roles.supported_lookup.sum()),
                'supported_profile_rows': int(enriched.supported_lookup.sum()),
                'compatibility_audit': {'split': 'original development rows only', 'accepted_mapping_rows': len(accepted),
                                        'spearman_rho': rho, 'inferential_test': None,
                                        'note': 'Descriptive only: score varies by six job titles, not by independent worker exposures. No thresholds fitted or original test labels used in the audit or any new evaluation.'},
                'fine_tuning_performed': False,
                'fine_tuning_decision': 'Target definitions are incompatible/unverified; original labels have no documented provenance. Role-level exposure is deterministic by job title and adds no within-role worker information.',
                'protected_files_sha256': before}
    (OUT / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)]
    return metadata, comparison, roles


if __name__ == '__main__':
    meta, comparison, roles = run_analysis()
    print(comparison.to_string(index=False))
    print(roles[['Job_Title', 'mapping_status', 'supported_lookup', 'exposure_percentile_low', 'exposure_percentile_high']].to_string(index=False))
    print(json.dumps({k: meta[k] for k in ['source_occupations', 'training_occupations', 'supported_profile_rows', 'compatibility_audit']}, indent=2))
