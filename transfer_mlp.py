"""A compact NumPy MLP with real encoder pretraining and supervised fine-tuning.

External regression: 70 occupational skill ratings -> 32 -> 16 -> exposure.
Internal classification: the same encoder + nine original profile inputs -> 32 -> 3.
Different heads preserve the distinction between exposure and challenge risk labels.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from modeling import INPUT_COLUMNS, TARGET, make_model
from occupational_exposure import ROLE_MAP, load_sources

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'transfer_mlp'
CLASSES = np.array(['Low', 'Medium', 'High'])
ENCODER = ['W1', 'b1', 'W2', 'b2']
SEEDS = [42, 7, 123]
CONDITIONS = ['Raw MLP', 'Enriched MLP from scratch', 'Pretrained frozen encoder',
              'Pretrained fine-tuned encoder', 'Shuffled-pretraining control']
CONFIG = {'external_max_epochs': 1200, 'external_patience': 70, 'external_lr': .003,
          'internal_max_epochs': 500, 'internal_patience': 50, 'head_warmup_epochs': 20,
          'internal_head_lr': .001, 'internal_encoder_lr': .0001, 'l2': .0001,
          'early_stop_min_delta': .0001, 'optimizer': 'full-batch Adam',
          'encoder_hidden': [32, 16], 'classification_hidden': 32}


def initialize(skill_dim, raw_dim, seed, encoder=None):
    rng = np.random.default_rng(seed + 110)
    p = {'W1': rng.normal(0, np.sqrt(2/skill_dim), (skill_dim, 32)), 'b1': np.zeros(32),
         'W2': rng.normal(0, np.sqrt(2/32), (32, 16)), 'b2': np.zeros(16)}
    if encoder is not None:
        for key in ENCODER:
            assert p[key].shape == encoder[key].shape
            p[key] = encoder[key].copy()
    head_rng = np.random.default_rng(seed + 999)
    p.update({'W3': head_rng.normal(0, np.sqrt(2/(16+raw_dim+1)), (16+raw_dim+1, 32)),
              'b3': np.zeros(32), 'W4': head_rng.normal(0, np.sqrt(1/32), (32, 3)), 'b4': np.zeros(3),
              'Wr': head_rng.normal(0, np.sqrt(1/16), (16, 1)), 'br': np.zeros(1)})
    return p


def forward(p, skills, raw=None, mask=None, task='classification'):
    z1 = skills @ p['W1'] + p['b1']; h1 = np.maximum(z1, 0)
    z2 = h1 @ p['W2'] + p['b2']; h2 = np.maximum(z2, 0)
    cache = {'skills': skills, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    if task == 'regression':
        return (h2 @ p['Wr'] + p['br']).ravel(), cache
    fused = np.column_stack([h2 * mask[:, None], raw, mask])
    z3 = fused @ p['W3'] + p['b3']; h3 = np.maximum(z3, 0)
    logits = h3 @ p['W4'] + p['b4']
    shifted = logits - logits.max(axis=1, keepdims=True)
    logs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    cache.update({'mask': mask, 'fused': fused, 'z3': z3, 'h3': h3, 'logs': logs})
    return np.exp(logs), cache


def loss_and_grad(p, skills, y, raw=None, mask=None, task='classification', l2=0):
    output, c = forward(p, skills, raw, mask, task)
    n = len(y)
    if task == 'regression':
        residual = output-y
        loss = .5*np.mean(residual**2)
        derivative = residual[:, None]/n
        grad = {'Wr': c['h2'].T @ derivative, 'br': derivative.sum(axis=0)}
        dh2 = derivative @ p['Wr'].T
        weight_keys = ['W1', 'W2', 'Wr']
    else:
        loss = -np.mean(c['logs'][np.arange(n), y])
        derivative = output.copy(); derivative[np.arange(n), y] -= 1; derivative /= n
        grad = {'W4': c['h3'].T @ derivative, 'b4': derivative.sum(axis=0)}
        dz3 = (derivative @ p['W4'].T) * (c['z3'] > 0)
        grad.update({'W3': c['fused'].T @ dz3, 'b3': dz3.sum(axis=0)})
        dh2 = (dz3 @ p['W3'].T)[:, :16] * c['mask'][:, None]
        weight_keys = ['W1', 'W2', 'W3', 'W4']
    dz2 = dh2 * (c['z2'] > 0)
    grad.update({'W2': c['h1'].T @ dz2, 'b2': dz2.sum(axis=0)})
    dz1 = (dz2 @ p['W2'].T) * (c['z1'] > 0)
    grad.update({'W1': skills.T @ dz1, 'b1': dz1.sum(axis=0)})
    for key in weight_keys:
        loss += .5*l2*np.sum(p[key]**2)
        grad[key] += l2*p[key]
    return float(loss), grad


class Adam:
    def __init__(self):
        self.m, self.v, self.steps = {}, {}, {}

    def update(self, p, gradient, learning_rates):
        for key, lr in learning_rates.items():
            g = gradient[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(g); self.v[key] = np.zeros_like(g); self.steps[key] = 0
            self.steps[key] += 1; t = self.steps[key]
            self.m[key] = .9*self.m[key] + .1*g
            self.v[key] = .999*self.v[key] + .001*g*g
            p[key] -= lr*(self.m[key]/(1-.9**t))/(np.sqrt(self.v[key]/(1-.999**t))+1e-8)


def train(p, skills, y, raw=None, mask=None, *, task='classification', validation=None,
          epochs=None, freeze_encoder=False, max_epochs=None, patience=None):
    """Select an epoch using only an inner validation set, or refit a fixed budget."""
    optimizer = Adam()
    max_epochs = epochs or max_epochs or CONFIG['internal_max_epochs']
    patience = patience or CONFIG['internal_patience']
    best_loss = np.inf; best_epoch = 1; best_state = None; stale = 0; history = []
    for epoch in range(1, max_epochs+1):
        loss, gradient = loss_and_grad(p, skills, y, raw, mask, task, CONFIG['l2'])
        if task == 'regression':
            rates = {k: CONFIG['external_lr'] for k in gradient}
        else:
            rates = {k: CONFIG['internal_head_lr'] for k in ['W3', 'b3', 'W4', 'b4']}
            if not freeze_encoder and epoch > CONFIG['head_warmup_epochs']:
                rates.update({k: CONFIG['internal_encoder_lr'] for k in ENCODER})
        optimizer.update(p, gradient, rates)
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite neural-network loss.')
        if validation is not None:
            vs, vy, vr, vm = validation
            val_loss, _ = loss_and_grad(p, vs, vy, vr, vm, task, l2=0)
            history.append({'epoch': epoch, 'training_objective': loss, 'validation_loss': val_loss})
            if val_loss < best_loss-CONFIG['early_stop_min_delta']:
                best_loss = val_loss; best_epoch = epoch; best_state = copy.deepcopy(p); stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
    if validation is not None:
        return best_state, best_epoch, history
    return p, max_epochs, history


def as_dense(value):
    return value.toarray() if hasattr(value, 'toarray') else np.asarray(value, dtype=float)


def profile_skills(profiles, role_vectors, scaler):
    values = np.full((len(profiles), len(scaler.feature_names_in_)), np.nan)
    mask = np.zeros(len(profiles), dtype=float)
    for i, role in enumerate(profiles.Job_Title.fillna('').astype(str)):
        if role in role_vectors:
            values[i] = role_vectors[role]; mask[i] = 1
    values = scaler.transform(pd.DataFrame(values, columns=scaler.feature_names_in_))
    values[mask == 0] = 0
    return values, mask


def external_pretraining(merged, features):
    scaler = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(merged[features])
    X_all = scaler.transform(merged[features]); y_all = merged.exposure_score.to_numpy()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(
        merged, groups=merged.soc_code.str[:2]))
    assert not set(merged.iloc[tr].soc_code.str[:2]) & set(merged.iloc[va].soc_code.str[:2])
    diagnostic_scaler = clone(scaler).fit(merged.iloc[tr][features])
    xtr = diagnostic_scaler.transform(merged.iloc[tr][features]); xva = diagnostic_scaler.transform(merged.iloc[va][features])
    pretraining, shuffled, metrics, history_rows = {}, {}, [], []
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'soc_code': merged.soc_code, 'pretraining_diagnostic_partition': ['train' if i in set(tr) else 'validation' for i in range(len(merged))]}).to_csv(OUT/'external_diagnostic_split.csv', index=False)
    for seed in SEEDS:
        print(f'Pretraining external encoder, seed {seed}', flush=True)
        initial = initialize(len(features), 0, seed)
        selected, epoch, history = train(copy.deepcopy(initial), xtr, y_all[tr], task='regression',
            validation=(xva, y_all[va], None, None), max_epochs=CONFIG['external_max_epochs'], patience=CONFIG['external_patience'])
        prediction, _ = forward(selected, xva, task='regression')
        metrics.append({'seed': seed, 'selected_epochs': epoch, 'stopped_epoch': len(history),
                        'validation_n': len(va), 'validation_mae': mean_absolute_error(y_all[va], prediction),
                        'validation_r2': r2_score(y_all[va], prediction)})
        history_rows.extend({'seed': seed, **row} for row in history)
        full, _, _ = train(copy.deepcopy(initial), X_all, y_all, task='regression', epochs=epoch)
        shuffled_labels = np.random.default_rng(seed+7000).permutation(y_all)
        negative, _, _ = train(copy.deepcopy(initial), X_all, shuffled_labels, task='regression', epochs=epoch)
        pretraining[seed] = {k: full[k].copy() for k in ENCODER}
        shuffled[seed] = {k: negative[k].copy() for k in ENCODER}
        assert any(not np.array_equal(pretraining[seed][k], initial[k]) for k in ENCODER)
        np.savez_compressed(OUT/f'external_pretrained_seed_{seed}.npz', **full)
        np.savez_compressed(OUT/f'external_shuffled_seed_{seed}.npz', **negative)
    pd.DataFrame(metrics).to_csv(OUT/'pretraining_metrics.csv', index=False)
    pd.DataFrame(history_rows).to_csv(OUT/'pretraining_learning_curves.csv', index=False)
    joblib.dump(scaler, OUT/'external_skill_scaler.joblib')
    return scaler, pretraining, shuffled


class TransferRiskPredictor:
    """Experimental three-seed fine-tuned ensemble accepting the nine CSV inputs."""
    def __init__(self, raw_preprocessor, skill_scaler, role_vectors, parameters):
        self.raw_preprocessor = raw_preprocessor; self.skill_scaler = skill_scaler
        self.role_vectors = role_vectors; self.parameters = parameters; self.classes_ = CLASSES.copy()

    def predict_proba(self, X):
        raw = as_dense(self.raw_preprocessor.transform(X))
        skills, mask = profile_skills(X, self.role_vectors, self.skill_scaler)
        return np.mean([forward(p, skills, raw, mask)[0] for p in self.parameters], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def run_experiment():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT/'ai_job_market_insights.csv', ROOT/'artifacts/automation_risk_pipeline.joblib',
                 ROOT/'artifacts/model_metadata.json', ROOT/'artifacts/test_predictions.csv',
                 ROOT/'artifacts/external_exposure/exposure_index_model.joblib']
    protected_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    split = json.loads((ROOT/'artifacts/split_manifest.json').read_text(encoding='utf-8'))
    # Drop all reserved rows before extracting labels or fitting any new transform.
    source = pd.read_csv(ROOT/'ai_job_market_insights.csv').iloc[split['development_row_indices']].copy()
    X = source[INPUT_COLUMNS].reset_index(drop=True)
    y = source[TARGET].map(dict(zip(CLASSES, range(3)))).to_numpy()
    assert len(X) == 400 and not set(split['development_row_indices']) & set(split['test_row_indices'])
    _, _, _, _, wide, merged = load_sources(); features = sorted(wide.columns)
    role_vectors = {role: np.nanmean(wide.loc[[code[:7] for code in codes], features].to_numpy(), axis=0)
                    for role, (status, codes, _) in ROLE_MAP.items() if status in ['role_equivalent', 'candidate_range']}
    with threadpool_limits(limits=2):
        scaler, pretrained, shuffled = external_pretraining(merged, features)
        skills, mask = profile_skills(X, role_vectors, scaler)
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y))
        raw_template = make_model('logistic')[:-1]
        metrics, predictions, epochs, split_rows = [], [], [], []
        for fold, (tr, va) in enumerate(splits, 1):
            split_rows.extend({'repeat': (fold-1)//5+1, 'fold': fold, 'development_position': int(i),
                               'source_row': split['development_row_indices'][i]} for i in va)
            for seed in SEEDS:
                started = time.monotonic()
                inner_tr, inner_va = next(StratifiedShuffleSplit(n_splits=1, test_size=.2,
                    random_state=seed+1000*fold).split(X.iloc[tr], y[tr]))
                fit_idx, stop_idx = tr[inner_tr], tr[inner_va]
                assert not set(fit_idx) & set(stop_idx) and not set(tr) & set(va)
                inner_processor = clone(raw_template).fit(X.iloc[fit_idx])
                raw_fit = as_dense(inner_processor.transform(X.iloc[fit_idx]))
                raw_stop = as_dense(inner_processor.transform(X.iloc[stop_idx]))
                outer_processor = clone(raw_template).fit(X.iloc[tr])
                raw_tr = as_dense(outer_processor.transform(X.iloc[tr])); raw_va = as_dense(outer_processor.transform(X.iloc[va]))
                head_seed = seed+10000*fold
                for condition in CONDITIONS:
                    enc = pretrained[seed] if condition.startswith('Pretrained') else (shuffled[seed] if condition=='Shuffled-pretraining control' else None)
                    use_mask = np.zeros_like(mask) if condition=='Raw MLP' else mask
                    freeze = condition in ['Raw MLP', 'Pretrained frozen encoder']
                    initial = initialize(len(features), raw_fit.shape[1], head_seed, enc)
                    _, selected_epoch, history = train(initial, skills[fit_idx], y[fit_idx], raw_fit, use_mask[fit_idx],
                        validation=(skills[stop_idx], y[stop_idx], raw_stop, use_mask[stop_idx]), freeze_encoder=freeze)
                    # Refit from the identical starting encoder on the full outer training fold.
                    initial = initialize(len(features), raw_tr.shape[1], head_seed, enc)
                    before_encoder = {k: initial[k].copy() for k in ENCODER}
                    fitted, _, _ = train(initial, skills[tr], y[tr], raw_tr, use_mask[tr], epochs=selected_epoch, freeze_encoder=freeze)
                    movement = float(np.sqrt(sum(np.sum((fitted[k]-before_encoder[k])**2) for k in ENCODER)))
                    if freeze: assert movement == 0
                    proba, _ = forward(fitted, skills[va], raw_va, use_mask[va]); pred = proba.argmax(axis=1)
                    train_proba, _ = forward(fitted, skills[tr], raw_tr, use_mask[tr])
                    row = {'condition': condition, 'seed': seed, 'fold': fold, 'repeat': (fold-1)//5+1,
                           'macro_f1': f1_score(y[va], pred, average='macro', labels=[0,1,2], zero_division=0),
                           'accuracy': accuracy_score(y[va], pred), 'balanced_accuracy': balanced_accuracy_score(y[va], pred),
                           'log_loss': log_loss(y[va], proba, labels=[0,1,2]),
                           'train_macro_f1': f1_score(y[tr], train_proba.argmax(axis=1), average='macro', labels=[0,1,2], zero_division=0),
                           'selected_epochs': selected_epoch, 'stopped_epoch': len(history), 'encoder_weight_change_l2': movement,
                           'mapped_validation_rows': int(mask[va].sum())}
                    for mapped, label in [(True,'mapped'), (False,'unmapped')]:
                        subset = (mask[va] == 1) if mapped else (mask[va] == 0)
                        row[f'{label}_macro_f1'] = f1_score(y[va][subset], pred[subset], average='macro', labels=[0,1,2], zero_division=0)
                    metrics.append(row)
                    predictions.extend({'condition': condition, 'seed': seed, 'fold': fold,
                        'source_row': split['development_row_indices'][i], 'mapped': bool(mask[i]),
                        'actual': CLASSES[y[i]], 'prediction': CLASSES[pr.argmax()],
                        **dict(zip(['score_Low','score_Medium','score_High'], pr))} for i, pr in zip(va, proba))
                    epochs.append({'condition': condition, 'seed': seed, 'fold': fold,
                                   'selected_epochs': selected_epoch, 'inner_validation_size': len(stop_idx)})
                print(f'Fold {fold}/15, seed {seed}: five conditions completed in {time.monotonic()-started:.1f}s', flush=True)
        metric_df = pd.DataFrame(metrics)
        metric_df.to_csv(OUT/'cv_fold_results.csv', index=False)
        pd.DataFrame(predictions).to_csv(OUT/'cv_predictions.csv', index=False)
        ensemble_rows = []
        prediction_df = pd.DataFrame(predictions)
        score_columns = ['score_Low', 'score_Medium', 'score_High']
        for (condition, fold), group in prediction_df.groupby(['condition', 'fold']):
            assert group.groupby('source_row').size().eq(len(SEEDS)).all()
            probabilities = group.groupby('source_row')[score_columns].mean()
            actual = group.groupby('source_row').actual.first().loc[probabilities.index]
            predicted = CLASSES[probabilities.to_numpy().argmax(axis=1)]
            ensemble_rows.append({'condition': condition, 'fold': fold,
                'macro_f1': f1_score(actual, predicted, average='macro', labels=CLASSES, zero_division=0),
                'accuracy': accuracy_score(actual, predicted)})
        ensemble_folds = pd.DataFrame(ensemble_rows)
        ensemble_folds.to_csv(OUT/'ensemble_cv_fold_results.csv', index=False)
        ensemble_folds.groupby('condition').agg(macro_f1_mean=('macro_f1','mean'),
            macro_f1_sd=('macro_f1','std'), accuracy_mean=('accuracy','mean')).to_csv(OUT/'ensemble_model_comparison.csv')
        pd.DataFrame(epochs).to_csv(OUT/'stopping_epochs.csv', index=False)
        pd.DataFrame(split_rows).to_csv(OUT/'outer_split_manifest.csv', index=False)
        comparison = metric_df.groupby('condition', sort=False).agg(
            macro_f1_mean=('macro_f1','mean'), macro_f1_sd=('macro_f1','std'), accuracy_mean=('accuracy','mean'),
            training_macro_f1_mean=('train_macro_f1','mean'), selected_epochs_median=('selected_epochs','median'),
            mapped_macro_f1_mean=('mapped_macro_f1','mean'), unmapped_macro_f1_mean=('unmapped_macro_f1','mean'))
        comparison.to_csv(OUT/'model_comparison.csv')
        matched = metric_df.pivot(index=['fold','seed'], columns='condition', values='macro_f1')
        delta = pd.DataFrame({'fine_tuned_minus_scratch': matched['Pretrained fine-tuned encoder']-matched['Enriched MLP from scratch'],
                              'fine_tuned_minus_raw': matched['Pretrained fine-tuned encoder']-matched['Raw MLP'],
                              'fine_tuned_minus_shuffled': matched['Pretrained fine-tuned encoder']-matched['Shuffled-pretraining control']})
        delta.to_csv(OUT/'paired_fold_differences.csv')
        metric_df.groupby(['condition','seed']).macro_f1.mean().to_csv(OUT/'seed_results.csv')
        # Export the user-requested fine-tuned model, regardless of whether it wins.
        raw_processor = clone(raw_template).fit(X)
        all_raw = as_dense(raw_processor.transform(X)); ensemble_params = []; exported_epochs = {}
        for seed in SEEDS:
            budget = max(1,int(np.median(metric_df.loc[(metric_df.condition=='Pretrained fine-tuned encoder') & (metric_df.seed==seed),'selected_epochs'])))
            initial = initialize(len(features), all_raw.shape[1], seed, pretrained[seed])
            fitted, _, _ = train(initial, skills, y, all_raw, mask, epochs=budget, freeze_encoder=False)
            assert any(not np.array_equal(fitted[k], pretrained[seed][k]) for k in ENCODER), 'Export must actually fine-tune the encoder.'
            exported_epochs[str(seed)] = budget; ensemble_params.append(fitted)
            np.savez_compressed(OUT/f'fine_tuned_seed_{seed}.npz', **fitted)
        model = TransferRiskPredictor(raw_processor, scaler, role_vectors, ensemble_params)
        joblib.dump(model, OUT/'fine_tuned_risk_model.joblib')
        metadata = {'target': 'Original Automation_Risk labels', 'class_order': CLASSES.tolist(),
                    'pretraining_target': 'Constructed language-model occupational exposure, not displacement',
                    'config': CONFIG, 'seeds': SEEDS, 'conditions': CONDITIONS,
                    'internal_training_rows': len(X), 'external_training_occupations': len(merged),
                    'shared_skill_features': features, 'mapped_development_rows': int(mask.sum()),
                    'mapping_policy': 'Only six accepted title mappings; ambiguous/proxy/unknown roles have a masked-out occupational branch. Source specialty vectors averaged within SOC, then equally across candidate occupations.',
                    'evaluation': 'Original 400 development rows, same 15 repeated stratified folds, three seeds. Inner 20% stratified validation selects epochs; full outer training refit uses that fixed budget. All internal preprocessing fits inside those training partitions.',
                    'checkpoint_policy': 'Validation can choose a checkpoint from the 20-epoch head warmup; such outer-fold refits retain a frozen encoder. Exported full-development checkpoints are verified to update the encoder.',
                    'external_stopping': 'One fixed holdout of SOC major groups chooses external epochs per seed; then refit on all 774 external occupations. This diagnostic is not a new independent exposure test.',
                    'control_matching': 'New heads start identically within fold/seed; enrichment and fine-tuning controls share optimizer, encoder rate, warmup, regularization and stopping. Shuffled source labels use the same pretraining epoch budget.',
                    'reserved_test_evaluated': False, 'export_fit': 'Fine-tuned three-seed ensemble trained on development rows only; median CV stopping epoch per seed. No independent evaluation of this final ensemble.',
                    'exported_epochs': exported_epochs, 'selected_export_condition': 'Pretrained fine-tuned encoder (requested, not selected as winner)',
                    'probabilities_calibrated': False, 'job_loss_probability': False,
                    'limitations': ['External descriptors are deterministic by six job titles; no new within-role information.',
                                    'Different source targets and geographic/temporal scope remain; successful transfer would predict challenge labels, not validate real risk.',
                                    'Repeated folds overlap; paired differences and standard deviations are descriptive, not significance tests.',
                                    'Exploratory follow-up after earlier test results were seen; no new independent test claim.',
                                    'Single-seed means and secondary three-seed ensemble CV are both reported; final full-development ensemble has no independent evaluation.'],
                    'protected_files_sha256': protected_hashes,
                    'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': __import__('sklearn').__version__},
                    'pretraining_encoder_sha256': {str(seed): hashlib.sha256(b''.join(pretrained[seed][k].tobytes() for k in ENCODER)).hexdigest() for seed in SEEDS}}
        for p in protected:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == protected_hashes[str(p.relative_to(ROOT))]
        (OUT/'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(comparison.to_string())
    print('Mean paired differences:', delta.mean().to_dict())
    return metadata, comparison


if __name__ == '__main__':
    # Persist importable classes rather than __main__ classes in joblib exports.
    from transfer_mlp import run_experiment as run_importable_experiment
    run_importable_experiment()
