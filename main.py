"""Reproducible, causal water-pump early-warning baseline. See README."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            average_precision_score, roc_auc_score, confusion_matrix,
                            ConfusionMatrixDisplay, precision_recall_curve)

ROOT = Path(__file__).resolve().parent
HORIZON = pd.Timedelta(hours=24)
TRAIN_END = pd.Timestamp('2018-06-01')
TEST_START = pd.Timestamp('2018-07-01')


def metrics(y, p, threshold):
    pred = p >= threshold
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                average_precision=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main(csv):
    out = ROOT / 'results'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    # 1. Inspect before conversion; timestamp is initially an object/string.
    raw = pd.read_csv(csv)
    raw.dtypes.astype(str).rename('dtype').to_csv(out / 'original_dtypes.csv')
    counts = raw.machine_status.value_counts()
    counts.to_csv(out / 'status_counts.csv')
    df = raw.copy()
    df.timestamp = pd.to_datetime(df.timestamp, errors='raise')
    df = df.sort_values('timestamp').set_index('timestamp')
    assert df.index.is_unique, 'Duplicate timestamps need an explicit resolution policy.'
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=1)).all(), 'Expected one-minute grid.'
    sensors = [c for c in df if c.startswith('sensor_')]
    missing = df[sensors].isna().mean() * 100
    missing.rename('missing_percent').to_csv(out / 'missing_percent.csv')
    # 2. Decide drops using training data only; preserve raw missingness for auditing.
    train_missing = df.loc[df.index < TRAIN_END, sensors].isna().mean()
    dropped = train_missing[train_missing > .5].index.tolist()
    kept = [c for c in sensors if c not in dropped]
    numeric = df[kept]
    # Carry past readings for at most 60 min; no interpolation/backfill from future.
    # Remaining gaps use fixed TRAIN-only medians. Historical context crosses splits.
    medians = numeric.loc[numeric.index < TRAIN_END].median()
    clean = numeric.ffill(limit=60).fillna(medians)
    assert not clean.isna().any().any()
    # 3. Exploratory ranking: median absolute standardized shift in 24h before each
    # training failure versus 7d-to-48h earlier NORMAL readings. No test selection.
    failures = df.index[df.machine_status.eq('BROKEN')]
    scores = []
    for event in failures[failures < TRAIN_END]:
        base = numeric.loc[(df.index >= event-pd.Timedelta(days=7)) &
                           (df.index < event-pd.Timedelta(hours=48)) & df.machine_status.eq('NORMAL')]
        pre = numeric.loc[(df.index >= event-HORIZON) & (df.index < event) & df.machine_status.eq('NORMAL')]
        scale = base.std().replace(0, np.nan)
        scores.append(((pre.mean()-base.mean()) / scale).abs())
    ranking = pd.concat(scores, axis=1).median(axis=1).sort_values(ascending=False)
    ranking.rename('median_absolute_standardized_shift').to_csv(out / 'pre_failure_sensor_ranking.csv')
    top = ranking.head(5).index.tolist()
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    for ax, col in zip(axes, top):
        ax.plot(clean.index[::15], clean[col].iloc[::15], lw=.6)
        for event in failures: ax.axvline(event, color='crimson', alpha=.7, lw=1)
        ax.set_ylabel(col)
    axes[0].set_title('Training-ranked sensors | red lines: failure moments')
    fig.tight_layout(); fig.savefig(figs / 'sensor_timeline.png', dpi=160); plt.close(fig)
    corr = numeric.loc[df.index < TRAIN_END].corr()
    corr.to_csv(out / 'training_correlation.csv')
    fig, ax = plt.subplots(figsize=(11, 10)); im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(kept)), kept, rotation=90, fontsize=5)
    ax.set_yticks(range(len(kept)), kept, fontsize=5)
    ax.set_title('Sensor correlation: training period only'); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(figs / 'correlation.png', dpi=160); plt.close(fig)
    # 4. Label (t, t+24h]; BROKEN/RECOVERING are never model inputs.
    # Exclude non-NORMAL minutes and the right-censored final 24 hours.
    event_ns = failures.asi8
    next_idx = np.searchsorted(event_ns, df.index.asi8, side='right')
    next_ns = np.full(len(df), np.iinfo(np.int64).max, dtype=np.int64)
    available = next_idx < len(event_ns)
    next_ns[available] = event_ns[next_idx[available]]
    y = pd.Series((next_ns <= (df.index+HORIZON).asi8).astype(int), index=df.index)
    eligible = df.machine_status.eq('NORMAL') & (df.index <= df.index.max()-HORIZON)
    # Past/current values only. Rolling windows include current minute.
    X = pd.concat([clean, clean.rolling(60, min_periods=1).mean().add_suffix('_mean60'),
                   clean.diff(60).fillna(0).add_suffix('_delta60')], axis=1).astype('float32')
    train = eligible & (df.index < TRAIN_END-HORIZON)
    val = eligible & (df.index >= TRAIN_END) & (df.index < TEST_START-HORIZON)
    test = eligible & (df.index >= TEST_START)
    # Purge 24h before boundaries so training/validation labels do not see next split.
    assert (df.index[train]+HORIZON).max() < TRAIN_END
    assert (df.index[val]+HORIZON).max() < TEST_START
    assert not set(X.columns).intersection({'machine_status', 'timestamp', 'target'})
    model = RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=20,
                                   class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    # All eligible training minutes, no random temporal split.
    model.fit(X.loc[train], y.loc[train])
    pv = model.predict_proba(X.loc[val])[:, 1]
    thresholds = np.arange(.05, .96, .05)
    tuning = pd.DataFrame([dict(threshold=float(t), **{k:v for k,v in metrics(y.loc[val], pv, t).items()
                           if k != 'confusion_matrix'}) for t in thresholds])
    tuning.to_csv(out / 'validation_thresholds.csv', index=False)
    threshold = float(tuning.loc[tuning.f1.idxmax(), 'threshold'])
    pt = model.predict_proba(X.loc[test])[:, 1]
    result = metrics(y.loc[test], pt, threshold)
    default_threshold_result = metrics(y.loc[test], pt, 0.5)
    (out / 'default_threshold_metrics.json').write_text(json.dumps(default_threshold_result, indent=2))
    prediction = pd.DataFrame({'target':y.loc[test], 'probability':pt})
    prediction['alert'] = prediction.probability >= threshold
    prediction.to_csv(out / 'test_predictions.csv')
    # 5. Evaluate event coverage and false alert episodes (consecutive alert runs).
    events = []
    for event in failures[failures >= TEST_START]:
        window = prediction.loc[(prediction.index >= event-HORIZON) & (prediction.index < event)]
        alarms = window.index[window.alert]
        events.append(dict(failure_time=str(event), eligible_minutes=len(window), detected=bool(len(alarms)),
                           first_alert_lead_hours=float((event-alarms[0]).total_seconds()/3600) if len(alarms) else None))
    event_table = pd.DataFrame(events); event_table.to_csv(out / 'test_failure_events.csv', index=False)
    starts = prediction.alert & ~(prediction.alert.shift(fill_value=False) &
                                  prediction.index.to_series().diff().eq(pd.Timedelta(minutes=1)))
    episodes = []; group = starts.cumsum()
    for _, segment in prediction.loc[prediction.alert].groupby(group[prediction.alert]):
        episodes.append(dict(start=str(segment.index[0]), end=str(segment.index[-1]), minutes=len(segment),
                             overlaps_pre_failure=bool(segment.target.any())))
    pd.DataFrame(episodes, columns=['start','end','minutes','overlaps_pre_failure']).to_csv(out / 'alert_episodes.csv', index=False)
    result.update(threshold=threshold, test_events=len(events), detected_events=int(event_table.detected.sum()),
                  false_alert_episodes=sum(not e['overlaps_pre_failure'] for e in episodes),
                  false_alert_minutes=int(((prediction.target == 0) & prediction.alert).sum()),
                  normal_exposure_days=float((prediction.target == 0).sum()/1440),
                  always_negative_accuracy=float((y.loc[test] == 0).mean()))
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importance.rename('importance').to_csv(out / 'feature_importance.csv')
    fig, ax = plt.subplots(figsize=(6, 5)); ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=['No failure <=24h','Failure <=24h']).plot(ax=ax, colorbar=False)
    ax.set_title('Final chronological test'); fig.tight_layout(); fig.savefig(figs / 'confusion_matrix.png', dpi=160); plt.close(fig)
    prec, rec, _ = precision_recall_curve(y.loc[test], pt)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(rec, prec, label=f"AP = {result['average_precision']:.3f}")
    ax.axhline(y.loc[test].mean(), ls='--', color='gray', label='Positive prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-recall on untouched test period'); ax.legend()
    fig.tight_layout(); fig.savefig(figs / 'precision_recall.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(prediction.index[::15], pt[::15], lw=.8)
    ax.axhline(threshold, color='orange', ls='--', label='Validation-selected threshold')
    for event in failures[failures >= TEST_START]:
        ax.axvspan(event-HORIZON, event, color='red', alpha=.15)
        ax.axvline(event, color='red', lw=1)
    ax.set(ylabel='Estimated risk score', title='Test warning scores | shaded: 24h before failure'); ax.legend()
    fig.tight_layout(); fig.savefig(figs / 'test_warning_scores.png', dpi=160); plt.close(fig)
    summary = dict(shape=list(raw.shape), statuses=counts.to_dict(), failures=[str(t) for t in failures],
                   dropped_sensors=dropped, top_sensors=top, dropped_index_columns=[c for c in raw if c.startswith('Unnamed:')],
                   split_rows={'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum())},
                   split_positives={'train':int(y[train].sum()),'validation':int(y[val].sum()),'test':int(y[test].sum())},
                   test=result)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data'/'sensor.csv')
    main(parser.parse_args().data)
