#!/usr/bin/env python3
# ============================================================
# Anti-Fraud Detection Pipeline
# Запуск: python3 anti_fraud_pipeline.py
# ============================================================

# ============================================================
# 1. Setup
# ============================================================
import subprocess
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
print("✅ Бібліотеки завантажено")

# ============================================================
# 2. Завантаження даних з BigQuery
# ============================================================
from google.cloud import bigquery

PROJECT = os.getenv("PROJECT_ID", "project-4034cdb0-f49f-4702-bbb")
DATASET = os.getenv("DATASET_ID", "fraud_detection")
client  = bigquery.Client(project=PROJECT)

def bq(table):
    print(f"  Завантаження {table}...")
    return client.query(f"SELECT * FROM `{PROJECT}.{DATASET}.{table}`").to_dataframe()

train_users        = bq("train_users")
train_transactions = bq("train_transactions")
test_users         = bq("test_users")
test_transactions  = bq("test_transactions")

# Parse timestamps
for df, col in [
    (train_transactions, 'timestamp_tr'),
    (test_transactions,  'timestamp_tr'),
    (train_users,        'timestamp_reg'),
    (test_users,         'timestamp_reg'),
]:
    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

print(f"✅ Train users: {train_users.shape}, Test users: {test_users.shape}")
print(f"✅ Train tx: {train_transactions.shape}, Test tx: {test_transactions.shape}")
print(f"✅ Fraud rate: {train_users['is_fraud'].mean():.4f}")

# ============================================================
# 3. Feature Engineering
# ============================================================
def build_features(users_df, transactions_df):
    df = users_df.copy()
    tx = transactions_df.copy()

    total_per_user = tx.groupby('id_user').size()

    # Domain
    df['domain'] = df['email'].apply(lambda x: x.split('@')[1] if pd.notna(x) else 'missing')

    # Basic aggregations
    df['tx_count'] = df['id_user'].map(total_per_user).fillna(0)

    amount_agg = tx.groupby('id_user')['amount'].agg(
        ['mean', 'std', 'max', 'min', 'median', 'sum']
    ).rename(columns={
        'mean': 'amount_mean', 'std': 'amount_std', 'max': 'amount_max',
        'min': 'amount_min', 'median': 'amount_median', 'sum': 'amount_sum'
    })
    amount_agg['amount_std']   = amount_agg['amount_std'].fillna(0)
    amount_agg['amount_range'] = amount_agg['amount_max'] - amount_agg['amount_min']
    df = df.merge(amount_agg, on='id_user', how='left')

    # Diversity
    df['unique_cards_per_user']  = df['id_user'].map(tx.groupby('id_user')['card_mask_hash'].nunique()).fillna(0)
    df['unique_card_holders']    = df['id_user'].map(tx.groupby('id_user')['card_holder'].nunique()).fillna(0)
    df['unique_error_types']     = df['id_user'].map(tx.groupby('id_user')['error_group'].nunique()).fillna(0)
    df['unique_card_brands']     = df['id_user'].map(tx.groupby('id_user')['card_brand'].nunique()).fillna(0)
    df['unique_card_types']      = df['id_user'].map(tx.groupby('id_user')['card_type'].nunique()).fillna(0)
    df['unique_card_countries']  = df['id_user'].map(tx.groupby('id_user')['card_country'].nunique()).fillna(0)

    # Error rates
    failed_per_user      = tx[tx['status'] == 'fail'].groupby('id_user').size()
    df['fail_status_coefficient'] = df['id_user'].map(failed_per_user / total_per_user).fillna(0)

    fraud_error_per_user  = tx[tx['error_group'] == 'fraud'].groupby('id_user').size()
    df['fraud_error_rate']  = df['id_user'].map(fraud_error_per_user / total_per_user).fillna(0)
    df['fraud_error_count'] = df['id_user'].map(fraud_error_per_user).fillna(0)

    for err in ['antifraud', '3ds error', 'card problem']:
        err_count = tx[tx['error_group'] == err].groupby('id_user').size()
        col_name  = f"error_{err.replace(' ', '_')}_rate"
        df[col_name] = df['id_user'].map(err_count / total_per_user).fillna(0)

    # Geo mismatch
    tx_cp = tx.copy()
    tx_cp['country_CP_mismatch'] = (tx_cp['card_country'] != tx_cp['payment_country'])
    df['country_CP_missmatch_coef'] = df['id_user'].map(
        tx_cp.groupby('id_user')['country_CP_mismatch'].mean()
    ).fillna(0)

    tx_cr = tx.merge(df[['id_user', 'reg_country']], on='id_user', suffixes=('', '_user'))
    tx_cr['card_reg_mismatch'] = (tx_cr['card_country'] != tx_cr['reg_country'])
    df['country_CReg_missmatch'] = df['id_user'].map(
        tx_cr.groupby('id_user')['card_reg_mismatch'].mean()
    ).fillna(0)

    # Interaction
    df['errors_x_cards']  = df['unique_error_types'] * df['unique_cards_per_user']
    df['holders_x_cards'] = df['unique_card_holders'] * df['unique_cards_per_user']

    # Ratios
    df['countries_per_tx'] = df['unique_card_countries'] / (df['tx_count'] + 1e-6)
    df['errors_per_tx']    = df['unique_error_types']    / (df['tx_count'] + 1e-6)
    df['holders_per_card'] = df['unique_card_holders']   / (df['unique_cards_per_user'] + 1e-6)
    df['holders_per_tx']   = df['unique_card_holders']   / (df['tx_count'] + 1e-6)

    # TX type
    df['tx_type_card_init_rate'] = df['id_user'].map(
        tx[tx['transaction_type'] == 'card_init'].groupby('id_user').size() / total_per_user
    ).fillna(0)

    # Velocity
    tx_sorted = tx.sort_values(['id_user', 'timestamp_tr'])
    tx_sorted['time_diff'] = tx_sorted.groupby('id_user')['timestamp_tr'].diff().dt.total_seconds()
    time_agg = tx_sorted.groupby('id_user')['time_diff'].min().rename('min_time_between_tx')
    df = df.merge(time_agg, on='id_user', how='left')

    # Time to first tx
    first_tx_time = tx.groupby('id_user')['timestamp_tr'].min()
    df['first_timestamp_tr'] = df['id_user'].map(first_tx_time)
    df['time_to_first_transaction'] = (
        df['first_timestamp_tr'] - df['timestamp_reg']
    ).dt.total_seconds().clip(lower=0)

    # Round 2 features
    df['unique_amounts']       = df['id_user'].map(tx.groupby('id_user')['amount'].nunique()).fillna(0)
    df['unique_amounts_per_tx'] = df['unique_amounts'] / (df['tx_count'] + 1e-6)

    tx['hour'] = tx['timestamp_tr'].dt.hour
    df['hour_std'] = df['id_user'].map(tx.groupby('id_user')['hour'].std()).fillna(0)

    # Max consecutive fails
    tx_s = tx.sort_values(['id_user', 'timestamp_tr']).copy()
    tx_s['is_fail']   = (tx_s['status'] == 'fail').astype(int)
    tx_s['streak_id'] = (tx_s['is_fail'] != tx_s.groupby('id_user')['is_fail'].shift()).cumsum()
    streak_lengths    = tx_s[tx_s['is_fail'] == 1].groupby(['id_user', 'streak_id']).size()
    max_fails         = streak_lengths.groupby('id_user').max() if len(streak_lengths) > 0 else pd.Series(dtype=float)
    df['max_consecutive_fails'] = df['id_user'].map(max_fails).fillna(0)

    # Card holder features
    holder_features = tx.groupby('id_user')['card_holder'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''
    )
    df['holder_name_length'] = df['id_user'].map(holder_features.apply(len)).fillna(0)
    df['holder_name_words']  = df['id_user'].map(holder_features.apply(lambda x: len(x.split()))).fillna(0)

    # First tx failed
    first_tx_data    = tx_sorted.groupby('id_user').first()
    df['first_tx_failed'] = df['id_user'].map(
        (first_tx_data['status'] == 'fail').astype(int)
    ).fillna(0)

    # Max hourly burst
    tx['hour_bucket'] = tx['timestamp_tr'].dt.floor('h')
    tx_per_hour       = tx.groupby(['id_user', 'hour_bucket']).size()
    df['max_hourly_burst'] = df['id_user'].map(tx_per_hour.groupby('id_user').max()).fillna(0)

    # Interaction round 2
    df['consec_fails_x_cards']   = df['max_consecutive_fails'] * df['unique_cards_per_user']
    df['burst_x_cards']          = df['max_hourly_burst']      * df['unique_cards_per_user']
    df['burst_x_errors']         = df['max_hourly_burst']      * df['unique_error_types']
    df['consec_fails_x_holders'] = df['max_consecutive_fails'] * df['unique_card_holders']
    df['burst_x_fail_rate']      = df['max_hourly_burst']      * df['fail_status_coefficient']

    # Graph features
    card_users = tx.groupby('card_mask_hash')['id_user'].nunique()
    tx['users_per_card'] = tx['card_mask_hash'].map(card_users)
    card_sharing = tx.groupby('id_user')['users_per_card'].agg(['max', 'mean'])
    card_sharing.columns = ['max_users_sharing_card', 'avg_users_sharing_card']
    df = df.merge(card_sharing, on='id_user', how='left')

    card_fail = tx.groupby('card_mask_hash')['status'].apply(lambda x: (x == 'fail').mean())
    tx['card_fail_rate'] = tx['card_mask_hash'].map(card_fail)
    card_fail_agg = tx.groupby('id_user')['card_fail_rate'].agg(['max', 'mean'])
    card_fail_agg.columns = ['worst_card_fail_rate', 'avg_card_fail_rate']
    df = df.merge(card_fail_agg, on='id_user', how='left')

    card_tx_count = tx.groupby('card_mask_hash').size()
    tx['card_total_tx'] = tx['card_mask_hash'].map(card_tx_count)
    card_activity = tx.groupby('id_user')['card_total_tx'].agg(['max', 'mean'])
    card_activity.columns = ['most_active_card_tx', 'avg_card_tx']
    df = df.merge(card_activity, on='id_user', how='left')

    # Sequence features
    def fails_before_first_success(group):
        statuses = group.sort_values('timestamp_tr')['status'].values
        for i, s in enumerate(statuses):
            if s == 'success':
                return i
        return len(statuses)

    fbs = tx.groupby('id_user').apply(fails_before_first_success, include_groups=False)
    df['fails_before_first_success'] = df['id_user'].map(fbs).fillna(0)

    def count_status_changes(group):
        statuses = group.sort_values('timestamp_tr')['status'].values
        if len(statuses) < 2:
            return 0
        return sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i-1])

    sc = tx.groupby('id_user').apply(count_status_changes, include_groups=False)
    df['status_changes']        = df['id_user'].map(sc).fillna(0)
    df['status_changes_per_tx'] = df['status_changes'] / (df['tx_count'] + 1e-6)

    first_5_cards = tx_sorted.groupby('id_user').head(5).groupby('id_user')['card_mask_hash'].nunique()
    df['unique_cards_first_5'] = df['id_user'].map(first_5_cards).fillna(0)

    # Amount patterns
    user_amount_entropy = tx.groupby('id_user')['amount'].apply(
        lambda a: entropy(a.value_counts(normalize=True))
    )
    df['amount_entropy'] = df['id_user'].map(user_amount_entropy).fillna(0)

    mode_conc = tx.groupby('id_user')['amount'].apply(
        lambda a: a.value_counts().iloc[0] / len(a) if len(a) > 0 else 0
    )
    df['amount_mode_concentration'] = df['id_user'].map(mode_conc).fillna(0)

    tx['is_small_amount'] = (tx['amount'] <= 5).astype(int)
    df['small_amount_rate'] = df['id_user'].map(tx.groupby('id_user')['is_small_amount'].mean()).fillna(0)

    # Cross features
    df['email_name']      = df['email'].apply(lambda x: x.split('@')[0].lower() if pd.notna(x) else '')
    main_holder           = tx.groupby('id_user')['card_holder'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''
    )
    df['main_card_holder'] = df['id_user'].map(main_holder).fillna('')
    df['name_in_email']    = df.apply(
        lambda row: any(part in row['email_name'] for part in row['main_card_holder'].split() if len(part) > 2),
        axis=1
    ).astype(int)

    def unique_first_names(holders):
        names = set()
        for h in holders:
            if isinstance(h, str) and h.strip():
                parts = h.strip().split()
                if parts:
                    names.add(parts[0].lower())
        return len(names)

    ufn = tx.groupby('id_user')['card_holder'].apply(unique_first_names)
    df['unique_first_names'] = df['id_user'].map(ufn).fillna(0)

    df = df.drop(columns=['first_timestamp_tr', 'email_name', 'main_card_holder'], errors='ignore')
    return df

print("\nБудуємо train фічі...")
df_train = build_features(train_users, train_transactions)
print(f"  ✅ {df_train.shape}")

print("Будуємо test фічі...")
df_test = build_features(test_users, test_transactions)
print(f"  ✅ {df_test.shape}")

# ============================================================
# 4. Target Encoding
# ============================================================
def kfold_smoothed_target_encode(train_df, test_df, col, target='is_fraud',
                                  n_splits=5, min_samples=50):
    global_mean = train_df[target].mean()
    train_df[f'{col}_enc'] = np.nan
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df, train_df[target]):
        fold_train = train_df.iloc[train_idx]
        stats      = fold_train.groupby(col)[target].agg(['mean', 'count'])
        smoothed   = (stats['count'] * stats['mean'] + min_samples * global_mean) / \
                     (stats['count'] + min_samples)
        train_df.loc[train_df.index[val_idx], f'{col}_enc'] = \
            train_df.iloc[val_idx][col].map(smoothed)
    train_df[f'{col}_enc'].fillna(global_mean, inplace=True)
    full_stats    = train_df.groupby(col)[target].agg(['mean', 'count'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + min_samples * global_mean) / \
                    (full_stats['count'] + min_samples)
    test_df[f'{col}_enc'] = test_df[col].map(full_smoothed).fillna(global_mean)
    return train_df, test_df

print("\nTarget encoding...")
for col in ['reg_country', 'gender', 'traffic_type', 'domain']:
    df_train, df_test = kfold_smoothed_target_encode(df_train, df_test, col)
    print(f"  ✅ {col}_enc")

# ============================================================
# 5. Graph Features з OOF захистом від leakage
# ============================================================
print("\nGraph features (OOF)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_train['fraud_card_exposure']   = 0.0
df_train['worst_card_fraud_rate'] = 0.0
df_train['avg_card_fraud_rate']   = 0.0

for train_idx, val_idx in kf.split(df_train, df_train['is_fraud']):
    fold_fraud_users = set(df_train.iloc[train_idx][df_train.iloc[train_idx]['is_fraud'] == 1]['id_user'])
    fold_fraud_cards = set(
        train_transactions[train_transactions['id_user'].isin(fold_fraud_users)]['card_mask_hash'].unique()
    )
    fold_tx         = train_transactions[train_transactions['id_user'].isin(df_train.iloc[train_idx]['id_user'])]
    fold_tx_labeled = fold_tx.merge(df_train.iloc[train_idx][['id_user', 'is_fraud']], on='id_user')
    fold_card_fraud_rate = fold_tx_labeled.groupby('card_mask_hash')['is_fraud'].mean()

    val_user_ids = set(df_train.iloc[val_idx]['id_user'])
    val_tx       = train_transactions[train_transactions['id_user'].isin(val_user_ids)]

    user_cards = val_tx.groupby('id_user')['card_mask_hash'].apply(set)
    exposure   = user_cards.apply(lambda cards: len(cards & fold_fraud_cards) / len(cards) if len(cards) > 0 else 0)
    df_train.loc[df_train.index[val_idx], 'fraud_card_exposure'] = \
        df_train.iloc[val_idx]['id_user'].map(exposure).fillna(0).values

    val_tx_rated      = val_tx.copy()
    val_tx_rated['cfr'] = val_tx_rated['card_mask_hash'].map(fold_card_fraud_rate).fillna(0)
    cfr_agg           = val_tx_rated.groupby('id_user')['cfr'].agg(['max', 'mean'])
    df_train.loc[df_train.index[val_idx], 'worst_card_fraud_rate'] = \
        df_train.iloc[val_idx]['id_user'].map(cfr_agg['max']).fillna(0).values
    df_train.loc[df_train.index[val_idx], 'avg_card_fraud_rate'] = \
        df_train.iloc[val_idx]['id_user'].map(cfr_agg['mean']).fillna(0).values

all_fraud_users      = set(df_train[df_train['is_fraud'] == 1]['id_user'])
all_fraud_cards      = set(train_transactions[train_transactions['id_user'].isin(all_fraud_users)]['card_mask_hash'].unique())
full_card_fraud_rate = train_transactions.merge(
    df_train[['id_user', 'is_fraud']], on='id_user'
).groupby('card_mask_hash')['is_fraud'].mean()

test_user_cards = test_transactions.groupby('id_user')['card_mask_hash'].apply(set)
df_test['fraud_card_exposure'] = df_test['id_user'].map(
    test_user_cards.apply(lambda cards: len(cards & all_fraud_cards) / len(cards) if len(cards) > 0 else 0)
).fillna(0)

test_tx_rated       = test_transactions.copy()
test_tx_rated['cfr'] = test_tx_rated['card_mask_hash'].map(full_card_fraud_rate).fillna(0)
cfr_test            = test_tx_rated.groupby('id_user')['cfr'].agg(['max', 'mean'])
df_test['worst_card_fraud_rate'] = df_test['id_user'].map(cfr_test['max']).fillna(0)
df_test['avg_card_fraud_rate']   = df_test['id_user'].map(cfr_test['mean']).fillna(0)
print("  ✅ Graph features готові")

# ============================================================
# 6. Anomaly Features
# ============================================================
print("\nAnomaly features (KNN)...")
anomaly_features = [
    'errors_x_cards', 'holders_x_cards',
    'unique_card_holders', 'unique_error_types', 'unique_cards_per_user',
    'tx_count', 'fail_status_coefficient', 'countries_per_tx',
    'amount_sum', 'amount_range', 'min_time_between_tx',
    'max_consecutive_fails', 'max_hourly_burst', 'unique_amounts_per_tx',
]
X_train_anom  = df_train[anomaly_features].fillna(0)
X_test_anom   = df_test[anomaly_features].fillna(0)
scaler        = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_anom)
X_test_scaled  = scaler.transform(X_test_anom)
nn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
nn.fit(X_train_scaled)
train_dist, _ = nn.kneighbors(X_train_scaled)
test_dist, _  = nn.kneighbors(X_test_scaled)
df_train['knn_avg_dist'] = train_dist.mean(axis=1)
df_test['knn_avg_dist']  = test_dist.mean(axis=1)
print("  ✅ KNN anomaly features готові")

# ============================================================
# 7. Feature Selection
# ============================================================
feature_cols = [
    'errors_x_cards', 'holders_x_cards',
    'consec_fails_x_cards', 'burst_x_cards', 'burst_x_errors',
    'consec_fails_x_holders', 'burst_x_fail_rate',
    'unique_card_holders', 'unique_error_types', 'unique_cards_per_user',
    'unique_card_brands', 'unique_card_types',
    'tx_count',
    'fail_status_coefficient', 'error_antifraud_rate',
    'fraud_error_count', 'fraud_error_rate', 'error_3ds_error_rate',
    'countries_per_tx', 'errors_per_tx', 'holders_per_card', 'holders_per_tx',
    'country_CP_missmatch_coef', 'country_CReg_missmatch',
    'amount_sum', 'amount_range',
    'min_time_between_tx',
    'tx_type_card_init_rate',
    'max_consecutive_fails', 'unique_amounts_per_tx', 'max_hourly_burst',
    'first_tx_failed', 'hour_std', 'holder_name_length', 'holder_name_words',
    'unique_amounts',
    'knn_avg_dist', 'fraud_card_exposure',
    'worst_card_fraud_rate', 'avg_card_fraud_rate',
    'max_users_sharing_card', 'avg_users_sharing_card',
    'worst_card_fail_rate', 'avg_card_fail_rate',
    'most_active_card_tx', 'avg_card_tx',
    'fails_before_first_success', 'status_changes', 'status_changes_per_tx',
    'unique_cards_first_5',
    'amount_entropy', 'amount_mode_concentration', 'small_amount_rate',
    'name_in_email', 'unique_first_names',
    'reg_country_enc', 'gender_enc', 'traffic_type_enc', 'domain_enc',
]
feature_cols = [c for c in feature_cols if c in df_train.columns and c in df_test.columns]

X      = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y      = df_train['is_fraud']
X_test = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n✅ Features: {len(feature_cols)}")
print(f"✅ Train: {X.shape}, Test: {X_test.shape}")
print(f"✅ Fraud rate: {y.mean():.4f}")

# ============================================================
# 8. Тренування LightGBM
# ============================================================
best_params = {
    'n_estimators':      400,
    'learning_rate':     0.02,
    'max_depth':         8,
    'num_leaves':        200,
    'min_child_samples': 50,
    'subsample':         0.6,
    'colsample_bytree':  0.8,
    'reg_alpha':         2.0,
    'reg_lambda':        1.5,
    'scale_pos_weight':  2,
    'random_state':      42,
    'verbose':           -1,
}

print("\nТренування LightGBM (CV)...")
model  = LGBMClassifier(**best_params)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f"✅ LightGBM CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# 9. Threshold Optimization
# ============================================================
print("\nПошук оптимального threshold...")
model_oof  = LGBMClassifier(**best_params)
oof_probas = cross_val_predict(model_oof, X, y, cv=cv, method='predict_proba')[:, 1]

thresholds     = np.arange(0.05, 0.70, 0.01)
f1_scores_list = [f1_score(y, (oof_probas >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores_list)]
best_f1        = max(f1_scores_list)

print(f"Default (0.50): F1={f1_score(y, (oof_probas >= 0.5).astype(int)):.4f}")
print(f"Optimal ({best_threshold:.2f}): F1={best_f1:.4f}")

y_pred_oof = (oof_probas >= best_threshold).astype(int)
print("\n" + classification_report(y, y_pred_oof, digits=4, target_names=['legit', 'fraud']))

# ============================================================
# 9.5 Reports (metrics + plot)
# ============================================================
os.makedirs("reports", exist_ok=True)

with open("reports/metrics.md", "w") as f:
    f.write("# Model Metrics (OOF)\n\n")
    f.write(f"- Best F1 (OOF): {best_f1:.4f}\n")
    f.write(f"- Optimal threshold: {best_threshold:.2f}\n")
    f.write(f"- Fraud rate: {y.mean()*100:.2f}%\n")

plt.figure(figsize=(6, 4))
plt.plot(thresholds, f1_scores_list, linewidth=2)
plt.axvline(best_threshold, color="red", linestyle="--")
plt.title("F1 vs Threshold (OOF)")
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/f1_vs_threshold.png", dpi=150)
plt.close()

# ============================================================
# 10. Submission
# ============================================================
print("\nГенерація submission...")
model_submit = LGBMClassifier(**best_params)
model_submit.fit(X, y)

test_probas_final = model_submit.predict_proba(X_test)[:, 1]
test_predictions  = (test_probas_final >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id_user':  df_test['id_user'],
    'is_fraud': test_predictions
})
submission.to_csv('submission.csv', index=False)

print(f"\n{'='*40}")
print(f"✅ submission.csv збережено!")
print(f"   Всього юзерів:   {len(submission)}")
print(f"   Помічено фродом: {test_predictions.sum()} ({test_predictions.mean()*100:.2f}%)")
print(f"   Best F1 (OOF):   {best_f1:.4f}")
print(f"   Threshold:       {best_threshold:.2f}")
print(f"{'='*40}")

# Зберегти в GCS (optional)
try:
    subprocess.run(
        ["gsutil", "cp", "submission.csv", "gs://ai_competition/output/submission.csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    print("✅ Збережено в gs://ai_competition/output/submission.csv")
    print("   Скачайте: gsutil cp gs://ai_competition/output/submission.csv .")
except FileNotFoundError:
    print("INFO: gsutil не знайдено — пропускаємо upload у GCS.")
except subprocess.CalledProcessError as e:
    print("WARN: Не вдалося завантажити у GCS.")
    if e.stderr:
        print(e.stderr.strip())
