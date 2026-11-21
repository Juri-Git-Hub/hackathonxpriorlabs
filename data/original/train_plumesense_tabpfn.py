"""Run TabPFN on held-out simulation episodes; use local TabPFN install."""
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from tabpfn import TabPFNClassifier

train = pd.read_csv('plumesense_tabpfn_train.csv')
test = pd.read_csv('plumesense_tabpfn_test.csv')
target = 'target_hazard_t4'
features = [c for c in train.columns if c != target]
model = TabPFNClassifier(device='cuda')
model.fit(train[features], train[target])
p = model.predict_proba(test[features])[:, list(model.classes_).index(1)]
y_pred = (p >= 0.5).astype(int)
print(classification_report(test[target], y_pred, zero_division=0))
if test[target].nunique() == 2:
    print('AUROC:', roc_auc_score(test[target], p))
print('Brier:', brier_score_loss(test[target], p))
out = pd.read_csv('plumesense_test.csv')
out['predicted_hazard_probability_t4'] = p
out['predicted_hazard_t4'] = y_pred
out.to_csv('plumesense_hazard_predictions.csv', index=False)
print('Saved plumesense_hazard_predictions.csv')
