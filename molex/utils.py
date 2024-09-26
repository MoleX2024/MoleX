from sklearn.metrics import roc_curve, auc
import re
from collections import defaultdict

def calculate_accuracy(true, pred):
    return sum(true == pred) / len(true)

def calculate_roc(true_labels, pred_proba):
    return auc(*roc_curve(true_labels, pred_proba)[:2])

def compute_frag_weights(ngram_list):
    frag_weights = defaultdict(float)
    
    for ngram, weight in ngram_list:
        for part in re.findall(r'\[([^\]]+)\]', ngram):
            if (m := re.search(r'frag(\d+)', part)):
                frag_weights[f'frag{m.group(1)}'] += weight
    
    for i in range(30):
        frag_weights[f'frag{i}'] += 0.0
    
    sorted_weights = sorted(frag_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return [list(item) for i, item in enumerate(sorted_weights) if i < 5 or abs(item[1]) > 1][:5]