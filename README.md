![./logo.png]

MoleX official repo

```
git clone https://github.com/MoleX2024/MoleX
cd MoleX
pip install .
```

Example usage
```
from molex import MoleX, calculate_accuracy, calculate_roc, compute_frag_weights
def main():
    molex_model = MoleX(
        embedding_model_name='moleX-berta',
        group_selfies_row='group_selfies',
        label_row='p_np',
        dataset_name='mutag_group_selfies',
    )
    molex_model.load_data()
    molex_model.fit()
    pred, proba = molex_model.predict()
    contribution_scores = molex_model.get_contribution_score()

    print(f"Accuracy: {calculate_accuracy(molex_model.test_labels, pred):.4f}")
    print(f"ROC AUC: {calculate_roc(molex_model.test_labels, proba):.4f}")
    print("NG contribution_scores:", compute_frag_weights(contribution_scores))

if __name__ == "__main__":
    main()
```
