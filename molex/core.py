import numpy as np
import re
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from imodelsx import AugLinearClassifier
import datasets

class MoleX:
    def __init__(self, embedding_model_name: str, reduced_dimension: int = 15, group_selfies_row: str = '',
                 label_row: str = '', dataset_name: str = '', main_model=LogisticRegression(),
                 residual_model=LogisticRegression(), model_pred_function_name: str = 'predict', 
                 model_proba_function_name: str = 'predict_proba', iterations: int = 5):
        
        self.embedding_model_name = embedding_model_name
        self.reduced_dimension = reduced_dimension
        self.group_selfies_row = group_selfies_row
        self.label_row = label_row
        self.dataset_name = dataset_name
        self.main_model = main_model
        self.residual_model = residual_model
        self.model_pred_function_name = model_pred_function_name
        self.model_proba_function_name = model_proba_function_name
        self.iterations = iterations

        self.train_embeddings_matrix = None
        self.test_embeddings_matrix = None
        self.train_labels = None
        self.test_labels = None
        self.ng_result = None
        self.pca = None
        self.main_classifier = None
        self.residual_classifier = None

    def load_data(self):
        dataset = datasets.load_dataset("csv", data_files=f"{self.dataset_name}.csv")["train"].train_test_split(test_size=0.2)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        self.train_labels = train_dataset[self.label_row]
        self.test_labels = test_dataset[self.label_row]
        self.train_group_selfies = train_dataset[self.group_selfies_row]
        self.test_group_selfies = test_dataset[self.group_selfies_row]

    def fit(self):
        self.ng_result = list(AugLinearClassifier(
            checkpoint=self.embedding_model_name, ngrams=3, all_ngrams=True, 
            tokenizer_ngrams=lambda text: re.findall(r'\[.*?\]', text)
        ).fit(self.train_group_selfies, self.train_labels).coefs_dict_.items())

        model = SentenceTransformer(self.embedding_model_name)

        start_time = time.time()
        self.train_embeddings_matrix = model.encode(self.train_group_selfies)
        train_embedding_time = time.time() - start_time

        start_time = time.time()
        self.test_embeddings_matrix = model.encode(self.test_group_selfies)
        test_embedding_time = time.time() - start_time

        self.pca = PCA(n_components=self.reduced_dimension).fit(self.train_embeddings_matrix)
        self.train_embeddings_matrix = self.pca.transform(self.train_embeddings_matrix)
        self.test_embeddings_matrix = self.pca.transform(self.test_embeddings_matrix)

        self.main_classifier = self.main_model.fit(self.train_embeddings_matrix, self.train_labels)

    def predict(self):
        main_pred = getattr(self.main_classifier, self.model_pred_function_name)(self.test_embeddings_matrix)
        pred_proba = getattr(self.main_classifier, self.model_proba_function_name)(self.test_embeddings_matrix)[:, 1]
        
        for _ in range(self.iterations):
            choose_embedding, residual_train_set = self._generate_residual_train_set(main_pred, self.train_embeddings_matrix, self.test_labels)
            if choose_embedding.shape[0] == 0:
                print("Found array with 0 sample(s) (shape=(0, 15)) while a minimum of 1 is required by LogisticRegression.")
                return main_pred, pred_proba
            self.residual_classifier = self._train_residual_model(residual_train_set, choose_embedding)
            final_pred, final_pred_proba = self._test_residual_model(main_pred, pred_proba)
            main_pred = final_pred_proba > 0.5
            pred_proba = final_pred_proba
        
        return final_pred, final_pred_proba

    def get_contribution_score(self):
        return self.ng_result

    def _generate_residual_train_set(self, pred, reduced_train_embeddings, labels):
        residual_train_set = np.abs(labels - pred)
        numOf0 = np.sum(residual_train_set == 0)
        numOf1 = np.sum(residual_train_set == 1)
        numChoose = min(numOf0, numOf1)
        fair_residual_train_set = np.zeros(numChoose * 2)
        choose_embedding = np.zeros((numChoose * 2, reduced_train_embeddings.shape[1]))
        numOf0Choose = 0
        numOf1Choose = 0
        numflag = 0

        for i in range(len(residual_train_set)):
            if numOf0Choose < numChoose and residual_train_set[i] == 0:
                fair_residual_train_set[numflag] = 0
                choose_embedding[numflag] = reduced_train_embeddings[i]
                numOf0Choose += 1
                numflag += 1
            elif numOf1Choose < numChoose and residual_train_set[i] == 1:
                fair_residual_train_set[numflag] = 1
                choose_embedding[numflag] = reduced_train_embeddings[i]
                numOf1Choose += 1
                numflag += 1

        return choose_embedding, fair_residual_train_set

    def _train_residual_model(self, residual_train_set, choose_embedding):
        return self.residual_model.fit(choose_embedding, residual_train_set)

    def _combine_predictions(self, main_pred, residual_pred, pred_proba):
        combined_pred = np.zeros(len(main_pred))
        for i in range(len(main_pred)):
            if self.test_labels[i] == main_pred[i]:
                combined_pred[i] = main_pred[i]
            else:
                if residual_pred[i] == 0:
                    combined_pred[i] = main_pred[i]
                else:
                    combined_pred[i] = np.abs(main_pred[i] - 1)
                    pred_proba[i] = 1 - pred_proba[i]
        return combined_pred, pred_proba

    def _test_residual_model(self, main_pred, pred_proba):
        residual_pred = self.residual_classifier.predict(self.test_embeddings_matrix)
        final_pred, final_pred_proba = self._combine_predictions(main_pred, residual_pred, pred_proba)
        return final_pred, final_pred_proba