import pandas as pd
from sklearn.metrics import precision_score


class Evaluator:
    def __init__(self, id_train, id_test):
        self.id_train = id_train
        self.id_test = id_test

    def oof_precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='binary', pos_label="Extrovert")

    def build_oof_df(self, oof_pred, y_true):
        return pd.DataFrame({
            'id': self.id_train,
            'oof_pred': oof_pred,
            'y_true':  y_true
        })

    def build_test_df(self, test_pred):
        return pd.DataFrame({
            'id': self.id_test,
            'prediction': test_pred
        })
