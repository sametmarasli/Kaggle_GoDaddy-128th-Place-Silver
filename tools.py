from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):

    """Select only specified columns."""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
	    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        return X_transformed
    
    # def get_feature_names_out(self, X):
    #     return self.features


    
class SimpleFeatureEngineering(BaseTransformer):
    def __init__(self, features):
        self.features = features
	    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        col1 = f'ratio_{self.features[0]}_{self.features[1]}'
        X_transformed[col1] = ((X_transformed[self.features[0]]/X_transformed[self.features[1]]) - 1).fillna(0)
        col2 = f'ratio_{self.features[1]}_{self.features[2]}'
        X_transformed[col2] = ((X_transformed[self.features[1]]/X_transformed[self.features[2]]) - 1).fillna(0)
        
        new_features = [col1,col2]
        return X_transformed[new_features]
        


class LagModel(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        # print(np.mean(X,axis='rows'))
    
        return np.mean(X,axis=1)



    

def create_submission(df_output,date_submission, model_name, local_score, sample_submission):
    
    df_output = df_output.reset_index().assign(
        row_id = lambda df: df.apply(lambda df: "{}_{}".format(int(df['cfips']),df['date']), axis='columns'))[['row_id','microbusiness_density']]

    submission = pd.concat((
        df_output,
        sample_submission[~sample_submission.row_id.isin(df_output.row_id)]))

    submission.to_csv(f"data/{date_submission}_{model_name}_local_{local_score}.csv",index=None)
    
    print(f"submission is created for date: {date_submission} model: {model_name} with score: {local_score}")
    display(submission.head())
    return submission
    
