from joblib import load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', self.numerical_transformer, self.numerical_cols),
            ('cat', self.categorical_transformer, self.categorical_cols)
        ])

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        # Transform data
        return self.preprocessor.transform(X)

data=pd.read_csv('data_final.csv')
target = 'Price'
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(columns=[target]), data[target],
                                                    test_size=0.2, random_state=42, stratify=None)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = DataPreprocessor(numerical_cols, categorical_cols)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X_train, Y_train)

def preprocess_input(data,preprocessor=preprocessor):
    col=["Brand","Model","Year","Mileage","Engine","Fuel_type","Transmission","Ext_col","Int_col","Accident","Clean_title"]
    X=pd.DataFrame([data],columns=col)
    return X

def model(data,model_pipeline=model_pipeline):
    ans=model_pipeline.predict(data)
    return ans[0]
