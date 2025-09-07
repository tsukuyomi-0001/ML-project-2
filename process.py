import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb


# ---------------------------- Data Loading ---------------------------- #
def load_dataset(filepath="dataset.csv") -> pd.DataFrame:
    file = Path(filepath)
    if file.exists() and file.is_file():
        try:
            return pd.read_csv(file)
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")
    else:
        raise FileNotFoundError(f"File '{filepath}' does not exist or is not readable.")


# ---------------------------- Feature Engineering ---------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['latlog_ratio'] = df['Lat'] / df['Long']
    df.index = pd.to_datetime(df['Date'])
    return df


def datetime_features_out(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df.columns = ['Date']
    date_parsed = pd.to_datetime(df['Date'])
    df['year'] = date_parsed.dt.year
    df['month'] = date_parsed.dt.month
    df['day'] = date_parsed.dt.day
    return df.drop(columns='Date')


def log_transform(data: pd.Series) -> pd.Series:
    epsilon = 1e-8
    return np.log1p(data + epsilon)


# ---------------------------- Pipelines ---------------------------- #
def build_feature_pipeline():
    obj_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )
    num_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'))
    datetime_pipeline = make_pipeline(FunctionTransformer(datetime_features_out, validate=False))

    return make_column_transformer(
        (obj_pipeline, make_column_selector(dtype_include=object)),
        (num_pipeline, make_column_selector(dtype_include=int)),
        (datetime_pipeline, ['Date'])
    )


def build_label_pipeline():
    return make_pipeline(FunctionTransformer(log_transform, validate=False))


# ---------------------------- Dataset Splitting ---------------------------- #
def split_by_country(df: pd.DataFrame, split_ratio=(0.85, 0.95)):
    train_set, valid_set, test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for country in df['Country/Region'].unique():
        full_data = df[df['Country/Region'] == country]
        n = len(full_data)
        train_end = int(n * split_ratio[0])
        valid_end = int(n * split_ratio[1])

        train_data = full_data.iloc[:train_end]
        valid_data = full_data.iloc[train_end:valid_end]
        test_data = full_data.iloc[valid_end:]

        train_set = pd.concat([train_set, train_data], ignore_index=True)
        valid_set = pd.concat([valid_set, valid_data], ignore_index=True)
        test_set = pd.concat([test_set, test_data], ignore_index=True)

    return train_set, valid_set, test_set


# ---------------------------- Model Training ---------------------------- #
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(eval_metric='rmse')
    model.fit(X_train, y_train)
    return model


def train_best_model(X_train, y_train):
    best_model = xgb.XGBRegressor(
        eval_metric='rmse',
        subsample=0.6,
        colsample_bytree=1.0,
        reg_lambda=0.1,
        reg_alpha=0.01,
        n_estimators=2500,
        max_depth=3,
        learning_rate=0.02,
        gamma=0.2
    )
    best_model.fit(X_train, y_train)
    return best_model


# ---------------------------- Main Execution ---------------------------- #
def main():
    dataset = load_dataset()
    dataset = add_features(dataset)

    pipeline = build_feature_pipeline()
    label_pipeline = build_label_pipeline()

    train_set, valid_set, test_set = split_by_country(dataset)

    X_train = pipeline.fit_transform(train_set)
    y_train = label_pipeline.fit_transform(train_set['Deaths'])

    X_valid = pipeline.transform(valid_set)
    y_valid = label_pipeline.transform(valid_set['Deaths'])

    X_test = pipeline.transform(test_set)
    y_test = label_pipeline.transform(test_set['Deaths'])

    # Initial model
    model = train_model(X_train, y_train)
    pred_log = model.predict(X_valid)
    pred = np.expm1(pred_log)
    rmse_initial = root_mean_squared_error(np.expm1(y_valid), pred)
    print(f"Initial Validation RMSE: {rmse_initial:.2f}")

    # Tuned model
    best_model = train_best_model(X_train, y_train)
    pred_log_best = best_model.predict(X_valid)
    pred_best = np.expm1(pred_log_best)
    rmse_best = root_mean_squared_error(np.expm1(y_valid), pred_best)
    print(f"Tuned Validation RMSE: {rmse_best:.2f}")

    # Final test evaluation
    pred_log_test = best_model.predict(X_test)
    pred_test = np.expm1(pred_log_test)
    rmse_test = root_mean_squared_error(np.expm1(y_test), pred_test)
    print(f"Test RMSE: {rmse_test:.2f}")


if __name__ == "__main__":
    main()
