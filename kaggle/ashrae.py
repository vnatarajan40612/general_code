# %% [code]
import pandas as pd
import numpy as np
import category_encoders
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import datetime
import gc

DATA_PATH = "../input/ashrae-energy-prediction/"

# %% [code]
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df



# %% [code]
train_df = pd.read_csv(DATA_PATH + 'train.csv')


# Remove outliers
building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')
weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')

# %% [code]
train_df = reduce_mem_usage(train_df,use_float16=True)
building_df = reduce_mem_usage(building_df,use_float16=True)
weather_df = reduce_mem_usage(weather_df,use_float16=True)

# %% [code]
train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')
train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

# %% [code]
train_df['timestamp'] = pd.to_datetime(train_df.timestamp)
train_df['month'] = train_df.timestamp.dt.month
train_df['day'] = train_df.timestamp.dt.day
train_df['hour'] = train_df.timestamp.dt.hour

train_df = train_df.drop(['timestamp'], axis=1)

# %% [code]
target = np.log1p(train_df["meter_reading"])
train = train_df[['building_id','site_id','primary_use','meter','square_feet','air_temperature','wind_direction','wind_speed','month','day','hour']]

# %% [code]
le = LabelEncoder()
train.primary_use = le.fit_transform(train.primary_use)

# %% [code]
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(train)
train = imp.transform(train)

# %% [code]
kf = KFold(n_splits=3)
models = []
for train_index, val_index in kf.split(train):
    train_features = train[train_index]
    train_target = target[train_index]
    
    val_features = train[val_index]
    val_target = target[val_index]
    
    model = LinearRegression()    
    model.fit(train_features, train_target)
    models.append(model)
    val_pred = model.predict(val_features)
    print(np.sqrt(mean_squared_error(val_target, val_pred)))
    del train_features, train_target, val_features, val_target

# %% [code]
del train, target

# %% [code]
test_df = pd.read_csv(DATA_PATH + 'test.csv')
row_ids = test_df["row_id"]
test_df.drop("row_id", axis=1, inplace=True)
test_df = reduce_mem_usage(test_df)

# %% [code]
test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')
del building_df
# %% [code]
weather_df = pd.read_csv(DATA_PATH + 'weather_test.csv')
weather_df = reduce_mem_usage(weather_df)

# %% [code]
test_df = test_df.merge(weather_df,how='left',on=['timestamp','site_id'])
del weather_df

# %% [code]
test_df['timestamp'] = pd.to_datetime(test_df.timestamp)
test_df['month'] = test_df.timestamp.dt.month
test_df['day'] = test_df.timestamp.dt.day
test_df['hour'] = test_df.timestamp.dt.hour

test_df = test_df.drop(['timestamp'], axis=1)

# %% [code]
test = test_df[['building_id','site_id','primary_use','meter','square_feet','air_temperature','wind_direction','wind_speed','month','day','hour']]
del test_df
test.head()

# %% [code]
le = LabelEncoder()
test.primary_use = le.fit_transform(test.primary_use)

# %% [code]
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(test)
test = imp.transform(test)


# %% [code]
results = 0
for model in models:
    results += np.expm1(model.predict(test))/ len(models)
    del model
    

# %% [code]
del test, models


# %% [code]
results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
del row_ids,results
results_df.to_csv("submission_test_1.csv", index=False)
