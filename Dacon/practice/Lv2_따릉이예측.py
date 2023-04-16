import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('data/따릉이예측/train.csv', encoding='utf-8')
t_data = pd.read_csv('data/따릉이예측/test.csv', encoding='utf-8')
df = pd.DataFrame(data)
t_df = pd.DataFrame(t_data)

rf_model = RandomForestRegressor(criterion='mse')
# 결측치 보간법으로 대체

df = df.interpolate(inplace=True)


