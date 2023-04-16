import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

# EDA 파트

data = pd.read_csv('data/따릉이예측/train.csv', encoding='utf-8')
t_data = pd.read_csv('data/따릉이예측/test.csv', encoding='utf-8')
df = pd.DataFrame(data)
t_df = pd.DataFrame(t_data)

print(df.shape)
print(df.head(5))

df2 = pd.DataFrame({
    'name': ['kwon', 'park', 'kim'],
    'age': [30, np.nan, 19],
    'class': [np.nan, np.nan, 1]
})

print(df2.isnull().sum())

# 전처리 파트

print('df_info: ', df.info())
# df2 = df2.dropna(axis=0)
print('결측치 제거 전: ', df2)
df2 = df2.fillna(df2.mean(numeric_only=True))
print('결측치 변경 후: ', df2)


# 모델링 파트
# 의사결정나무란?: 결정 트리는 의사결정규칙과 그 결과들을 트리 구조로 도식화한 의사결정지원 도구의 일종

dT_model = DecisionTreeClassifier()

x_train = df.drop(['count'], axis=1)
x_train = x_train.fillna(x_train.mean(numeric_only=True))
y_train = df['count']
# print('x_train: ',x_train)
# print('y_train: ',y_train)

dT_model.fit(x_train, y_train)

t_df = t_df.fillna(t_df.mean(numeric_only=True))

result = dT_model.predict(t_df)
print(t_df)
print(result)

rst = pd.read_csv('data/따릉이예측/submission.csv', encoding='utf-8')

rst = pd.DataFrame(rst)
rst['count'] = result

# rst.to_csv('data/따릉이예측/result.csv', index=False)



