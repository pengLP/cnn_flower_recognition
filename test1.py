import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# index_col=0  将第0列作为行索引
train_df = pd.read_csv('C:\\Users\\Lp\\Desktop\\data\\train.csv', index_col=0)
test_df = pd.read_csv('C:\\Users\\Lp\\Desktop\\data\\test.csv', index_col=0)
print(train_df.head())
print(train_df.shape)

prices = pd.DataFrame({'price': train_df['SalePrice'], 'log(price+1)': np.log1p(train_df['SalePrice'])})
prices.hist()  # 画图 看一下标签是否平滑
plt.show()

y_train = np.log1p(train_df.pop('SalePrice'))  # 将原来的标签删除 剩下log(price+1)列的数据
all_df = pd.concat((train_df, test_df), axis=0)  # 将train_df, test_df合并

# 有些数据的取值只有四五，或者可数个。这类数据我们转为one_hot编码

# 发现MSSubClass值应该是分类值
print(all_df['MSSubClass'].dtypes)  # int64
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
print(all_df['MSSubClass'].value_counts())


print(pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())


all_dummy_df = pd.get_dummies(all_df)


print(all_dummy_df.isnull().sum().sort_values(ascending=False).head())

# 我们用均值填充
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

# 再检查一下是否有缺失值
print(all_dummy_df.isnull().sum().sum())

# 先找出数字型数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print(numeric_cols)

# 对其标准化
numeric_col_mean = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols]-numeric_col_mean) / numeric_col_std

# 将合并的数据此时进行拆分  分为训练数据和测试数据
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X_train = dummy_train_df.values
X_test = dummy_test_df.values

# from sklearn.ensemble import RandomForestRegressor
#
# max_features = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
# test_scores = []
# for max_feat in max_features:
#     print("======================")
#     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(max_features, test_scores)
# plt.title('Max Features vs CV Error')
# plt.show()

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")
# 用sklearn自带的cross validation方法来测试模型
params = [1, 2, 3, 4, 5, 6]
test_scores = []
for param in params:
    print("=====+++++++++++++++++++++++")
    clf = XGBRegressor(max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title("max_depth vs CV Error")
plt.show()
