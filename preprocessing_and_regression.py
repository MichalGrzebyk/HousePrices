import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew, norm
import statsmodels.api as sm
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet


global data


def load_csv(p1, p2):
    return pd.read_csv(p1), pd.read_csv(p2)


def impute_knn(df):
    ttn = df.select_dtypes(include=[np.number])
    ttc = df.select_dtypes(exclude=[np.number])

    cols_nan = ttn.columns[ttn.isna().any()].tolist()  # columns w/ nan
    cols_no_nan = ttn.columns.difference(cols_nan).values  # columns w/n nan

    for col in cols_nan:
        imp_test = ttn[ttn[col].isna()]  # indicies which have missing data will become our test set
        imp_train = ttn.dropna()  # all indicies which which have no missing data
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ttn.loc[ttn[col].isna(), col] = knr.predict(imp_test[cols_no_nan])

    return pd.concat([ttn, ttc], axis=1)


def bin_MSSubClass(x):
    if x > 20:
        x = 21
    return x


def bin_OverallQual(x):
    if x > 6:
        x = 0
    else:
        x = 1
    return x


def bin_exterior(x, col_name):
    global data
    if x != data[col_name].value_counts().index[0] and x != data[col_name].value_counts().index[0]:
        x = 'Other'
    return x


def bin_string(x, col_name):
    global data
    if x != data[col_name].value_counts().index[0]:
        x = 'Other'
    return x


def data_preprocessing(d):
    target = d['SalePrice']
    d2 = d.copy()
    del d2['SalePrice']

    global data
    data = d2.loc[:, d2.columns[d2.isnull().mean() < 0.3]]

    for column in data.columns:
        if len(data[column].value_counts()) > 0.5 * data.shape[0]:
            data = data.drop(column, axis=1)

    for column in data.columns:
        if data[column].value_counts().values[0] > 0.7 * data.shape[0]:
            data = data.drop(column, axis=1)

    cat_cols = []
    num_cols = []

    for column in data.columns:
        if data[column].nunique() < 30:
            cat_cols.append(column)
        else:
            num_cols.append(column)

    cols_final = data.columns

    for column in cols_final:
        if column in cat_cols:
            data[column] = data[column].fillna(value=data[column].value_counts().index[0])
        else:
            data[column] = data[column].fillna(value=data[column].mean())

    data['MSSubClass'] = data['MSSubClass'].apply(bin_MSSubClass)
    data['OverallQual'] = data['OverallQual'].apply(bin_OverallQual)

    data['Exterior1st'] = data['Exterior1st'].apply(bin_exterior, args=['Exterior1st'])
    data['Exterior2nd'] = data['Exterior2nd'].apply(bin_exterior, args=['Exterior2nd'])

    data['LotShape'] = data['LotShape'].apply(bin_string, args=['LotShape'])
    data['HouseStyle'] = data['HouseStyle'].apply(bin_string, args=['HouseStyle'])

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data[cat_cols])
    data_cat = enc.transform(data[cat_cols])
    data_cat.columns = enc.get_feature_names(cat_cols)

    data_cat = pd.DataFrame.sparse.from_spmatrix(data_cat)
    data_cat.columns = enc.get_feature_names(cat_cols)
    data = data.drop(cat_cols, axis=1)
    data = pd.concat([data, data_cat], axis=1)

    scaler = MinMaxScaler()

    for column in num_cols:
        normalized = scipy.stats.yeojohnson(data[column])[0]
        data[column] = scaler.fit_transform(normalized.reshape(-1, 1))

    return data, target


def data_preprocessing2(d):
    target = d['SalePrice']
    data = d.drop(['SalePrice'], axis=1)

    nan = pd.DataFrame(data.isna().sum(), columns=['NaN_sum'])
    nan['feat'] = nan.index
    nan['Perc(%)'] = (nan['NaN_sum'] / 1460) * 100
    nan = nan[nan['NaN_sum'] > 0]
    nan = nan.sort_values(by=['NaN_sum'])
    nan['Usability'] = np.where(nan['Perc(%)'] > 20, 'Discard', 'Keep')

    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YrSold'] = data['YrSold'].apply(str)
    data['MoSold'] = data['MoSold'].apply(str)

    data['Functional'] = data['Functional'].fillna('Typ')
    data['Electrical'] = data['Electrical'].fillna("SBrkr")
    data['KitchenQual'] = data['KitchenQual'].fillna("TA")
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    data["PoolQC"] = data["PoolQC"].fillna("None")
    data["Alley"] = data["Alley"].fillna("None")
    data['FireplaceQu'] = data['FireplaceQu'].fillna("None")
    data['Fence'] = data['Fence'].fillna("None")
    data['MiscFeature'] = data['MiscFeature'].fillna("None")

    for col in ('GarageArea', 'GarageCars'):
        data[col] = data[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        data[col] = data[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('None')

    useless = ['GarageYrBlt', 'YearRemodAdd']
    data = data.drop(useless, axis=1)

    data = impute_knn(data)

    objects = []
    for i in data.columns:
        if data[i].dtype == object:
            objects.append(i)
    data.update(data[objects].fillna('None'))

    data["SqFtPerRoom"] = data["GrLivArea"] / (data["TotRmsAbvGrd"] + data["FullBath"] +
                                               data["HalfBath"] + data["KitchenAbvGr"])

    data['Total_Home_Quality'] = data['OverallQual'] + data['OverallCond']

    data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath']
                               + (0.5 * data['BsmtHalfBath']))

    data["HighQualSF"] = data["1stFlrSF"] + data["2ndFlrSF"]

    # Converting non-numeric predictors stored as numbers into string

    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YrSold'] = data['YrSold'].apply(str)
    data['MoSold'] = data['MoSold'].apply(str)

    # Creating dummy variables from categorical features

    train_test_dummy = pd.get_dummies(data)

    # Fetch all numeric features

    # train_test['Id'] = train_test['Id'].apply(str)
    numeric_features = train_test_dummy.dtypes[train_test_dummy.dtypes != object].index
    skewed_features = train_test_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[skewed_features > 0.5]
    skew_index = high_skew.index

    # Normalize skewed features using log_transformation

    for i in skew_index:
        train_test_dummy[i] = np.log1p(train_test_dummy[i])

    target_log = np.log1p(target)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("qq-plot & distribution SalePrice ", fontsize=15)

    sm.qqplot(target_log, stats.t, distargs=(4,), fit=True, line="45", ax=ax[0])
    sns.distplot(target_log, kde=True, hist=True, fit=norm, ax=ax[1])
    plt.show()

    data = data[['Total_Home_Quality', 'YrSold', 'MSSubClass', 'Total_Bathrooms', 'HighQualSF', 'SqFtPerRoom']]
    data.to_csv('data.csv')
    return data, target


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def cv_rmse(model, train, target_log, kf):
    rmse = np.sqrt(-cross_val_score(model, train, target_log, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


def test_methods():
    warnings.filterwarnings(action="ignore")

    train, test = load_csv('train.csv', 'test.csv')

    train.columns = train.columns.str.replace(' ', '')  # Replacing the white spaces in columns' names

    train_data, train_targets = data_preprocessing(train)

    train_data.to_csv('train_post.csv')

    regr = RandomForestRegressor(random_state=42)

    params = {'max_depth': (4, 6, 8)}
    clf = GridSearchCV(regr, params)
    clf.fit(train_data, train_targets)
    regr = clf.best_estimator_
    regr_score = clf.best_score_

    regr_importance = pd.Series(regr.feature_importances_)
    sorted_importance = regr_importance.sort_values(ascending=False)

    xgb = xgboost.XGBRegressor(random_state=42)

    params = {'max_depth': (3, 4, 5)}
    clf = GridSearchCV(xgb, params)
    clf.fit(train_data, train_targets)
    xgb = clf.best_estimator_
    xgb_score = clf.best_score_
    xgb_importance = xgb.feature_importances_

    en = ElasticNet(random_state=42)

    params = {'selection': ('cyclic', 'random')}
    clf = GridSearchCV(en, params)
    clf.fit(train_data, train_targets)
    en = clf.best_estimator_
    en_score = clf.best_score_
    en_importance = permutation_importance(en, train_data, train_targets, scoring='neg_mean_squared_error').importances_mean

    importance_data = pd.Series(regr_importance * regr_score + xgb_importance * xgb_score + en_importance * en_score)
    importance = pd.Series(data=importance_data.values, index=train_data.columns)
    plot = importance.sort_values(ascending=False).head(20).plot.bar()
    plot.set_title('Best scoring Features')
    plot.set_xlabel('Feature')
    plot.set_ylabel('Sum of scores, weighted by model performance')
    # plt.savefig('overall_best')
    # plt.show()
    # importance2 = importance.sort_values(ascending=False)
    # importance2.to_csv('importance.csv')

    # train_data = train_data[['GarageCars_3', 'OverallQual_1', 'OverallQual_0', 'Fireplaces_0', 'BsmtQual_Ex', 'FullBath_1',
    #                          'KitchenQual_Ex', 'ExterQual_TA', 'KitchenQual_TA', 'BsmtExposure_No', 'GarageFinish_Unf',
    #                          'GarageFinish_Fin', 'BedroomAbvGr_4', 'BsmtFinType1_GLQ', 'HalfBath_1', 'ExterQual_Ex',
    #                          'HalfBath_0', 'Fireplaces_2', 'BsmtExposure_Gd', 'BsmtQual_TA']]

    catb = CatBoostRegressor()
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    score_catb = cv_rmse(catb, train_data, train_targets, kf)
    print('CatBoost: ', score_catb.mean(), score_catb.std())

    gbr = GradientBoostingRegressor()
    score_gbr = cv_rmse(gbr, train_data, train_targets, kf)
    print('GradientBoosting: ', score_gbr.mean(), score_gbr.std())


if __name__ == '__main__':
    test_methods()
