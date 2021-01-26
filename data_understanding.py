import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
from scipy import stats


def load_csv(p1, p2):
    return pd.read_csv(p1), pd.read_csv(p2)


def histogram_of_prices():
    warnings.filterwarnings(action="ignore")

    train, test = load_csv('train.csv', 'test.csv')
    data_w = train.copy()
    data_w.columns = data_w.columns.str.replace(' ', '')  # Replacing the white spaces in columns' names

    (mu, sigma) = norm.fit(data_w['SalePrice'])
    shap_t, shap_p = stats.shapiro(data_w['SalePrice'])

    print("Skewness: %f" % abs(data_w['SalePrice']).skew())
    print("Kurtosis: %f" % abs(data_w['SalePrice']).kurt())
    print("Shapiro_Test: %f" % shap_t)
    print("Shapiro_Test: %f" % shap_p)

    plt.figure(figsize=(12, 6))
    sns.distplot(data_w['SalePrice'], kde=True, hist=True, fit=norm)
    plt.title('SalePrice distribution vs Normal Distribution', fontsize=13)
    plt.xlabel("House's sale Price in $", fontsize=12)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.savefig('hist_prices.png')
    plt.show()


def correlation_matrix():
    train, test = load_csv('train.csv', 'test.csv')
    data_w = train.copy()
    data_w.columns = data_w.columns.str.replace(' ', '')

    f, ax = plt.subplots(figsize=(30, 25))
    mat = data_w.corr('pearson')
    mat = round(mat, 3)
    mask = np.triu(np.ones_like(mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('histogram.png', dpi=300)
    plt.show()


def plots():
    train, test = load_csv('train.csv', 'test.csv')
    data_w = train.copy()
    data_w.columns = data_w.columns.str.replace(' ', '')

    # OverallQual - Sale Price [Pearson = 0.8]
    figure, ax = plt.subplots(1, 3, figsize=(20, 8))
    sns.stripplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[0])
    sns.violinplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[1])
    sns.boxplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[2])
    plt.savefig('overallQual.png')
    plt.show()

    # TotRmsAbvGrd - Sale Price [Pearson = 0.53]
    figure, ax = plt.subplots(1, 3, figsize=(20, 8))
    sns.stripplot(data=data_w, x='TotRmsAbvGrd', y='SalePrice', ax=ax[0])
    sns.violinplot(data=data_w, x='TotRmsAbvGrd', y='SalePrice', ax=ax[1])
    sns.boxplot(data=data_w, x='TotRmsAbvGrd', y='SalePrice', ax=ax[2])
    plt.savefig('TotRmsAbvGrd.png')
    plt.show()

    # GrLivArea - Sale Price [corr = 0.71]
    Pearson_GrLiv = 0.71
    plt.figure(figsize=(12, 6))
    sns.regplot(data=data_w, x='GrLivArea', y='SalePrice', scatter_kws={'alpha': 0.2})
    plt.title('GrLivArea vs SalePrice', fontsize=12)
    plt.legend(['$Pearson=$ {:.2f}'.format(Pearson_GrLiv)], loc='best')
    plt.savefig('GrLivArea.png')
    plt.show()

    # TotalBsmtSF - Sale Price [corr = 0.61]
    Pearson_TBSF = 0.61
    plt.figure(figsize=(12, 6))
    sns.regplot(data=data_w, x='TotalBsmtSF', y='SalePrice', scatter_kws={'alpha': 0.2})
    plt.title('TotalBsmtSF vs SalePrice', fontsize=12)
    plt.legend(['$Pearson=$ {:.2f}'.format(Pearson_TBSF)], loc='best')
    plt.savefig('TotalBsmtSF.png')
    plt.show()

    # YearBuilt vs SalePrice [corr = 0.52]
    Pearson_YrBlt = 0.56
    plt.figure(figsize=(12, 6))
    sns.regplot(data=data_w, x='YearBuilt', y='SalePrice', scatter_kws={'alpha': 0.2})
    plt.title('YearBuilt vs SalePrice', fontsize=12)
    plt.legend(['$Pearson=$ {:.2f}'.format(Pearson_YrBlt)], loc='best')
    plt.savefig('TotalBsmtSF.png')
    plt.show()


def median():
    train, test = load_csv('train.csv', 'test.csv')
    data_w = train.copy()
    data_w.columns = data_w.columns.str.replace(' ', '')

    plt.figure(figsize=(10, 5))
    sns.barplot(x='YrSold', y="SalePrice", data=data_w, estimator=np.median)
    plt.title('Median of Sale Price by Year', fontsize=13)
    plt.xlabel('Selling Year', fontsize=12)
    plt.ylabel('Median of Price in $', fontsize=12)
    plt.show()


if __name__ == '__main__':
    median()
