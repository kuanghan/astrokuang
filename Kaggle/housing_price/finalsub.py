import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
import seaborn as sns


# -------------------
# Generate features
# -------------------
SUBMISSION_FILE = 'sample_submission.csv'
YSCALER = StandardScaler()


def count_encoding(df, col, dropcol=False):
    """
    Encode each category by its counts as a fraction of the total sample.
    """
    df[col].fillna('None', inplace=True)
    counts = df[col].value_counts()
    # counts = np.log1p(counts)
    frac = counts / float(len(df))
    countscol = df[col].copy()
    for cat in frac.index:
        countscol.replace(cat, round(frac[cat], 2), inplace=True)
    df[col+'_counts'] = countscol
    if dropcol:
        df.drop(col, axis=1, inplace=True)
    return df


def impute_LotFrontage(df, q=20):
    """Impute LotFrontage from LotArea, the easy way."""
    df['LotAreaBins'] = pd.qcut(np.sqrt(df['LotArea']), q=q)
    df['LotFrontage'].fillna(df.groupby('LotAreaBins').transform(np.median)['LotFrontage'],
                            inplace=True)
    df['LotFrontage'] = df['LotFrontage'].astype('float')
    df = df.drop('LotAreaBins', axis=1)
    return df


def gen_features(df):
    """
    Order of procedure:
    - log-transform numerical columns;
    - impute LotFrontage;
    - process a few special columns;
    - drop some columns that have almost no information;
    - log-transform numerical columns;
    - create dummy variables for categorical columns.
    """
    out = df.copy()
    out = out.drop('Id', axis=1)
    if 'SalePrice' in out.columns:
        out = out.drop('SalePrice', axis=1)

    # Generate columns containing counts
    for col in ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour',
        'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'BsmtExposure',
        'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',
        'Fence', 'PavedDrive', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']:
        out = count_encoding(out, col, dropcol=False)

    out['HasFence'] = out['Fence'].notnull().astype('int')

    out['YrSold_from_2007'] = out['YrSold'] - 2007
    out.drop('YrSold', axis=1, inplace=True)

    out['MoSold_x'] = np.sin(out['MoSold'] * 30 * np.pi / 180.)
    out['MoSold_y'] = np.cos(out['MoSold'] * 30 * np.pi / 180.)
    out.drop('MoSold', axis=1, inplace=True)

    # Quality columns
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu',
                'GarageQual', 'GarageCond', 'PoolQC']:
        out[col].replace('Ex', 5, inplace=True)
        out[col].replace('Gd', 4, inplace=True)
        out[col].replace('TA', 3, inplace=True)
        out[col].replace('Fa', 2, inplace=True)
        out[col].replace('Po', 1, inplace=True)
        out[col].fillna(0, inplace=True)  # fill with all zeros for now
        out.loc[:, col] = MaxAbsScaler().fit_transform(
            out[col].values.reshape(-1, 1)).ravel()

    out.loc[:, 'MSSubClass'] = out.loc[:, 'MSSubClass'].astype('str').ravel()
    
    # Numerical features with missing values to be imputed
    out = impute_LotFrontage(out, q=10)   # LotFrontage
    
    # impute missing GarageYrBlt by replacing with YearBuilt
    out['GarageYrBlt'].fillna(out['YearBuilt'], inplace=True)
    # suspicious of the houses remodeled in 1950... set it to YearBuilt if it's 1950
    out['YearRemodAdd'] = np.where(out['YearRemodAdd'] == 1950,
                                   out['YearBuilt'], out['YearRemodAdd'])
    
    # Columns describing year can be converted into age, then normalized
    for col in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
#         out = dummify(out, col, fillna=True, value='None')
        newcol = 'Age' + col
        out[newcol] = out[col] - out[col].min()
        out = out.drop(col, axis=1)

    # CentralAir: 1 or 0
    out['CentralAir'].replace('Y', 1, inplace=True)
    out['CentralAir'].replace('N', -1, inplace=True)

    out['Pool'] = np.where(out['PoolArea'] > 0, 1, 0)
    out['Basement'] = np.where(out['BsmtQual'].notnull(), 1, 0)
    out['Utilities'] = np.where(out['Utilities'] == 'AllPub', 1, 0)
    out['MiscFeature'] = np.where(out['MiscFeature'].notnull(), 1, 0)


    # These columns should be dropped because either there are too many
    # missing values or they are redundant
    out.drop(['Alley', 'MiscVal'],
             axis=1, inplace=True)

    out.loc[:, 'BsmtFinSF'] = out['BsmtFinSF1'] + out['BsmtFinSF2']
    out.loc[:, 'BsmtFinSF'] = out.loc[:, 'BsmtFinSF'].fillna(0)
    out.drop(['BsmtFinSF2'], axis=1, inplace=True)

    # log-transform numerical columns
    numcols = ['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF',
               'LowQualFinSF', 'GrLivArea', 'BsmtFinSF', 'BsmtFinSF1',
               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', #'3SsnPorch',
               'ScreenPorch', 'GarageArea', '1stFlrSF', '2ndFlrSF',
               'LotFrontage', 'AgeYearBuilt', 'AgeYearRemodAdd',
               'AgeGarageYrBlt', '3SsnPorch']
    out.loc[:, numcols] = out.loc[:, numcols].fillna(0)
    outdoor_area = out['WoodDeckSF'] + out['OpenPorchSF'] + out['EnclosedPorch'] + out['3SsnPorch'] + out['ScreenPorch'] + out['PoolArea']
    out['Outdoor_Ratio'] = outdoor_area.astype('float') / out['GrLivArea']
    out.loc[:, numcols] = np.around(np.log1p(out.loc[:, numcols]), 2)
    out.drop(['3SsnPorch', 'PoolArea'], axis=1, inplace=True)

    for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
               'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
               'GarageCars']:
        out.loc[:, col] = out.loc[:, col].fillna(0).ravel()
    out['Garage'] = out['GarageFinish'].notnull().astype('int')

    # Scale numerical columns by MaxAbsScaler
    numcols2 = [x for x in out.columns if out[x].dtype != np.dtype('O')]
    for col in numcols2:
        if col in ['MoSold_x', 'MoSold_y']:
            continue
        out.loc[:, col] = RobustScaler().fit_transform(
            out[col].values.reshape(-1, 1))

    # Inspect the average of each numerical column
    # avg = np.mean(out.loc[:, numcols2], axis=0)
    # print(avg)

    # create dummy variables
    out = pd.get_dummies(out, dummy_na=True, drop_first=False)

    x = out.isnull().sum()
    # print("Columns containing missing data:")
    # print(x[x > 0])
            
    return out

# -----------------------
# Read training data and generate features
# -----------------------
def process_data():
    print('\n' + '-' * 50)
    print('Processing training and test data together...')
    print('-' * 50)

    train = pd.read_csv('train.csv')
    y_train = YSCALER.fit_transform(
        np.log(train['SalePrice']).values.reshape(-1, 1)).ravel()
    test = pd.read_csv('test.csv')
    all_data = pd.concat([train, test])
    all_data = gen_features(all_data)
    x_train = all_data.iloc[:len(train)].values
    x_test = all_data.iloc[len(train):].values
    print("x_train.shape = {}".format(x_train.shape))
    print("x_test.shape = {}".format(x_test.shape))
    return x_train, x_test, y_train, all_data


# -----------------------
# Build regressors and look at CV score
# -----------------------
def compare_CV(x_train, y_train):
    print('\n' + '-' * 50)
    print('Comparing CV Scores...')
    print('-' * 50)
    clf1 = LinearRegression(fit_intercept=True)
    clf2 = Lasso(alpha=1.e-3)  # Lasso performs much worse than simple linear regression (0.545)
    clf3 = Ridge(alpha=25)  # slighly better than simple linear regression
    clf4 = RandomForestRegressor(n_estimators=500)
    clf5 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
        max_depth=3)  # the best performance so far
    
    regressors = {}
    regressors['LinearRegression'] = clf1
    regressors['Lasso'] = clf2
    regressors['Ridge'] = clf3
    regressors['RandomForestRegressor'] = clf4
    regressors['GradientBoostingRegressor'] = clf5
    
    for x in ['Lasso', 'Ridge', 'GradientBoostingRegressor']:
    # for x in ['Lasso', 'Ridge']:
        reg = regressors[x]
        scores = cross_val_score(reg, X=x_train, y=y_train,
                                 scoring='r2', cv=5)
        print("Average score for {} = {:.3f}".format(x, np.mean(scores)))
        print(scores)
        print()

    return regressors

# -----------------------
# Also try xgboost
# -----------------------
def run_xgboost(x_train, y_train):
    print('\n' + '-' * 50)
    print('Running XGBOOST...')
    print('-' * 50)
    kf = KFold(n_splits=5)
    xgb_scores = []
    for train_index, cv_index in kf.split(x_train):
        X_train, X_cv = x_train[train_index], x_train[cv_index]
        Y_train, Y_cv = y_train[train_index], y_train[cv_index]
        dtrain = xgb.DMatrix(X_train, Y_train)
        dcv = xgb.DMatrix(X_cv)
        plist = {'learning_rate': 0.3, 'objective': 'reg:linear',
                 'max_depth': 5, 'subsample': 1, 'lambda': 0.8,
                 'eval_metric': 'rmse', 'nrounds': 500}
        bst = xgb.train(dtrain=dtrain, params=plist)
        Y_pred = bst.predict(dcv)
        xgb_scores.append(r2_score(Y_cv, Y_pred))
    print("XGBOOST mean CV score: {:.3f}".format(np.mean(xgb_scores)))


# -------------------------------
# Write a class for a stacked model
# -------------------------------
def fit_stack(x_train, y_train, x_test, regressors=[]):
    assert len(regressors) > 0, "Please provide models to stack."
    print('\n' + '-' * 50)
    print('Building & fitting a stacked model...')
    print('-' * 50)
    class Stacked(object):
        def __init__(self, models):
            self.models = models
    #         self.stacked_model = Ridge()
            
        def train(self, Xtrain, Ytrain):
            X_mid = []
            for x in self.models:
                x.fit(Xtrain, Ytrain)
                X_mid.append(x.predict(Xtrain))
            self.X_mid = np.column_stack(X_mid)
    #         self.stacked_model.fit(self.X_mid, Ytrain)
            
        def predict(self, Xtest):
            X_mid = []
            for x in self.models:
                X_mid.append(x.predict(Xtest))
            X_mid = np.column_stack(X_mid)
    #         Ytest = self.stacked_model.predict(X_mid)
            Ytest = np.mean(X_mid, axis=1)
            return Ytest
    
    # Check the CV score
    stacked_scores = []
    kf = KFold(n_splits=5)
    for train_index, cv_index in kf.split(x_train):
        SM = Stacked(regressors)
        X_train, X_cv = x_train[train_index], x_train[cv_index]
        Y_train, Y_cv = y_train[train_index], y_train[cv_index]
        SM.train(X_train, Y_train)
        Y_cvpred = SM.predict(X_cv)
        stacked_scores.append(r2_score(Y_cv, Y_cvpred))
    
    print("Mean CV score for the stacked model: {:.3f}".format(np.mean(stacked_scores)))
    
    # Fit the stacked model and make predictions
    SM = Stacked(regressors)
    SM.train(x_train, y_train)
    y_pred = SM.predict(x_test)
    lims = [-4.5, 4.5]

    plt.figure()
    plt.scatter(SM.predict(x_train), y_train)
    plt.plot(lims, lims, ls='--', c='black')
    plt.xlim(*lims)
    plt.ylim(*lims)

    return y_pred

# -----------------------
# Write predictions to a submission file
# -----------------------
def write_submission(y_pred):
    """
    Mainly to test if my submission file has bugs...
    The CV scores didn't change that much from sub12.
    """
    print('\n' + '-' * 50)
    print('Writing submission file No. 15...')
    print('-' * 50)
    
    submission = pd.read_csv(SUBMISSION_FILE)
    saleprice = np.exp(YSCALER.inverse_transform(y_pred))
    saleprice = np.around(saleprice, 2)
    submission['SalePrice'] = saleprice
    submission.to_csv('houseprice_sub15.csv', index=None)
    return saleprice
