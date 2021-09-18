import pandas as pd
import numpy as np

import os

class Preprocessor:
    def __init__(self):
        pass
    
    def load(self, inp_dir):
        self._df_org_category_names = pd.read_csv(inp_dir+'category_names.csv')
        self._df_org_item_categories = pd.read_csv(inp_dir+'item_categories.csv')
        self._df_org_sales_history = pd.read_csv(inp_dir+'sales_history.csv', parse_dates=['日付'])
        self._df_org_test = pd.read_csv(inp_dir+'test.csv')
    
    def createData(self, beginDate, endDate, purposeDate):
        df_products = self._df_org_item_categories.copy()
        df_products = df_products.merge(self._df_org_category_names, on='商品カテゴリID')

        df_train = self._df_org_sales_history[['日付', '店舗ID', '商品ID', '売上個数']]
        df_train = df_train.loc[beginDate:endDate]
        df_train = df_train.set_index(['日付', '店舗ID', '商品ID'])
        df_train = df_train.groupby(
            [
                pd.Grouper(level='日付', freq='M'),
                pd.Grouper(level='店舗ID'),
                pd.Grouper(level='商品ID')
            ]
        ).sum()
        df_train = df_train.reset_index(['店舗ID', '商品ID'])
        df_train['year'] = df_train.index.year
        df_train['month'] = df_train.index.month

        print('df_train:\n{}'.format(df_train))

        df_test = self._df_org_test.copy()
        df_test['year'] = pd.to_datetime(purposeDate).year
        df_test['month'] = pd.to_datetime(purposeDate).month

        df_test = df_test.merge(df_products, on='商品ID')

        df_test.set_index('index', inplace=True)
        df_test.drop('商品カテゴリ名', axis=1, inplace=True)

        df_test = df_test.reindex(columns=df_train.columns)
        df_test.drop('売上個数', axis=1, inplace=True)

        print('df_test:\n{}'.format(df_test))

        return df_train.drop('売上個数', axis=1), df_train['売上個数'], df_test