import pandas as pd
import numpy as np

import os

import global_valiables

class Preprocessor:
    def __init__(self):
        pass
    
    def load(self, inp_dir):
        self._df_org_category_names = pd.read_csv(inp_dir+'category_names.csv')
        self._df_org_item_categories = pd.read_csv(inp_dir+'item_categories.csv')
        self._df_org_sales_history = pd.read_csv(inp_dir+'sales_history.csv', parse_dates=['日付'])
        self._df_org_test = pd.read_csv(inp_dir+'test.csv', index_col=0)
    
    def createData(self, beginDate, endDate, purposeDate):
        df_products = self._df_org_item_categories.copy()
        df_products = df_products.merge(self._df_org_category_names, on='商品カテゴリID')

        ####################
        # trainデータ編集

        ## sales_historyから必要情報を抽出
        df_train = self._df_org_sales_history[['日付', '店舗ID', '商品ID', '売上個数']]
        ## 売上個数を日別データから月別データに変換
        df_train = df_train.set_index(['日付', '店舗ID', '商品ID'])
        df_train = df_train.groupby(
            [
                pd.Grouper(level='日付', freq='M'),
                pd.Grouper(level='店舗ID'),
                pd.Grouper(level='商品ID')
            ]
        ).sum()
        df_train = df_train.reset_index(['店舗ID', '商品ID'])

        ###################
        # testデータ編集

        ## test.csvをコピー
        df_test = self._df_org_test.copy()
        ## 商品情報を追加
        df_test = df_test.merge(df_products, on='商品ID')
        ## datetime index変更
        df_test['日付'] = pd.to_datetime(purposeDate)
        df_test.set_index('日付', inplace=True)
        ## trainデータの列順に合わせる
        df_test = df_test.reindex(columns=df_train.columns)

        ###################
        # 結合して編集

        ## train, test結合
        df_concat = pd.concat([df_train, df_test], axis=0)
        ## year, month列追加
        df_concat['year'] = df_concat.index.year
        df_concat['month'] = df_concat.index.month

        train = df_concat.loc[beginDate:endDate]
        test = df_concat.loc[purposeDate]

        if global_valiables.verbose:
            print('trainData:\n{}'.format(train))
            print('testData:\n{}'.format(test))

        return train.drop('売上個数', axis=1), train['売上個数'], test.drop('売上個数', axis=1)