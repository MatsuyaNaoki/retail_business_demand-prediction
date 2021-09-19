import pandas as pd
import numpy as np

import os

import global_valiables

class Preprocessor:
    def __init__(self):
        pass
    
    def load(self, inp_dir):
        """load data

        Args:
            inp_dir (str): input dir name
        """        
        self._df_org_category_names = pd.read_csv(inp_dir+'category_names.csv')
        self._df_org_item_categories = pd.read_csv(inp_dir+'item_categories.csv')
        self._df_org_sales_history = pd.read_csv(inp_dir+'sales_history.csv', parse_dates=['日付'])
        self._df_org_test = pd.read_csv(inp_dir+'test.csv', index_col=0)
    
    def createData(self, beginDate, endDate, purposeDate):
        """create data

        Args:
            beginDate (str): learn begin date
            endDate (str): learn end date
            purposeDate (str): test date

        Returns:
            X_train, y_trian, X_test: dataset
        """        
        df_products = self._df_org_item_categories.copy()
        df_products = df_products.merge(self._df_org_category_names, on='商品カテゴリID')
        purposeYear = pd.to_datetime(purposeDate).year
        purposeMonth = pd.to_datetime(purposeDate).month

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
        df_train['year'] = df_train.index.year
        df_train['month'] = df_train.index.month

        ###################
        # testデータ編集

        ## test.csvをコピー
        df_test = self._df_org_test.copy()
        ## 商品情報を追加
        df_test = df_test.merge(df_products, on='商品ID')
        ## datetime index変更
        df_test['日付'] = pd.to_datetime(purposeDate)
        df_test.set_index('日付', inplace=True)
        df_test['year'] = df_test.index.year
        df_test['month'] = df_test.index.month
        ## trainデータの列順に合わせる
        df_test = df_test.reindex(columns=df_train.columns)
        
        ###################
        # 結合して編集

        ## train, test結合
        df_concat = pd.concat([df_train, df_test], axis=0)
        ## 月ブロック追加
        gp_time = df_train.groupby(['year', 'month']).count().reset_index()[['year', 'month']]
        gp_time['month_block'] = list(range(len(gp_time)))
        gp_time = gp_time.append({'year':purposeYear, 'month': purposeMonth, 'month_block': gp_time.tail(1)['month_block'].values[0]+2}, ignore_index=True)
        df_concat = df_concat.merge(gp_time, on=['year', 'month'])

        ##################
        # 特徴量エンジニアリング
        
        ## ラグ特徴量追加
        lags = [1,2,3,4,5,6,12]
        df_feat = df_concat.copy()
        for lag in lags:
            df_lag = df_concat[['店舗ID', '商品ID', '売上個数', 'month_block']]
            df_lag.loc[:,'month_block'] = df_lag.loc[:,'month_block']+lag
            df_lag = df_lag.rename(columns={'売上個数': '売上個数_def_'+str(lag)+'month'})
            df_feat = df_feat.merge(df_lag, on=['month_block', '店舗ID', '商品ID'], how='left')
        df_feat.drop('month_block', axis=1, inplace=True)

        ## カテゴリID追加
        # df_feat = df_feat.merge(self._df_org_item_categories, on='商品ID', how='left')
        
        test = df_feat[(df_feat['year']==purposeYear) & (df_feat['month']==purposeMonth)]
        train = df_feat[~((df_feat['year']==purposeYear) & (df_feat['month']==purposeMonth))]

        if global_valiables.verbose:
            print('trainData:\n{}'.format(train))
            print('testData:\n{}'.format(test))

        return train.drop('売上個数', axis=1), train['売上個数'], test.drop('売上個数', axis=1)