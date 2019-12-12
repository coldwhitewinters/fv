# -*- coding: utf-8 -*-

import pandas as pd
from fbprophet import Prophet
import pmdarima as pmd
    
class DecProphet:
    def __init__(self, endog="y", exog=None, ds="ds", yearly=False, time_range=range(6,21)):
        self.endog = endog
        self.exog = exog
        self.ds = ds
        self.yearly = yearly
        self.time_range = time_range
        self.models = {}
        self.forecasts = {}
        
    def prepare(self, df):
        df = df.rename(columns={self.ds:"ds", self.endog:"y"})
        return df
        
    def decompose(self, df):
        df_dec = {}
        df = self.prepare(df)
        for k in self.time_range:
            df_dec[k] = df.loc[df.ds.dt.hour == k, :].reset_index(drop=True)
        return df_dec
        
    def fit(self, df):
        df_dec = self.decompose(df)
        for k in df_dec.keys():
            df_k = df_dec[k]
            m = Prophet(yearly_seasonality=self.yearly, 
                        weekly_seasonality=False, 
                        daily_seasonality=False)
            if self.exog:
                for reg in self.exog:
                    m.add_regressor(reg)
            m.fit(df_k)
            self.models[k] = m
            
    def predict(self, df):
        df_dec = self.decompose(df)
        for k in df_dec.keys():
            df_k = df_dec[k]
            m = self.models[k]
            self.forecasts[k] = m.predict(df_k)
        fcst = pd.concat(self.forecasts.values(), 
                         ignore_index=True).sort_values(by="ds")
        return fcst
      
        
class DecARIMA:
    def init(self, endog="y", exog=None, ds="ds", yearly=False, time_range=range(6,21)):
        self.endog = endog
        self.exog = exog
        self.ds = ds
        self.yearly = yearly
        self.time_range = time_range
        self.models = {}
        self.forecasts = {}
        
    def decompose(self, df, time_range=range(6,21)):
        df_dec = {}
        df = df.set_index(self.ds)
        for k in time_range:
            df_dec[k] = df.loc[df.index.hour == k, :]
        return df_dec

    def fit(self, df):
        df_dec = self.decompose(df)
        for k in df_dec.keys():
            df_k = df_dec[k]
            model = pmd.auto_arima(df_k.y, df_k[self.exog], 
                                   suppress_warnings=True)
            self.models[k] = model

    def predict(self, df):
        df_dec = self.decompose(df)
        for k in df_dec.keys():
            df_k = df_dec[k]
            model = self.models[k]
            part_fcst = model.predict(n_periods=len(df_k), exogenous=df_k[self.exog])
            self.forecasts[k] = pd.Series(part_fcst, index=df_k.index)
        full_fcst = pd.concat(self.forecasts.values(), 
                              ignore_index=False).sort_index()
        return full_fcst


