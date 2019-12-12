import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class DecSMA:
    def __init__(self, endog="y", exog="x", ds="ds", window=3, n_bins=5):
        self.endog = endog
        self.exog = exog
        self.ds = ds
        self.window = window
        self.model = None
        self.kbins = KBinsDiscretizer(n_bins=[n_bins], encode='ordinal')

    def fit(self, df):
        values = df[self.exog].to_numpy().reshape(-1, 1)
        df[self.exog] = self.kbins.fit_transform(values)
        self.model = df[[self.ds, self.endog, self.exog]]

    def predict(self, df):
        values = df[self.exog].to_numpy().reshape(-1, 1)
        df[self.exog] = self.kbins.transform(values)
        for k in df.index:
            h = df.loc[k, self.ds].hour
            cond = df.loc[k, self.exog]
            mask = (self.model[self.ds].dt.hour == h) & (self.model[self.exog] == cond)
            df.loc[k, "yhat"] = self.model.loc[mask, self.endog].tail(self.window).mean()
        return df
