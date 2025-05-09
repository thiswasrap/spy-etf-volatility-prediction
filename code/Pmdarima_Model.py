from tqdm.notebook import trange, tqdm
import pandas as pd
import numpy as np
import time
import datetime
from functools import reduce
from pathlib import Path
import pickle as pkl
from multiprocessing import cpu_count, Pool
import pmdarima as pm
from pmdarima import pipeline
from pmdarima.metrics import smape
from sklearn.metrics import mean_squared_error as mse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal
import mplfinance as mpl
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from warnings import catch_warnings, filterwarnings

# Ensure proper matplotlib and seaborn configuration
plt.style.use('ggplot')
sns.set_theme(style="darkgrid")

# Global font settings for plotting
font = {'family': 'sans-serif', 'sans-serif': 'Tahoma', 'weight': 'normal', 'size': '16'}
matplotlib.rc('font', **font)

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)

# Paths and calendar configurations
TOP = Path(__file__ + '../../..').resolve()
NYSE = mcal.get_calendar('NYSE')
CBD = NYSE.holidays()

class Pmdarima_Model:
    def __init__(self, df, data_name, n, periods, freq, train_size=80, trend='c', with_intercept='auto',
                 order=(0,1,0), s_order=(0,0,0), seas=0, fit_seas=False, f_seas=252, k=4, 
                 estimate_diffs=False, impute=False, AA_d=None, AA_D=None, date=True, fourier=True,
                 box=False, log=False, verbose=1):
        try:
            assert(type(df) in (pd.Series, pd.DataFrame)), "Data is not of type Pandas Series or DataFrame."
            assert(type(df.index) == pd.DatetimeIndex), "Data index is not of type Pandas DatetimeIndex."
        except AssertionError as e:
            print(e)
            raise

        self.df = pd.DataFrame(df) if type(df) == pd.Series else df
        if impute:
            self.df = df.interpolate()
        self.hist_dates_df = pd.DataFrame(self.df.index, columns=['date'])
        self.train_size = train_size
        self.df_train, self.df_test = pm.model_selection.train_test_split(self.df, train_size=self.train_size/100)
        self.dates = df.index
        self.length = df.shape[0]
        self.data_name = data_name
        self.ts = data_name.replace(' ', '_')
        self.timeframe = f'{n} {periods.title()}'
        self.tf = f'{n}{periods[0].upper()}'
        self.freq = freq
        self.f = freq.split()[0] + freq.split()[1][0].upper()
        self.m = seas
        self.f_m = f_seas
        self.k = k
        self.estimate_diffs = estimate_diffs
        self.p, self.d, self.q = order
        self.fit_seas = fit_seas
        self.P, self.D, self.Q = s_order
        self.t = trend
        self.n_diffs = AA_d
        self.ns_diffs = AA_D
        if self.estimate_diffs:
            self.__estimate_diffs()
        self.with_intercept = with_intercept
        self.mod_order = f'({self.p}, {self.d}, {self.q})[\'{self.t}\']'
        self.date = date
        self.fourier = fourier
        self.box = box
        self.log = log
        self.__train_test_split_dates()
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        self.mod_params, self.mod_params_df, self.mod_pipe = self.__reset_mod_params('adhoc')
        self.y_hat = None
        self.conf_ints = None
        self.AIC = None
        self.RMSE = np.inf
        self.RMSE_pc = np.inf
        self.SMAPE = np.inf
        self.GS_first_mod = True
        self.mod_CV_filepath = f'{TOP}/model_CV_scores/{self.ts}_{self.tf}_{self.f}.csv'
        if verbose:
            print('Successfully created instance of Class Pmdarima_Model.')

    def __estimate_diffs(self):
        kpss_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='adf', max_d=6)
        self.n_diffs = max(adf_diffs, kpss_diffs)

        if self.fit_seas:
            ocsb_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ocsb', max_D=6)
            ch_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ch', max_D=6)
            self.ns_diffs = max(ocsb_diffs, ch_diffs)

    def __train_test_split_dates(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.__split_df_dates(self.df_train, self.df_test)

    def __split_df_dates(self, train, test):
        X_train = pd.DataFrame(train.index)
        y_train = train.values
        X_test = pd.DataFrame(test.index, index=range(X_train.size, self.length))
        y_test = test.values
        return X_train, y_train, X_test, y_test

    def __reset_mod_params(self, init=None):
        if init:
            mod_params, mod_params_df, mod_pipe = self.__setup_mod_params(self.p, self.d, self.q, self.t,
                                                                          self.P, self.D, self.Q, self.m,
                                                                          self.with_intercept, self.f_m, self.k,
                                                                          self.date, self.fourier, self.box,
                                                                          self.log, func='adhoc', verbose=1)
            return mod_params, mod_params_df, mod_pipe
        else:
            return None, None, None

    @staticmethod
    def __unpickle_model(ts, tf, f, func='GS'):
        pkl_filepath = Pmdarima_Model.__get_pkl_filepath(ts, tf, f, func=func)
        mod_file = open(pkl_filepath, 'rb')
        mod_data = pkl.load(mod_file)
        mod_file.close()
        return mod_data

    @staticmethod
    def __get_pkl_filepath(ts, tf, f, func='GS'):
        return f'{TOP}/models/{ts}_{tf}_{f}_{func}_best_model.pkl'

    def __pickle_model(self, func='AA', verbose=1):
        def __pickle_it(params, pipe, params_df, scores, results, func_type='adhoc', verbose=1):
            mod_file = open(pkl_filepath, 'wb')
            pkl.dump((params, pipe, params_df, scores, results), mod_file)
            mod_file.close()

        scores = (self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE)
        results = (self.y_hat, self.conf_ints)
        if func == 'AA':
            func_type = 'AutoARIMA'
            params = self.AA_best_params
            pipe = self.AA_mod_pipe
            params_df = self.AA_best_mod_params_df
        elif func == 'GS':
            func_type = 'GridSearchCV'
            params = self.GS_best_params
            pipe = self.GS_best_mod_pipe
            params_df = self.GS_best_mod_params_df
        else:
            func_type = 'adhoc'
            params = self.mod_params
            pipe = self.mod_pipe
            params_df = self.mod_params_df

        pkl_filepath = Pmdarima_Model.__get_pkl_filepath(self.ts, self.tf, self.f, func=func)

        if os.path.exists(pkl_filepath):
            mod_data = Pmdarima_Model.__unpickle_model(self.ts, self.tf, self.f, func=func)
            if self.RMSE < mod_data[3][2]:
                __pickle_it(params, pipe, params_df, scores, results, func_type, verbose)
        else:
            mod_file = open(pkl_filepath, 'wb')
            __pickle_it(params, pipe, params_df, scores, results, func_type, verbose)
            print(f'Saved best {func_type} model as {pkl_filepath}.') if verbose else None
        return

    def run_stepwise_CV(self, model=None, func='AA', dynamic=False, verbose=1, visualize=True, return_conf_int=True):
        model_str = ''
        if func == 'AA':
            if not model:
                model = self.AA_mod_pipe
            mod_params = self.AA_best_params
            mod_params_df = self.AA_best_mod_params_df
        elif func == 'GS':
            if not model:
                model = self.GS_best_mod_pipe
            mod_params = self.GS_best_params
            mod_params_df = self.GS_best_mod_params_df
        else:
            mod_params_df = self.mod_params_df
            if not model:
                model = self.mod_pipe

        if verbose:
            print(f'Starting step-wise cross-validation for {model_str}...')

        y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic, verbose=verbose)
        self.y_hat = y_hat
        self.conf_ints = conf_ints
        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=model, verbose=verbose)

        mod_params_df['Scored'].values[0] = True
        mod_params_df['AIC'].values[0] = '%.4f' % (self.AIC)
        mod_params_df['RMSE'].values[0] = '%.4f' % (self.RMSE)
        mod_params_df['RMSE%'].values[0] = '%.4f' % (self.RMSE_pc)
        mod_params_df['SMAPE'].values[0] = '%.4f' % (self.SMAPE)
        mod_params_df['CV_Time'].values[0] = '%.4f' % (self.end-self.start)

        if func == 'AA':
            self.AA_best_mod_params_df = mod_params_df
            self.__pickle_model(func='AA', verbose=verbose)
        if func == 'GS':
            self.GS_best_mod_params_df = mod_params_df
            self.__pickle_model(func='GS', verbose=verbose)
        if func == 'adhoc':
            self.mod_params_df = mod_params_df
            self.__pickle_model(func='adhoc', verbose=verbose)

        index = csv_write_data(self.mod_CV_filepath, mod_params_df, verbose=verbose)
        print()

        if visualize:
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func=func)

        return self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE

    def run_auto_pipeline(self, show_summary=False, visualize=False, return_conf_int=True, verbose=1):
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        self.__run_auto_pipeline(show_summary=show_summary, return_conf_int=return_conf_int, verbose=verbose)
        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=self.AA_mod_pipe, verbose=verbose)

        if visualize:
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func='AA')

        return self.AA_mod_pipe

    def plot_test_predict(self, y_hat=None, conf_ints=True, ylabel=None, fin=True, func='AA'):
        if ylabel is None:
            ylabel = self.data_name
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.X_train, self.y_train, color='blue', alpha=0.5, label='Training Data')
        ax.plot(self.X_test, y_hat, color='green', marker=',', alpha=0.7, label='Predicted')
        ax.plot(self.X_test, self.y_test, color='magenta', alpha=0.3, label='Actual')
        if conf_ints:
            conf_int = np.asarray(self.conf_ints)
            ax.fill_between(self.X_test.date, conf_int[:, 0], conf_int[:, 1], alpha=0.5, color='orange', label="Confidence Intervals")
        ax.legend(loc='upper left', borderaxespad=0.5, prop={"size":16})
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Test vs Predict with Confidence Intervals\n', size=26)
        ax.set_ylabel(ylabel, size=18)
        ax.set_xlabel(ax.get_xlabel(), size=18)
        ax.tick_params(axis='y', size=4, width=1.5, labelsize=16)
        ax.tick_params(axis='x', size=4, width=1.5, labelsize=16)
        ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=24)
        plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict.png')

    def plot_forecast_conf(self, ohlc_df=None, ohlc_fc_df=None, hist_df=None, y_hat=None, y_hat_df=None, conf_ints=True,
            lookback=0, ylabel=None, fin=False, all_ohlc=False, func='GS'):
        days_fc = self.days_fc
        dates = self.df_with_fc.index
        hist_ind = np.arange(self.df.shape[0])
        fc_ind = np.arange(self.df.shape[0], self.df.shape[0]+days_fc)
        if lookback:
            dates = dates[-lookback-days_fc:]
            hist_ind = np.arange(lookback)
            fc_ind = np.arange(lookback, lookback+days_fc)
            num_months = np.floor(dates.shape[0]/21)
            numticks = int(num_months+1)
            while numticks > 21:
                numticks = int(np.ceil(num_months/2)+1)

        ylabel = self.data_name
        df_with_fc = self.df_with_fc.loc[dates]
        conf_filename = ''
        conf_title = ''
        if fin:
            ohlc_df = ohlc_df[-lookback:]
            all_ohlc_df = ohlc_df.append(ohlc_fc_df)
        else:
            hist_df = hist_df[-lookback:]

        fig, ax = plt.subplots(figsize=(20, 12))
        if fin:
            if not all_ohlc:
                mpl.plot(ohlc_df, type='candle', style="yahoo", ax=ax)
                ax.plot(fc_ind, y_hat_df, 'g.', markersize=7.5, alpha=0.7, label='Forecast')
            if all_ohlc:
                mpl.plot(all_ohlc_df, type='candle', style="yahoo", ax=ax)
        else:
            sns.lineplot(x=hist_ind, y=hist_df, color='blue', alpha=0.5, label='Historical')
            sns.scatterplot(x=fc_ind, y=y_hat_df.iloc[:, 0], color='g', s=15, alpha=1, label='Forecast')
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")

        ax.set_ylim(get_y_lim(df_with_fc.iloc[:, 0]))
        equidate_ax(fig, ax, df_with_fc.index.date, days_fc=days_fc)
        if lookback:
            x_tick_locator = ticker.LinearLocator(numticks=numticks)
            ax.get_xaxis().set_major_locator(x_tick_locator)
        if conf_ints:
            conf_filename = '_Conf'
            conf_title = ' with Confidence Intervals'
            conf_int = np.asarray(self.fc_conf_ints)
            if fin:
                ax.fill_between(fc_ind, conf_int[:, 0], conf_int[:, 1], alpha=0.3, color='orange', label="Confidence Intervals")
            else:
                ax.fill_between(fc_ind, conf_int[:, 0], conf_int[:, 1], alpha=0.3, facecolor='orange', label="Confidence Intervals")

        ax.set_ylabel(f'{ylabel} (USD)', size=20)
        fig.subplots_adjust(top=0.92)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Historical vs Forecast{conf_title}\n', size=24)
        ax.legend(loc='upper left', borderaxespad=0.5, prop={"size":20})

        if func == 'AA':
            ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=24)
            plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=24)
            plt.savefig(f'{TOP}/images/GridSearch/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=24)
            plt.savefig(f'{TOP}/images/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')

    def __get_model_scores(self, y_test, y_hat, y_train, model, verbose=0, debug=False):
        try:
            AIC = model.named_steps['arima'].aic()
        except AttributeError:
            AIC = model.named_steps['arima'].model_.aic()
        RMSE = mse(y_test, y_hat, squared=False)
        RMSE_pc = 100 * RMSE / y_train.mean()
        SMAPE = smape(y_test, y_hat)
        if verbose:
            print("Test AIC: %.3f" % AIC)
            print("Test RMSE: %.3f" % RMSE)
            print("This is %.3f%% of the avg observed value." % RMSE_pc)
            print("Test SMAPE: %.3f%%\n" % SMAPE)
        if debug:
            print("AIC: %.3f | RMSE: %.3f | RMSE%%=%.3f%% | SMAPE %.3f%%" % (AIC, RMSE, RMSE_pc, SMAPE))
        return AIC, RMSE, RMSE_pc, SMAPE
