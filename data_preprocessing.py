import os
import glob
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import kpss
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_and_concatenate(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    # For each file, read it into a DataFrame and add a new column 'VM'
    df_from_each_file = (pd.read_csv(f, sep=';\t').assign(VM=os.path.basename(f).split('.')[0]) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file)
    return concatenated_df

df_july = load_and_concatenate('rnd/2013-7/')
df_august = load_and_concatenate('rnd/2013-8/')
df_september = load_and_concatenate('rnd/2013-9/')

concatenated_df = pd.concat([df_july, df_august, df_september])

concatenated_df.head()
concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit = 's')
concatenated_df.apply(pd.to_numeric, errors='ignore')

concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)
concatenated_df['month']=concatenated_df.Timestamp.dt.month
concatenated_df['day']=concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp',inplace=True)
concatenated_df = concatenated_df.fillna(method='ffill')
concatenated_df.head()
hourlydat = concatenated_df.resample('H').sum()
hourlydat.to_csv('hourlydata.csv')
hourlydat.head()

concatenated_df['start'] = concatenated_df.index
concatenated_df['target'] = concatenated_df['CPU usage [MHZ]']

df2 = concatenated_df.groupby('VM').resample('1min')['target'].mean().to_frame()

df2.head()
df2 = df2.fillna(method='ffill')
df2.to_csv('df2.csv')
df3 = concatenated_df.groupby('VM').resample('1min')['CPU capacity provisioned [MHZ]'].mean().to_frame()
df3 = df3.fillna(method='ffill')
df3.head()
df3.to_csv('df3.csv')



'''CPU Capacity Provisioning and Usage Analysis'''
overprovision = pd.DataFrame(hourlydat['CPU usage [MHZ]'])
overprovision['CPU capacity provisioned'] = pd.DataFrame(hourlydat['CPU capacity provisioned [MHZ]'])

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
xy_font_size = 14
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=overprovision.index, y='CPU usage [MHZ]',
             data=overprovision, ax=ax, label='Actual CPU Usage [MHz]', color='steelblue', linewidth=2.5)
sns.lineplot(x=overprovision.index, y='CPU capacity provisioned',
             data=overprovision, ax=ax, label='CPU Capacity Provisioned [MHz]', color='tomato', linewidth=2.5)
ax.set_ylabel((r'CPU Frequency [MHz]'), fontsize=xy_font_size, labelpad=9)
ax.set_xlabel('Time', fontsize=xy_font_size, labelpad=9)
ax.legend(loc='best')
ax.ticklabel_format(axis='y', style='sci', scilimits=(1,6))
ax.set_ylim(0, None)
plt.savefig('dataset_cpu_usage.png', dpi=700)
plt.show()


'''KPSS stationarity test'''
def kpss_test(timeseries):
    statistic, p_value, n_lags, critical_values = kpss(timeseries, regression='ct')
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
kpss_test(overprovision['CPU usage [MHZ]'])


'''Data pre-processing for DeepAR and Chronos'''
files = glob.glob(os.path.join('rnd/2013-7', "*.csv"))
files_first200 = files[:150]
dfs = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files_first200]
df = pd.concat(dfs, ignore_index=True)

files2 = glob.glob(os.path.join('rnd/2013-8', "*.csv"))
files2_first200 = files2[:150]
dfs2 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files2_first200]
df2 = pd.concat(dfs2, ignore_index=True)

files3 = glob.glob(os.path.join('rnd/2013-9', "*.csv"))
files3_first200 = files3[:150]
dfs3 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files3_first200]
df3 = pd.concat(dfs3, ignore_index=True)

newdat = pd.concat([df, df2])
concatenated_df = pd.concat([newdat, df3])
concatenated_df.head()

concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit='s')
concatenated_df.describe()
concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)
concatenated_df['month'] = concatenated_df.Timestamp.dt.month
concatenated_df['day'] = concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp', inplace=True)
concatenated_df["CPU usage prev"] = concatenated_df['CPU usage [%]'].shift(1)
concatenated_df["CPU_diff"] = concatenated_df['CPU usage [%]'] - concatenated_df["CPU usage prev"]
concatenated_df["received_prev"] = concatenated_df['Network received throughput [KB/s]'].shift(1)
concatenated_df["received_diff"] = concatenated_df['Network received throughput [KB/s]']- concatenated_df["received_prev"]
concatenated_df["transmitted_prev"] = concatenated_df['Network transmitted throughput [KB/s]'].shift(1)
concatenated_df["transmitted_diff"] = concatenated_df['Network transmitted throughput [KB/s]']- concatenated_df["transmitted_prev"]

concatenated_df["start"] = concatenated_df.index
concatenated_df['target'] = concatenated_df['CPU usage [MHZ]']

df2 = concatenated_df.groupby('VM').resample('1min')['target'].mean().to_frame()
df2.reset_index(level=0, inplace=True)
df2.head()
df2 = df2.fillna(method='ffill')
df2.to_csv('df2.csv')

df3 = concatenated_df.groupby('VM').resample('1min')['CPU capacity provisioned [MHZ]'].mean().to_frame()
df3.reset_index(level=0, inplace=True)
df3 = df3.fillna(method='ffill')
df3.head()
df3.to_csv('df3.csv')
