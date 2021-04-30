import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from matplotlib import pyplot as plt

'''functions'''
def rmse(time, volume, dV, dt):
    return np.sqrt(1/(time**2) * (dV ** 2) + (volume ** 2) / (time ** 4) * (dt ** 2))
    
'''Generate Histograms'''
# read lab data
data = pd.read_csv('data_cleaned.csv')
# create figures, subplots
fig1, axs1 = plt.subplots(2,2, figsize=(7,4), tight_layout=True)
fig2, axs2 = plt.subplots(2,2, figsize=(7,4), tight_layout=True)
fig1.suptitle('Data With Outliers (Lab)')
fig2.suptitle('Data Without Outliers (Lab)')
# calculate flow rate
data['flow rate (mL/s)'] = data['volume (mL)'] / data['time (s)']

# table 1: means, variances, rsme etc.
table1_column={}

# for each cylinder size
for i, (c, cyl) in enumerate(data.groupby('cyl size')):
    # for each flow 
    for j, (f, flow) in enumerate(cyl.groupby('flow')):
        # name distributions
        exp = 'In Lab'
        which_Q = '{}, vol={}, rate={}'.format(exp, c, f)
        
        # make two separate distributions, one with outliers and one without
        with_outliers = flow
        without_outliers = flow[stats.zscore(flow['flow rate (mL/s)']) <= 3]

        '''Question 2: '''
        # set axis titles
        axs1[i,j].set_title(which_Q)
        axs2[i,j].set_title(which_Q)
        # plot histogram with outliers
        sns.histplot(with_outliers['flow rate (mL/s)'], ax=axs1[i,j], bins=11)
        # plot hist without outliers
        sns.histplot(without_outliers['flow rate (mL/s)'], ax=axs2[i,j], bins=11)

        if c == 'large':
            # large cyl case
            dV = 5
            dt = .5
        elif c == 'small':
            # small cyl case
            # use least optimistic measure of dV dt
            dV = 1
            dt = .5

        '''Question 3, 4'''
        # we remove outliers from time, vol measurements individually
        times = flow[stats.zscore(flow['time (s)']) <= 3]['time (s)']
        vols = flow[stats.zscore(flow['volume (mL)']) <= 3]['volume (mL)']

        table1_row = {}
        # write data
        table1_row['Time mean'] = times.mean()
        table1_row['Time variance'] = stats.tstd(times)
        table1_row['Volume mean'] = vols.mean()
        table1_row['Volume variance'] = stats.tstd(vols)
        table1_row['Flow Rate mean'] = without_outliers['flow rate (mL/s)'].mean()
        table1_row['Flow Rate variance'] = stats.tstd(without_outliers['flow rate (mL/s)'])
        table1_row['Flow Rate predicted RSME'] = rmse(times.mean(), vols.mean(), dV, dt)
        # dict-of-dicts
        table1_column[which_Q] = table1_row

fig0, ax0 = plt.subplots()
sns.relplot(data=data[data['cyl size'] == 'large'], x='time (s)', y='volume (mL)', hue='flow')
sns.relplot(data=data[data['cyl size'] == 'small'], x='time (s)', y='volume (mL)', hue='flow')

# create figures, subplots
fig3, axs3 = plt.subplots(1,2, figsize=(7,4), tight_layout=True)
fig4, axs4 = plt.subplots(1,2, figsize=(7,4), tight_layout=True)
fig3.suptitle('Data With Outliers (Home)')
fig4.suptitle('Data Without Outliers (Home)')

# read home data
data_home = pd.read_csv('data_home_cleaned.csv')
# calculate flow rate
data_home['flow rate (mL/s)'] = data_home['volume (mL)'] / data_home['time (s)']

# for each flow rate
for i, (f, flow) in enumerate(data_home.groupby('flow')):
    # name distributions
    exp = 'At Home'
    which_Q = '{}, rate={}'.format(exp, f)

    # set titles
    axs3[i].set_title(which_Q)
    axs4[i].set_title(which_Q)

    # histplot with outliers
    sns.histplot(data=flow['flow rate (mL/s)'], ax=axs3[i], bins=11)
    # remove flow outliers
    without_outliers = flow[stats.zscore(flow['flow rate (mL/s)']) <= 3]
    # re-plot
    sns.histplot(data=without_outliers['flow rate (mL/s)'], ax=axs4[i], bins=11)

    # we remove outliers from time, vol measurements individually
    times = flow[stats.zscore(flow['time (s)']) <= 3]['time (s)']
    vols = flow[stats.zscore(flow['volume (mL)']) <= 3]['volume (mL)']

    dV = 1
    dt = 0.2


    table1_row = {}
    # write data
    table1_row['Time mean'] = times.mean()
    table1_row['Time variance'] = stats.tstd(times)
    table1_row['Volume mean'] = vols.mean()
    table1_row['Volume variance'] = stats.tstd(vols)
    table1_row['Flow Rate mean'] = without_outliers['flow rate (mL/s)'].mean()
    table1_row['Flow Rate variance'] = stats.tstd(without_outliers['flow rate (mL/s)'])
    table1_row['Flow Rate predicted RSME'] = rmse(times.mean(), vols.mean(), dV, dt)
    # dict-of-dicts
    table1_column[which_Q] = table1_row

pd.DataFrame(table1_column).transpose().to_csv('table1.csv')

# show plot    
plt.show()
