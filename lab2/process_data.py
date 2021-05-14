import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# exponential fn to fit to curves
def expfun(t, c1, a, T_inf):
    return - c1 * np.exp(-a * t) + T_inf

rawdata = pd.read_csv('lab2_home_raw_data.csv')

fig1, ax1 = plt.subplots(figsize=(7,3), tight_layout=True)
fig2, ax2 = plt.subplots(figsize=(7,3), tight_layout=True)

# we iterate through each type, of which
# there are two: (uninsulated, insulated)
curve_parameters = dict()
for runtype, d1 in rawdata.groupby('type'):
    xs = d1['time'][d1['time'] <= 50]
    ys = d1['tempK'][d1['time'] <= 50] - 273.15
    '''
    fit to our model of the system. For a body for which LC assumption is true, the governing
    equation is given by:

    dTs/dt = - (h As) / (m Cv) * ( T_s(t) - T_inf )
    
    We can group parameters:
    dTs/dt = - a * (Ts(t) - Tinf)

    This 1st order ode has solution of the form:
    where c1, a, and T_inf are free parameters

    T_s(t) = - c1 e^( -a t ) + T_inf

    So we fit the exp curve to those parameters
    '''
    params, _ = scipy.optimize.curve_fit(expfun, xs, ys, p0=(1, 1e-2, 1))
    xxs = np.linspace(np.min(xs), np.max(xs), 900)
    yys = expfun(xxs, *params)
    curve_parameters[runtype] = {
        'c1' : params[0],
        'hAs/mCv' : params[1],
        'T_inf' : params[2]
    }

    # we iterate through each run
    if runtype == 'uninsulated':
        for run, d2 in d1.groupby('video_fname'):
            ax1.scatter(d2['time'][d2['time'] <= 50], d2['tempK'][d2['time'] <= 50]- 273.15, marker='+', label=runtype)
        ax1.plot(xxs, yys, 'k', label='exp fitted - {}'.format(runtype))
        ax1.set_ylabel('Temperature (C)')
        ax1.set_xlabel('Time (s)')
        ax1.legend()

    elif runtype == 'insulated':
        for run, d2 in d1.groupby('video_fname'):
            ax2.scatter(d2['time'][d2['time'] <= 50], d2['tempK'][d2['time'] <= 50]- 273.15, marker='+', label=runtype)
        ax2.plot(xxs, yys, 'k', label='exp fitted - {}'.format(runtype))
        ax2.set_ylabel('Temperature (C)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()

pd.DataFrame(curve_parameters).to_csv('table1.csv')
plt.show()



