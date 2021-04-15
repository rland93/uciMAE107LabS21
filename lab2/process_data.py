import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# exponential fn to fit to curves
def expfun(t, c1, a, T_inf):
    return - c1 * np.exp(-a * t) + T_inf

rawdata = pd.read_csv('lab2_home_raw_data.csv')
fig, ax = plt.subplots(2, sharex=True)
# we iterate through each type, of which
# there are two: (uninsulated, insulated)
curve_parameters = dict()
for runtype, d1 in rawdata.groupby('type'):
    xs = d1['time']
    ys = d1['tempK']
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
            ax[0].scatter(d2['time'][d2['time'] <= 50], d2['tempK'][d2['time'] <= 50], marker='.', label=runtype)
        ax[0].plot(xxs, yys, 'k', label='exp fitted - {}'.format(runtype))
        ax[0].set_ylabel('Temperature (K)')
        ax[0].set_xlabel('Time (s)')
        ax[0].legend()

    elif runtype == 'insulated':
        for run, d2 in d1.groupby('video_fname'):
            ax[1].scatter(d2['time'][d2['time'] <= 50], d2['tempK'][d2['time'] <= 50], marker='.', label=runtype)
        ax[1].plot(xxs, yys, 'k', label='exp fitted - {}'.format(runtype))
        ax[1].set_ylabel('Temperature (K)')
        ax[1].set_xlabel('Time (s)')
        ax[1].legend()
for ins, params in curve_parameters.items():
    print(ins)
    for k, param in params.items():
        print('\t{}: {}'.format(k, param))
plt.show()



