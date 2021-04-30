import pandas as pd
import numpy as np
import scipy.stats as stats

# length, visc., density
L, mu, rho = 0.36, 0.0010016, 1000

data = pd.read_csv('./rawdata.csv')
data['dp'] = data['p1 mm'] - data['p2 mm']
data['Q m3/s'] = data['vol mL'] / 1000000 / data['time s']
data['Q mL/s'] = data['vol mL'] / data['time s']
data['dp Pa'] = data['dp'] / .10197
sfunc = lambda t: True if t not in ['Venturi', 'Orifice'] else False
data = data[data['type'].apply(sfunc)]
data['A'] = data['d1'] * data['d1'] * np.pi / 4.0
data['uavg'] = data['Q m3/s'] / data['A']
data['Re'] = rho * data['uavg'] * data['d1'] / mu
data['f'] = data['d1'] / L * data['dp Pa'] / (0.5 * rho * data['uavg'] * data['uavg'])

data['head loss'] = data['dp Pa'] / (9.81 * rho)
data['pump power'] = data['dp Pa'] * data['Q m3/s']


t1d, t2d, t3d = {}, {}, {}
for D, df in data.groupby('d1'):
    # to calculate var
    errPa =stats.tstd(df['dp Pa'].values)
    errV =stats.tstd(df['vol mL'].values)
    errt =stats.tstd(df['time s'].values)
    # flow rate error
    df['errQ'] = 1/df['time s'] * errV - 1/(df['time s']* df['time s']) * errt # mL
    # velocity error
    df['errVe'] = df['errQ'] / 1000000 / df['A'].max() #mL -> m^3 then convert to m/s
    errRe = (rho * df['errVe'] * D / mu).mean()
    # f error each 
    df['errf'] = 2 * D / (L * rho * df['uavg']**2) * errPa - 6 * D * df['dp Pa'] / (L * rho * df['uavg']**3) * df['errVe']
    # f error overall
    errf = df['errf'].mean()
    # tabulate
    t1d[D] = {
        'dp Pa' : errPa,
        'vol mL' : errV,
        'time s' : errt, 
        'Re' :  errRe,
        'f' : errf
    }
    # pump power
    power = df['pump power'].mean()
    # cost function
    cfunc = lambda price, hperday: power * price * hperday * 3600 * 365 * 10 / 3600000
    volfunc = lambda hperday: 3600 * hperday * 365 * 10 * df['Q m3/s'].mean()
    t2d[D] = {
        'Re': df['Re'].mean(),
        'f' : df['f'].mean(),
    }
    t3d[D] = {
        'Q m3/s' : df['Q mL/s'].mean(),
        'head loss' : df['head loss'].mean(),
        'Pump Power' : power,
        'cost (peak)' : cfunc(0.25, 8),
        'cost (off-peak)' : cfunc(0.10, 16),
        'cost (peak, normalized)' : cfunc(0.25, 8) / volfunc(8) * 100,
        'cost (off-peak, normalized)' : cfunc(0.10, 16) / volfunc(16) * 100,
    }
table1 = pd.DataFrame(t1d)
print(table1)
print('\n\n')
table2 = pd.DataFrame(t2d)
print(table2)
print('\n\n')
table3 = pd.DataFrame(t3d)
print(table3)
print('\n\n')
table1.to_csv('table1.csv')
table2.to_csv('table2.csv')
table3.to_csv('table3.csv')