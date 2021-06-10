import os
import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
import pandas as pd
from types import SimpleNamespace
from scipy import optimize
from matplotlib import pyplot as plt

from matplotlib.ticker import MultipleLocator, LogLocator


### Unit Conversions
ft_to_m = lambda x: 0.3048 * x
m_to_ft = lambda x: x / 0.3048
degf_to_degc = lambda x: (x - 32) * 5/9
degc_to_degf = lambda x: x * 9/5 + 32
gpm_to_m3s = lambda x: x * 6.30902e-5
psi_to_Pa = lambda x: x * 6894.76
lbft3_to_kgm3 = lambda x: 16.0185 * x
perkwh_to_perjoule = lambda x: x / 3600000

# back to imperial, for output tables
watt_to_hp = lambda x: x * 0.00134102
ms_to_fts = lambda x: x * 3.28084
Pa_to_psi = lambda x: x / 6894.76
m_to_in = lambda x: x * 39.3701

# 1/ft --> 1/m
perft_to_perm = lambda x: x / 0.3048

### Constant Namespaces
# system constants
sysc = SimpleNamespace()
sysc.met = SimpleNamespace()
sysc.imp = SimpleNamespace()
# financial constants
finc = SimpleNamespace()

### System Constants
# length of pipe run
sysc.imp.L = 8000 # feet
sysc.met.L = ft_to_m(sysc.imp.L) # meters

# flow rate
sysc.imp.Qh = 2000 # gallons per minute
sysc.imp.Qc = 28000 # gallons per minute
sysc.met.Qh = gpm_to_m3s(sysc.imp.Qh) # m3/s
sysc.met.Qc = gpm_to_m3s(sysc.imp.Qc) # m3/s

# water ingress temps
sysc.imp.Tc = 40 # fahrenheit
sysc.imp.Th = 351 # fahrenheit
sysc.met.Tc = degf_to_degc(sysc.imp.Tc) # celcius
sysc.met.Th = degf_to_degc(sysc.imp.Th) # celcius

# water pressures
sysc.imp.Ph = 230 + 14.7 # psi (abs)
sysc.imp.Pc = 0 + 14.7 # psi (abs)
sysc.met.Ph = psi_to_Pa(sysc.imp.Ph) # Pa
sysc.met.Pc = psi_to_Pa(sysc.imp.Pc) # Pa

# water dyn. viscosity
sysc.met.dynv_h = 0.0001528 # Pa * s [Engineering Toolbox]
sysc.met.dynv_c = 0.0015484 # Pa * s [Engineering Toolbox]
# water kin. viscosity
sysc.met.kinv_h = 1.718e-7 # m^2 / s [Engineering Toolbox]
sysc.met.kinv_c = 0.0000015486 # m^2 / s [Engineering Toolbox]

# water dens.
sysc.imp.dens_h = 55.55  # lb/ft^3
sysc.imp.dens_c = 62.425 # lb/ft^3
sysc.met.dens_h = lbft3_to_kgm3(sysc.imp.dens_h)
sysc.met.dens_c = lbft3_to_kgm3(sysc.imp.dens_c)

# pipe roughness
sysc.eta = 4.5e-5 # unitless, [Engineering Toolbox]
sysc.energyprice = 0.199 # dollars per kwh
sysc.energyprice_J = perkwh_to_perjoule(sysc.energyprice) # dollars per joule

# read pipe cost info
pipecost = pd.read_csv('pipecost.csv', dtype=np.float64)
# index by diameter in inches
pipecost.index = pipecost['diameter'].values
# unit conversions
pipecost['D'] = ft_to_m(pipecost['diameter'] / 12)
pipecost['cost/m'] = perft_to_perm(pipecost['cost/ft'])

# hot pipe
pipeH = pd.DataFrame(pipecost[['cost/m','D']])
# cold pipe
pipeC = pd.DataFrame(pipecost[['cost/m','D']])

# flow velocity
def flowvel(Q, D):
    return Q * 4 / (np.pi * D * D)
# Reynolds no
def Re(V, D, dynv):
    return V * D / dynv

### Flow Velocities, Re numbers
# flow velocities
pipeH['V'] = flowvel(sysc.met.Qh, pipeH['D'])
pipeC['V'] = flowvel(sysc.met.Qc, pipeC['D'])
# reynolds numbers
pipeH['Re'] = Re(pipeH['V'], pipeH['D'], sysc.met.kinv_h)
pipeC['Re'] = Re(pipeC['V'], pipeC['D'], sysc.met.kinv_c)

# Friction Factor
def fricFac(Re, eta, D):
    if Re > 4100:
        return f_colebrook_white(Re, eta, D)
    else:
        return f_blasius(Re)

def f_colebrook_white(Re, eta, D):
    # cb-white equation
    D = m_to_ft(D)/12
    def func(f): 
        return (-2*np.log10((2.51/(Re*np.sqrt(f))) + (eta/(3.71*D))) - 1.0/np.sqrt(f))
    # solve for f using root
    return np.squeeze(optimize.root(func, 0.02).x)

def f_blasius(Re):
    return 64/Re

# Friction Factor:
# we have to solve individually because root cannot be vectorized
fricFac_wrap = lambda x: fricFac(x['Re'], sysc.eta, x['D'])

pipeH['f'] = pipeH.apply(fricFac_wrap, axis=1)
pipeC['f'] = pipeC.apply(fricFac_wrap, axis=1)

# Pressure Drop
def pdrop(L, D, rho, V, f):
    return L/D * (rho * V**2)*f

pipeH['dP'] = pdrop(sysc.met.L, pipeH['D'], sysc.met.dens_h, pipeH['V'], pipeH['f'])
pipeC['dP'] = pdrop(sysc.met.L, pipeC['D'], sysc.met.dens_c, pipeC['V'], pipeC['f'])

# Pumping Power
def ppower(dP, Q):
    return dP * Q

pipeH['Pw'] = ppower(pipeH['dP'], sysc.met.Qh)
pipeC['Pw'] = ppower(pipeC['dP'], sysc.met.Qc)

### Pumping Energy
def energy_1yr(pwr):
    return pwr * 60 * 60 * 24 * 365.25

def cost_1yr(watts):
    return watts / 1000 * 24 * 365.25 * sysc.energyprice

pipeH['E_1yr'] = energy_1yr(pipeH['Pw'])
pipeC['E_1yr'] = energy_1yr(pipeC['Pw'])
# Pumping Cost (1 year)
pipeH['1ycost'] = cost_1yr(pipeH['Pw'])
pipeC['1ycost'] = cost_1yr(pipeC['Pw'])




###### Capital Costs
# define constants
costc = SimpleNamespace()
costc.capMultiplier = 10
costc.apr = 0.065
costc.yearTerm = 25

# annual average interest
def calc_loan_term(C, apr, n):
    # calculate monthly payment
    m = 0
    for i in range(n):
        m += (1+apr) ** i
    monthly = (apr + 1/m) * C
    interest = C * apr
    principal = monthly - interest
    remaining = C - principal

    amort = {
        'payment no.' : [1],
        'interest payment' : [round(interest, 2)],
        'principal payment' :[round(principal,2)],
        'total payment':[round(principal + interest, 2)],
        'remaining':[round(remaining, 2)],
    }
    for i in list(range(n))[1:]:
        interest = remaining * apr
        principal = monthly - interest
        remaining -= principal
        amort['payment no.'].append(i+1)
        amort['interest payment'].append(round(interest,2))
        amort['principal payment'].append(round(principal,2))
        amort['total payment'].append(round(principal + interest, 2))
        amort['remaining'].append(round(remaining, 2))
    return pd.DataFrame(amort, index=amort['payment no.'])

def capcost_mult(yearterm, apr):
    n = 0
    for i in range(yearterm):
        n += (1 + apr)**i
    return apr + 1/n



pipeH['capital'] = pipeH['cost/m'] * costc.capMultiplier * sysc.met.L
pipeC['capital'] = pipeC['cost/m'] * costc.capMultiplier * sysc.met.L

pipeH['M'] = pipeH['capital'] * capcost_mult(costc.yearTerm, costc.apr)
pipeC['M'] = pipeC['capital'] * capcost_mult(costc.yearTerm, costc.apr)

pipeH['total_cost'] = (pipeH['M'] + pipeH['1ycost'])
pipeC['total_cost'] = (pipeC['M'] + pipeC['1ycost'])


### Output Table 1
t1_pipeH = pd.DataFrame(index=pipeH.index)
t1_pipeC = pd.DataFrame(index=pipeC.index)
# flow vel
t1_pipeH['avg. flow vel'] = ms_to_fts(pipeH['V'])
t1_pipeC['avg. flow vel'] = ms_to_fts(pipeC['V'])
# Re
t1_pipeH['Re'] = pipeH['Re']
t1_pipeC['Re'] = pipeC['Re']
# rough/dia
t1_pipeH['eta/D'] = sysc.eta / m_to_in(pipeH['D'])
t1_pipeC['eta/D'] = sysc.eta / m_to_in(pipeC['D'])
# f
t1_pipeH['f'] = pipeH['f']
t1_pipeC['f'] = pipeC['f']
# dP
t1_pipeH['dP'] = Pa_to_psi(pipeH['dP'])
t1_pipeC['dP'] = Pa_to_psi(pipeC['dP'])
# pump power
t1_pipeH['Pw'] = watt_to_hp(pipeH['Pw'])
t1_pipeC['Pw'] = watt_to_hp(pipeC['Pw'])
# annual energy cost
t1_pipeH['1ycost'] = pipeH['1ycost'] / 1000
t1_pipeC['1ycost'] = pipeC['1ycost'] / 1000


##### Output Table
t2_pipeH = pd.DataFrame(index=pipeH.index)
t2_pipeC = pd.DataFrame(index=pipeC.index)
t2_pipeH['capital'] = pipeH['capital']
t2_pipeC['capital'] = pipeC['capital']
t2_pipeH['capcost'] = round(pipeH['M'], 2)
t2_pipeC['capcost'] = round(pipeC['M'], 2)
t2_pipeH['total_cost'] = pipeH['total_cost']
t2_pipeC['total_cost'] = pipeC['total_cost']



# Average interest

for p, t2 in zip((pipeH, pipeC), (t2_pipeH, t2_pipeC)):
    inter = []
    princ = []
    mortg = []
    energ = []
    total = []
    for dia, cap, energc in p[['capital','1ycost']].itertuples():
        df = calc_loan_term(cap, costc.apr, 25)
        dia = dia
        inter.append(round(df['interest payment'].mean(),2))
        princ.append(round(df['principal payment'].mean(),2))
        mortg.append(round(df['total payment'].mean(),2))
        energ.append(round(energc, 2))
        total.append(round(energc + round(df['total payment'].mean(),2), 2))

    t2['average interest'] = inter
    t2['average principal'] = princ
    t2['mortgage payment'] = mortg
    t2['energy cost'] = energ
    t2['total cost'] = total

t2_pipeH = t2_pipeH.apply(lambda x: x/1000)
t2_pipeC = t2_pipeC.apply(lambda x: x/1000)

print(t2_pipeH)
print(t2_pipeC)
        
        


### Convert Types
def cvtfloat(df):
    return df.astype(np.float64)
t1_pipeC, t1_pipeH, t2_pipeC, t2_pipeH, pipeC, pipeH = map(cvtfloat, (t1_pipeC, t1_pipeH, t2_pipeC, t2_pipeH, pipeC, pipeH))
sc_df = lambda dseries: (dseries.index, dseries.values)
fig1, ax1 = plt.subplots(ncols=2, tight_layout=True, figsize=(7.5,4.5))


for a in np.ndindex(ax1.shape):
    ax1[a].grid(which='minor', color='#CCCCCC', zorder=0)

#### PLOTS
def set_log_chart_params(ax):
    for a in np.ndindex(ax.shape):
        ax[a].set_yscale('log')
        ax[a].set_xlabel('Pipe Diameter (in)')
        ax[a].set_ylabel('Cost ($)')
        ax[a].legend()
        # x ticks
        ax[a].xaxis.set_major_locator(MultipleLocator(4))
        ax[a].xaxis.set_minor_locator(MultipleLocator(2))
        ax[a].yaxis.set_major_locator(LogLocator(base=10))
        subs = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
        locmin = LogLocator(base=10.0,subs=subs,numticks=12)
        ax[a].yaxis.set_minor_locator(locmin)
        ax[a].grid(which='minor', color='#f0f0f0', zorder=0)
        ax[a].grid(which='major', color='#ccc', zorder=0)

### Plot parameters
pltc = SimpleNamespace()
pltc.hotl = 4.0
pltc.hotu = 28.0
pltc.coll = 16.0
pltc.colu = 36.0
pltc.encol = 'tab:blue'
pltc.capcol = 'tab:red'

### Bar Plots of costs

ax1[0].set_ylim( (1e4, 1e8) )
ax1[1].set_ylim( (1e4, 1e8) )

ax1[0].bar(
    pipeH['M'][pltc.hotl:pltc.hotu].index, 
    pipeH['M'][pltc.hotl:pltc.hotu] + pipeH['1ycost'][pltc.hotl:pltc.hotu], 
    color=pltc.capcol, 
    label='Capital Cost',
    bottom=pipeH['1ycost'][pltc.hotl:pltc.hotu].values,
    zorder=2,
)
ax1[0].bar(
    pipeH['1ycost'][pltc.hotl:pltc.hotu].index, 
    pipeH['1ycost'][pltc.hotl:pltc.hotu].values, 
    label='Energy Cost',
    color=pltc.encol,
    zorder=2,
)
ax1[0].set_title('HTW Yearly Costs (By Type)')
ax1[1].bar(
    pipeC['M'][pltc.coll:pltc.colu].index, 
    pipeC['M'][pltc.coll:pltc.colu] + pipeC['1ycost'][pltc.coll:pltc.colu], 
    color=pltc.capcol, 
    label='Capital Cost',
    bottom=pipeC['1ycost'][pltc.coll:pltc.colu].values,
    zorder=2,
)
ax1[1].bar(
    pipeC['1ycost'][pltc.coll:pltc.colu].index, 
    pipeC['1ycost'][pltc.coll:pltc.colu].values, 
    label='Energy Cost',
    color=pltc.encol, 
    zorder=2,
)
ax1[1].set_title('CTW Yearly Costs (By Type)')
set_log_chart_params(ax1)


### Scatter Plots of costs
fig2, ax2 = plt.subplots(ncols=2, tight_layout=True, figsize=(7.5,4.5))
ax2[0].set_ylim( (1e4, 1e8) )
ax2[1].set_ylim( (1e4, 1e8) )
ax2[0].scatter(*sc_df(pipeH['total_cost']), marker='x', color='r', label='total cost',zorder=2,)
ax2[0].scatter(*sc_df(pipeH['1ycost']), marker='.', color='b', label='energy cost',zorder=2,)
ax2[0].scatter(*sc_df(pipeH['M']), marker='.', color='g', label='capital cost',zorder=2,)
ax2[0].axhline(y=pipeH['total_cost'][14], color='k', linestyle=':')
ax2[0].axvline(x=14, color='k', linestyle=':')
ax2[0].set_title('HTW Yearly Costs')

ax2[1].scatter(*sc_df(pipeC['total_cost']), marker='x', color='r', label='total cost',zorder=2,)
ax2[1].scatter(*sc_df(pipeC['1ycost']), marker='.', color='b',  label='energy cost',zorder=2,)
ax2[1].scatter(*sc_df(pipeC['M']), marker='.', color='g',  label='capital cost',zorder=2,)
ax2[1].axhline(y=pipeC['total_cost'][32], color='k', linestyle=':')
ax2[1].axvline(x=32, color='k', linestyle=':')
ax2[1].set_title('CTW Yearly Costs')

set_log_chart_params(ax2)

Hcost = pipeH['capital'][pipeH['total_cost'].idxmin()]
Ccost = pipeC['capital'][pipeC['total_cost'].idxmin()]
amortH = calc_loan_term(Hcost / 1000, costc.apr, 25)
amortC = calc_loan_term(Ccost / 1000, costc.apr, 25)

TABLEDIR = './tables'
os.makedirs(TABLEDIR)

amortH.to_csv(TABLEDIR + '/amortH.csv')
amortC.to_csv(TABLEDIR + '/amortC.csv')

t1_pipeH.to_csv(TABLEDIR + '/t1h.csv')
t1_pipeC.to_csv(TABLEDIR + '/t1c.csv')

t2_pipeH.to_csv(TABLEDIR + '/t2h.csv')
t2_pipeC.to_csv(TABLEDIR + '/t2c.csv')

plt.show()
