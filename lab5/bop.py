import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./BOP_raw.csv')
# calculate mass of the stack from pct o2 and mass of gas
def mstack(eta, mgas):
    mult = (119/2)* ( (2+eta)/(7-33*eta) ) + 1
    return mgas * mult
# calculate CO2 from input
def co2(mgas):
    return mgas * (44/16)

def kwh_co2kg(co2ph):
    return  co2ph / (data['MW'] * 1000)

data['mstack'] = mstack(data['pctO2'], data['mgas'])
data['co2(lbm/h)'] = co2(data['mgas'])
data['co2(kg/h)'] = data['co2(lbm/h)'] * 0.45359237
data['kWh/lbmCO2'] = kwh_co2kg(data['co2(lbm/h)'])

print(data['kWh/lbmCO2'].mean())

fig, ax1 = plt.subplots(tight_layout=True, figsize=(9,6))
ax1.plot(data['time'], data['co2(lbm/h)'])
ax1.set_xlabel('Time')
ax1.set_ylabel('CO2 Emission (lbm/h)')
ax1.set_title('CO2 Emission Over Time in UCI Central Plant')
plt.xticks(rotation=90)
plt.show()