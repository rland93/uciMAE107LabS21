import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
plt.rcParams.update({
    'text.usetex' : True,
})
'''
# constants
rho_air = 1.2754 # kg/m
D_cyl = 0.5 * 2.54 * 0.01 # in -> m

# data import
data = pd.read_csv('./rawdata.csv')
data_nocyl = pd.read_csv('./rawdata_nocyl.csv')

# find nocyl data
data_nocyl['dh_cm'] = data_nocyl['dh'] * 2.54
data_nocyl['dist_cm'] = data_nocyl['dist'] * 2.54
data_nocyl['dp_pa'] = data_nocyl['dh_cm'] * 98.1
data_nocyl['dist_m'] = data_nocyl['dist_cm'] * 0.01
data_nocyl['v_ms'] = np.sqrt( 2 * data_nocyl['dp_pa'] / rho_air )

U1 = data_nocyl['v_ms'].max()
print(U1)
# unit conversions
data['dh_cm'] = data['dh'] * 2.54 # in->cm
data['dist_cm'] = data['dist'] * 2.54 # in->cm

data['dp_pa'] = data['dh_cm'] * 98.1 # cm h20 -> Pa
data['dist_m'] = data['dist_cm'] * 0.01 # cm -> m
data['normdist'] = data['dist_m'] / D_cyl
data['normdist_byw'] = data['dist_m'] / (4.5 * 2.54 * 0.01)
data['v_ms'] = np.sqrt( ( 2*data['dp_pa'] / rho_air ))
data['vnorm_ms'] = data['v_ms'] / U1

data['X'] = data['v_ms'] / U1
data['mdef'] = data['X'] * (1 - data['X'])

fig1, axs1 = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10,5))
fig1.suptitle('Normalized Stream Velocities')
# q 3a
for i, (loc, locd) in enumerate(data.groupby('loc')):
    axs1[i].set_title('Cylinder in location: {}'.format(loc))
    axs1[i].plot(locd['normdist'], locd['vnorm_ms'])
    axs1[i].set_xlabel('normalized distance (by cylinder width)')
    axs1[i].set_ylabel('normalized stream velocity (by max velocity)')

fig2, axs2 = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10,5))
fig2.suptitle('Normalized Momentum Deficits')
# q 3b
for i, (loc, locd) in enumerate(data.groupby('loc')):
    # display normalized distance chart
    # locd['normdist_byw'] = locd['normdist_byw'] - locd['normdist_byw'].min()
    axs2[i].set_title('Cylinder in location: {}'.format(loc))
    axs2[i].plot(locd['normdist_byw'], locd['mdef'])
    axs2[i].set_xlabel('normalized distance (by cross section width)')
    axs2[i].set_ylabel('normalized momentum deficit')

    # displaying text of max velocity
    max_U = locd['v_ms'].iloc[locd['mdef'].argmax()]
    max_y = locd['mdef'].max()
    max_x = locd['normdist_byw'].iloc[locd['mdef'].argmax()]
    # string to display
    max_str = r"$\mathcal{X}_{max} =$" + str( round(locd['mdef'].max(),2) ) + '\n' + r"$U_{max}=$" + str(round(max_U, 2)) + r"$m/s$"
    axs2[i].text(max_x, max_y, max_str)
    axs2[i].set_ylim(0, .35)

    # get C_D
    dy = locd['normdist_byw'].max() - locd['normdist_byw'].min()
    c_d = locd['mdef'].sum() * dy
    print('c_d={}'.format( round(c_d, 2) ) )

plt.show()