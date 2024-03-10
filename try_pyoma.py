import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
from pyoma2.OMA import SingleSetup

data_path = "./dataset/bists/ambient_response.pkl"
with open(data_path, "rb") as f:
    data = pd.read_pickle(f)
data = data["acc"].T

Pali_ss = SingleSetup(data, fs=20)
# fig1, ax1 = Pali_ss.plot_data()
# plt.show()

# Initialise the algorithms
fsdd = FSDD_algo(name="FSDD", nxseg=2000, method_SD="per", pov=0.5)
# ssicov = SSIcov_algo(name="SSIcov", br=50, ordmax=80)

# Add algorithms to the single setup class
# Pali_ss.add_algorithms(ssicov, fsdd)
Pali_ss.add_algorithms(fsdd)

# Run all or run by name
# Pali_ss.run_by_name("SSIcov")
Pali_ss.run_by_name("FSDD")
Pali_ss.MPE("FSDD", sel_freq=[0.56, 1.49, 2.41, 3.28, 4.02, 4.78, 5.23], MAClim=0.95)

# Pali_ss.run_all()

# save dict of results
# fsdd_res = dict(fsdd.result)

print(fsdd.result.Fn)
print(fsdd.result.Phi)
print(fsdd.result.Xi)
fig3, ax3 = fsdd.plot_CMIF(freqlim=(0.2, 8))
plt.tick_params(axis="both", direction="in")
plt.ylim([-80, 10])
plt.show()

# figs, axs = Pali_ss[fsdd.name].plot_FIT(freqlim=(0.2, 8))
# plt.show()
figs, axs = fsdd.plot_FIT(freqlim=(0.2, 8))
plt.show()
