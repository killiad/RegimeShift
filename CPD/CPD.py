#!/usr/bin/env python
import numpy as np
import math
import pandas as pd
import gudhi as gd
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import ruptures as rpt

from mpi4py import MPI
from CPD_Functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    print("imports complete")

#Field characteristics
field = 'Pool13'
cols = ['TN', 'TP', 'TEMP', 'CHLcal', 'DO', 'SS', 'VEL', 'WDP']
fields = {'Onalaska' : "~/TDA/Data/Interpolated/Middle_50_Onalaska_1995_2020_49_",
          'Pool13' : "~/TDA/Data/Interpolated/Middle_50_Pool13_1995_2019_54_"}
season_size = {'Onalaska' : 49, 'Pool13' : 54}
ending_years = {'Onalaska' : 2020, 'Pool13' : 2019}

multiplier = 1
alg = 'Binseg'
n = season_size[field]
starting_year = 1995
ending_year = ending_years[field]
ticks = np.arange(0,4*n*(ending_year-starting_year+1),8*n)
labels = np.arange(starting_year,ending_year+1,2)

#Makes sure 8 nodes are used
col = comm.scatter(cols, root=0)
field_path = fields[field] + col + '.csv'
df = pd.read_csv(r"{}".format(field_path))
#graph_name = 'Graphs/CPD/' + str(multiplier) + '/' + alg +'/' + alg + '_' + str(multiplier) + 'n_' + field + '_' + col
graph_name = 'Graphs/Final/Test/CPD_' + field + '_' + col
title = alg + ' ' + field + ': ' + col

#Begin topological manipulation
data = np.array(df[col])
data, scalar = Normalize_Time_Series(data)
betti_nums = []
X = Generate_Sliding_Window(data, multiplier*n)
for point in X:
    betti_nums.append(Generate_Betti_Numbers(point))
adjusted_betti_nums, pca = Reduce_Dimensions(betti_nums)
adjusted_betti_nums = adjusted_betti_nums.flatten()

#Change Point Detection
model = "l2"  
algo = rpt.Binseg(model=model).fit(adjusted_betti_nums)
#algo = rpt.Pelt(model=model, min_size=4*n,jump=4*n).fit(adjusted_betti_nums)
bkps = algo.predict(n_bkps=1)
# show results
rpt.show.display(adjusted_betti_nums, bkps, figsize=(10, 6))
plt.title(title)
plt.xticks(ticks,labels)
plt.savefig(graph_name, bbox_inches="tight")

bkps_time = Raw_Series_To_Years(starting_year, n, bkps)