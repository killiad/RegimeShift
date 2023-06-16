#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits import mplot3d
import math
from sklearn.cluster import KMeans
from ripser import Rips
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import ruptures as rpt
import gudhi as gd
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI
from Period_Functions import *

field = 'Onalaska'
#cols = ['TN', 'TP', 'TEMP', 'CHLcal', 'DO', 'SS', 'VEL', 'WDP']
cols = ['TEMP', 'TP', 'DO', 'SS', 'CHLcal']
fields = {'Onalaska' : "~/TDA_Final/Data/Middle_50_Onalaska_1995_2019_48_",
          'Pool13' : "~/TDA_Final/Data/Middle_50_Pool13_1995_2019_54_"}
season_size = {'Onalaska' : 48, 'Pool13' : 54}
ending_years = {'Onalaska' : 2019, 'Pool13' : 2019}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#From Data_Prep.py
n = season_size[field]
ending_year = ending_years[field]
starting_year = 1995
years = [x for x in range(starting_year,ending_year)]
seasons = ["SPRING", "SUMMER", "FALL", "WINTER"]

#total_window = 2000
time_step = 1.0/(4*n)
w = 12*n
dim = 4
d = int(n / 3)
n_cells = 8

points = [x for x in range(4*n*len(years))]
middle_points = points[int(w/2):len(points)-int(w/2)]
cur_labels = middle_points[2*n::8*n]
new_labels = years[2:-1:2]

periods_df = pd.DataFrame()

for col in cols:
    field_path = fields[field] + col + '.csv'
    fix_seasons = pd.read_csv(r"{}".format(field_path))
    time_series = np.array(fix_seasons[col])
    total_window = len(time_series)
    periods = []
    collective_periods = np.zeros(len(middle_points), dtype='double')
    
    for i in range(total_window):
        if i % size == rank:
            print("Rank: " + str(rank) + " " + str(i) + " / " + str(total_window))
            window = get_middle_window(time_series, w, i)
            if window is None:
                print("Not a valid midpoint")
                continue
            swe = sliding_window_embedding(window, dim, d)
            diagram = plot_persistent_homology_diagram(swe, False)
            if diagram == 0:
                periods.append(0)
                print("Not enough data to do SWE, returning 0")
                continue
            l1, l2 = calculate_norms(diagram)
            try:
                landmarks, cells = generate_voronoi_cells(swe, n_cells)
            except ValueError:
                periods.append(0)
                print("Failed to make Voronoi cells, returning 0")
                continue
            jumps = generate_vector_jumps(landmarks,cells)
            summary = Jump_Summary(jumps, 20)
            periods.append(estimate_period(summary, time_step))

    #Gather periods
    comm.Barrier()
    if rank == 0:
        print("Period Analysis Complete!")

    loc = 0
    for i in range(len(middle_points)):
        r = i % size
        if r == 0:
            if rank == 0:
                collective_periods[i] = periods[loc]
                loc += 1
        else:
            if rank == 0:
                collective_periods[i] = comm.recv(source=r, tag=int(i / size))
            elif rank == r:
                comm.send(periods[loc], dest=0, tag=loc)
                loc += 1
    
    if rank == 0:
        periods_df[col] = collective_periods

    graph_name = 'Graphs/' + field + "_" + col + ".png"
    if rank == 0:
        print("All Periods Received!")
        temp = range(len(collective_periods))
        plt.clf()
        #index = [i / (4.0 * n) for i in range(len(collective_periods))]
        #plt.plot(index,collective_periods)
        plt.scatter(temp,collective_periods, s=25)
        plt.ylim(bottom=0)
        plt.xlabel('Years')
        plt.ylabel('Period (In Years)')
        plt.xticks(cur_labels, new_labels)
        plt.title(field + " Periods: " + col)
        plt.savefig(graph_name)
        print(graph_name, " successfully created.")

if rank == 0:
    periods_df.to_csv("Middle_Periods.csv")
    print("Complete.")