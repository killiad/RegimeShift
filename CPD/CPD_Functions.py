import numpy as np
import math
import pandas as pd
import gudhi as gd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import ruptures as rpt

def Normalize_Time_Series(time_series):
    time_series = [[i] for i in time_series]
    scaler = MinMaxScaler()
    scaler = scaler.fit(time_series)
    normalized = scaler.transform(time_series)
    return (normalized, scaler)


def Generate_Sliding_Window(time_series, size):
    return [time_series[i:i+size] for i in range(len(time_series) - size + 1)]


def sliding_window_embedding(time_series, n, d):
    size = len(time_series)
    swe = [0 for x in range(size - n*d)]
    for i in range(len(swe)):
        swe[i] = [time_series[i + k*d] for k in range(n+1)]
    return np.array(swe)


def Generate_Betti_Numbers(window, p=0, step=.01, n=1):
    min_distance = Get_Minimum_Distance(window)
    steps = math.ceil(min_distance / step)
    eps = step * steps
    betti = [len(window) for x in range(steps)]
    #set eps = min distance between any two points
    #round eps up to next multiple of step
    #first eps / step entries for betti vector is number points in the window / time series
    #continue as normal
    fixed_window = [[i] for i in window]
    while eps < n:
        rips_complex = gd.RipsComplex(fixed_window,max_edge_length=eps)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=p)
        simplex_tree.compute_persistence()
        #print("Number of simplices in the V-R complex: ",simplex_tree.num_simplices())
        betti_num = simplex_tree.betti_numbers()[0]
        betti.append(betti_num)
        eps += step
    return betti


def Get_Minimum_Distance(window):
    min_distance = abs(window[0]-window[1])
    for i in range(len(window)):
        for j in range(i+1,len(window)):
            distance = abs(window[i]-window[j])
            if distance < min_distance:
                min_distance = distance
    return min_distance


def Reduce_Dimensions(data, dim=1):
    pca = PCA(n_components=dim)
    return (pca.fit_transform(data), pca)

def Raw_Series_To_Years(starting_year, n, bkpts):
    times = []
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    for point in bkpts:
        year = point / (4*n)
        remainder = year - int(year)
        season = seasons[int(4*remainder)]
        times.append((starting_year + int(year), season))
    return times