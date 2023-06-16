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

print("Imports complete")
print("Beginning Data Preperation")

df = pd.read_csv(r"~/TDA_Final/Data/Fix_Seasons_Onalaska_1995_2019_97.csv")
cols = ['TEMP', 'TN', 'TP', 'CHLcal', 'DO', 'SS', 'VEL', 'WDP']
years = [x for x in range(1995,2019)]
seasons = ["SPRING", "SUMMER", "FALL", "WINTER"]

#You may need to run this multiple times to determine n
n = 48

lower = .25
upper = .75

for col in cols:
    df_col = df[[col, 'YEAR','SEASON', 'MONTH', 'TIME']]
    columns = {x : [] for x in df_col.columns}
    middle_50 = pd.DataFrame(columns)
    for year in years:
        for season in seasons:
            counter = 0
            season_df = df_col[df_col['YEAR']==year]
            season_df = season_df[season_df['SEASON']==season]
            q1 = season_df[col].quantile(lower)
            q3 = season_df[col].quantile(upper)
            for index, row in season_df.iterrows():
                if q1 <= row[col] and row[col] <= q3:
                    counter = counter + 1
                    if counter > n:
                        break
                    middle_50.loc[len(middle_50.index)] = row

    middle_50.to_csv("~/TDA_Final/Data/Middle_50_Onalaska_1995_2019_" + str(n) + "_" + col + ".csv", index=False) 

    run_test = True
    if run_test:
        print("Col: " + col)
        for year in years:
            for season in seasons:
                test = middle_50[middle_50['YEAR']==year]
                test = test[test['SEASON']==season]
                print(year, " ", season, " ", len(test))

print("Data Prep Complete complete")