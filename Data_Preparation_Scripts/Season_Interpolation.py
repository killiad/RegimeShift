import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits import mplot3d
import math
from sklearn.cluster import KMeans
#from ripser import Rips
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import ruptures as rpt
import gudhi as gd
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


print("Imports complete")

predicted_df = pd.read_csv(r"~/TDA_Final/Data/water_full.csv")
time_df = pd.read_csv(r"~/TDA_Final/Data/water_data_qfneg.csv")

field = 'Onalaska, WI'
predicted_df = predicted_df[predicted_df['FLDNUM']==field]

predicted_df.insert(0,'TIME',time_df['TIME'],True)
predicted_df["MONTH"] = pd.DatetimeIndex(predicted_df["DATE"]).month
predicted_df["DATE"] = pd.to_datetime(predicted_df["DATE"])
predicted_df["SEASON"] = predicted_df["MONTH"]
seasons = {3 : 'SPRING',
           4 : 'SPRING',
           5 : 'SPRING',
           6 : 'SUMMER',
           7 : 'SUMMER',
           8 : 'SUMMER',
           9 : 'FALL',
           10 : 'FALL',
           11: 'FALL',
           12: 'WINTER',
           1: 'WINTER',
           2: 'WINTER'}
predicted_df = predicted_df.replace({"SEASON" : seasons})

stratum = {'Main channel' : 1, 'Side channel' : 2, 'Backwater area contiguous to the main channel' : 3}
predicted_df = predicted_df.replace({"STRATUM" : stratum})
predicted_df.sort_values(by=["DATE","TIME"], inplace=True)

#Add date
date = []
for index, row in predicted_df.iterrows():
    time = row["TIME"]
    time = time.split(":")
    h = int(time[0])
    m = int(time[1])
    date.append(row["DATE"] + pd.DateOffset(hours=h, minutes=m))
date = np.array(date)
predicted_df["DATE"] = date

#Choose years and test number of points, if desired
run_test_1 = False
run_test_2 = True
years = [x for x in range(1995,2019)]
seasons = ["SPRING", "SUMMER", "FALL", "WINTER"]
n = 97
if run_test_1:
    for year in years:
        for season in seasons:
            test = predicted_df[predicted_df['YEAR']==year]
            test = test[test['SEASON']==season]
            print(year, " ", season, " ", len(test))

if run_test_2:
    for year in years:
        for season in seasons:
            test = predicted_df[predicted_df['YEAR']==year]
            test = test[test['SEASON']==season]
            if(len(test) < n):
                print("Year ", year, " season ", season, " has ", len(test), " points!")

#construct data frame, with same number of points in each season, that we will use
columns = {x : [] for x in predicted_df.columns}
fix_seasons = pd.DataFrame(columns)
for year in years:
    print(year)
    exception = False
    
    year_df = predicted_df[predicted_df["YEAR"]==year]
    for season in seasons:
        
        #only have this run for Pool 13
        if field == 'Bellevue, IA':
            if year == 1999 and season == "WINTER":
                exception = True
            
        seasonal_df = year_df[year_df["SEASON"]==season]
        if seasonal_df.empty or exception:
            print("Interpolating for " + season + " " + str(year))
            prev_year = year - 1
            prev_year_df = fix_seasons[fix_seasons["YEAR"]==prev_year]
            seasonal_df = prev_year_df[prev_year_df["SEASON"]==season]
            seasonal_df['YEAR'] = seasonal_df['YEAR'].map({prev_year:year, np.nan:year}, na_action=None)
            seasonal_df.loc[seasonal_df.YEAR==prev_year,'DATE']=np.nan
            seasonal_df.loc[seasonal_df.YEAR==prev_year,'TIME']=np.nan
        counter = 0
        for index, row in seasonal_df.iterrows():
            if counter >= n:
                break
            fix_seasons.loc[len(fix_seasons.index)] = row
            counter += 1

fix_seasons.to_csv("~/TDA_Final/Data/Fix_Seasons_Onalaska_1995_2019_" + str(n) + ".csv", index=False)            

print("Data Prep Complete complete")