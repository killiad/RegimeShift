import pandas as pd
import scipy.stats as stats
import numpy as np

def split(df, year):
    before = df[df['YEAR']<year]
    after = df[df['YEAR']>year]
    return (before, after)

fields = {'Onalaska' : "~/TDA_Final/Data/Middle_50_Onalaska_1995_2019_48_",
          'Pool13' : "~/TDA_Final/Data/Middle_50_Pool13_1995_2019_54_"}
sample_size = 60
bootstrap = 1000

cols = ['SS', 'CHLcal', 'TP', 'DO']
field_names = ['Onalaska', 'Pool13']
seasons = ['SPRING', 'SUMMER', 'FALL', 'WINTER']
starting_year = 1995
breakpoints = {'Onalaska' : {'SS' : 2014, 'CHLcal' : 2008, 'TP' : 2013, 'DO' : 2015},
               'Pool13' : {'SS' : 2013, 'CHLcal' : 2007, 'TP' : 2017, 'DO' : 2001}}
before_avg = []
after_avg = []
before_var = []
after_var = []
breakpoint_list = []
p_val = []
field_list = []
season_list = []
col_list = []

for field in field_names:
    for season in seasons:
        for col in cols:
            if field == 'Pool13' and (col == 'TP' or col == 'DO'):
                continue
            print(field,season,col)
            name = fields[field] + col + '.csv'
            df = pd.read_csv(name)
            df = df[df['SEASON']==season]
            before, after = split(df, breakpoints[field][col])
            temp_p_val = []
            
            field_list.append(field)
            season_list.append(season)
            col_list.append(col)
            breakpoint_list.append(breakpoints[field][col])
            before_avg.append(np.mean(before[col]))
            after_avg.append(np.mean(after[col]))
            before_var.append(np.var(before[col]))
            after_var.append(np.var(after[col]))
            for x in range(bootstrap):
                temp_p_val.append(stats.ttest_ind(a=before[col].sample(sample_size),b=after[col].sample(sample_size), equal_var=False)[1])
            p_val.append(sum(temp_p_val) / len(temp_p_val))

d = {'Field' : field_list, 'Season' : season_list, 'Variable' : col_list, 'p-Value': p_val, 'Breakpoint' : breakpoint_list, 'Before Average' : before_avg, 'After Average' : after_avg, 'Before Variance' : before_var, 'After Variance' : after_var}
results = pd.DataFrame(data=d)
results.to_csv('~/TDA_Final/T_Test/Results.csv', index=False)