#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:33:27 2021

@author: abcd
"""

import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LinearRing

def coordinates_to_field(df):
    a = 6378.137
    e = 0.0818192
    k0 = 0.9996
    E0 = 500
    N0 = 0
   # lon = 116.3124683;  经度
    # lat = 40.02493832; 纬度
    lon1 = []
    lat1 = []
    lons = df['longitude'].to_list()
    lats = df['latitude'].to_list()
    for i in range(len(df)):
        lon = lons[i]
        lat = lats[i]
        Zonenum = int(lon / 6) + 31
        lamda0 = (Zonenum - 1) * 6 - 180 + 3
        lamda0 = lamda0 * math.pi / 180
        phi = lat * math.pi / 180
        lamda = lon * math.pi / 180
        v = 1 / math.sqrt(1 - e ** 2 * math.sin(phi) ** 2)
        A = (lamda - lamda0) * math.cos(phi)
        T = math.tan(phi) ** 2
        C = e ** 2 * math.cos(phi) * math.cos(phi) / (1 - e ** 2)
        s = (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * phi - \
            (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * phi) + \
            (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * phi) - \
            35 * e ** 6 / 3072 * math.sin(6 * phi)
        UTME = E0 + k0 * a * v * (A + (1 - T + C)*A ** 3 / 6+(5 - 18 * T + T ** 2) * A ** 5 / 120)
        UTMN = N0 + k0 * a * (s + v * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2) * A ** 6 / 720))
        # UTME,UTMN based on Kilometres. Converting them into Meters
        UTME = UTME * 1000
        UTMN = UTMN * 1000
        lat1.append(UTME)
        lon1.append(UTMN)
    print (lon1)
    print (lat1)
    return pd.DataFrame({'X': lon1
                        , 'Y': lat1})

def VexRotated (vertex, RM):
    rotation = np.dot(vertex, RM)
    return rotation[0,0], rotation[0,1]

def read_match_data(MATCHDATADIR):
    '''
    read match (SSG) match information: Date, Category, Format, Team in SSGs, Player Name, Split Start Time, Split End Time
    '''
    match = pd.read_excel(MATCHDATADIR, sheet_name = 1, usecols = ['Date', 'Category', 'Format','Team in SSGs', 'Player Name', 'Split Start Time', 'Split End Time'])
    return match

def TeamTracking(file_dir, teamname, StartTS, EndTS, RM):
    file_list = os.listdir(os.path.join(file_dir, teamname))
    print (f"{file_list}/n")
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    # Create a DataFrame for later use
    TeamPosition = pd.DataFrame()
    
    for file in file_list:
        print (file)
        path = os.path.join(file_dir, teamname, file)
        position = pd.read_csv(path, usecols = [0, 2, 3])
        position['Excel Timestamp'] = position['Excel Timestamp'].round(6)
        StartIndex = position.loc[position['Excel Timestamp'] == StartTS].index[0] # Getting StartIndex from position data
        EndIndex = position.loc[position['Excel Timestamp'] == EndTS].index[-1] # Getting EndIndex from position data
        position = position.iloc[StartIndex:EndIndex+1,:] # Subsetting
        print (position)
        ###
        ### Lat & Lon to X & Y
        a = 6378.137
        e = 0.0818192
        k0 = 0.9996
        E0 = 500
        N0 = 0
        lon1 = []
        lat1 = []
        lons = position[' Longitude'].to_list()
        lats = position[' Latitude'].to_list()
        for k in range(len(position)):
            lon = lons[k]
            lat = lats[k]
            Zonenum = int(lon / 6) + 31
            lamda0 = (Zonenum - 1) * 6 - 180 + 3
            lamda0 = lamda0 * math.pi / 180
            phi = lat * math.pi / 180
            lamda = lon * math.pi / 180
            v = 1 / math.sqrt(1 - e ** 2 * math.sin(phi) ** 2)
            A = (lamda - lamda0) * math.cos(phi)
            T = math.tan(phi) ** 2
            C = e ** 2 * math.cos(phi) * math.cos(phi) / (1 - e ** 2)
            s = (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * phi - \
                (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * phi) + \
                (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * phi) - \
                35 * e ** 6 / 3072 * math.sin(6 * phi)
            UTME = E0 + k0 * a * v * (A + (1 - T + C)*A ** 3 / 6+(5 - 18 * T + T ** 2) * A ** 5 / 120)
            UTMN = N0 + k0 * a * (s + v * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2) * A ** 6 / 720))
            # UTME,UTMN based on Kilometres. Converting them into Meters
            UTME = UTME * 1000
            UTMN = UTMN * 1000
            lat1.append(UTME)
            lon1.append(UTMN)
             #print (lat1)
             #print (lon1)
        position["X"] = lon1
        position["Y"] = lat1
        ###
        ### apply RM to LAT & LON
        for i in range (len(position)):
            pos = position.iloc[[i], [3,4]]
            position.iloc[[i], [3, 4]] = np.dot(pos, RM)
        ### switch X and Y?
        position[["X", "Y"]] = position[["Y", "X"]]
        ###
        ### Whether each Timestamp is unique?
        if len(position["Excel Timestamp"].unique()) != len(position):
            print ("!!! Same Timestamp occurs !!!")
        ###
        ### drop lan & lon
        position.drop(columns=[" Latitude", " Longitude"], inplace = True)
        ###
        ### amend column name
        playername = file[12:16]
        position.columns = ["Timestamp", "{}_x".format(playername), "{}_y".format(playername)]
        ###
        ### merging players in the same team together
        if file_list.index(file) == 0:
            TeamPosition = position
        else:
            TeamPosition = pd.merge(TeamPosition, position, on = 'Timestamp', how = 'outer')
    ###
    ### Modifying column name to "PLAYERNAME_x" or "PLAYERNAME_y"   !!! probably meeting problems due to different column number !!!
    TeamPosition.sort_values(by = 'Timestamp', axis=0, ascending = True, inplace = True)
    ### reset Frame
    TeamPosition.reset_index(drop = True)
    print (TeamPosition)
    return TeamPosition

# 1) reading match info, selecting useful columns

MATCHDATADIR = '/Users/abcd/Documents/Liverpool/positional data/Match/Z_SSG_DataBase_1.xlsx'

match_info = read_match_data(MATCHDATADIR)
match_info.to_excel("SSG_1.xlsx")

# 2) reading pitch info

## BSU Pitch
#df = pd.read_excel("/Users/abcd/Documents/Liverpool/positional data/Pitch/BSUpitch#3.xlsx")

## Spanish Academy Pitch
df = pd.read_excel("/Users/abcd/Documents/Liverpool/positional data/Pitch/Spanish Academy Pitch.xlsx")

# 3) Covert Lat & Lon to 2D Coordinates, and plot

ini_xyco_pitch = coordinates_to_field(df) #coordinates(x, y) of pitch, a dataframe
#print(ini_xyco_pitch)

# plot the pitch (explicitly closed polygon)
ini_pitch_x, ini_pitch_y = LinearRing(zip(ini_xyco_pitch['X'], ini_xyco_pitch['Y'])).xy
fig = plt.figure()
plt.plot(ini_pitch_x, ini_pitch_y, "g", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


# 4) Rotation

""" Rotation Matrix """

'''
1) find the Origin, left below one of 4 vertices
2) find the other vertex that should be on x-axis
2) calculate the angle
3) matrix
4) apply into other vertices
'''
### Origin and the other vertex
ini_xyco_pitch.sort_values(by = 'Y', axis=0, ascending = True, inplace = True)
Origin = ini_xyco_pitch[0:2].sort_values(by = 'X', axis=0, ascending=True).iloc[[0]]
print (Origin)
TheOther = ini_xyco_pitch[0:2].sort_values(by = 'X', axis=0, ascending=True).iloc[[-1]] # the other vertex on x-axis
print (TheOther)
ThirdVex = ini_xyco_pitch[2:3]
print (ThirdVex)
FourthVex = ini_xyco_pitch[3:4]
print (FourthVex)

### angle
#arr_x = ini_xyco_pitch.sort_values(by = 'Y', axis=0, ascending = True)[0:2]['X']
#arr_y = ini_xyco_pitch.sort_values(by = 'Y', axis=0, ascending = True)[0:2]['Y']
dx = abs(float(TheOther['X']) - float(Origin['X']))
dy = abs(float(TheOther['Y']) - float(Origin['Y']))
angle = np.arctan2(dy, dx) * 180 / np.pi

### matrix
#rotationmatrix = np.array([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180), 0],
#                          [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180), 0],
#                          [0, 0, 1]])

# Rotation Matrix for clockwise rotating (RW_CW)
RM_CW = np.array([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180)],
                  [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])

# Rotation Matrix for counter-clockwise rotating (RW_CCW)
RM_CCW = np.array([[np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)],
                  [-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])

# Clockwise or Counterclockwise rotating
if float(Origin['Y']) < float(TheOther['Y']):
    RotationMatrix = RM_CW
else:
    RotationMatrix = RM_CCW

### apply
#def VexRotated (vertex, RM):
#    rotation = np.dot(vertex, RM)
#    return rotation[0,0], rotation[0,1]
    
PitchRotated = pd.DataFrame(columns = ['X', 'Y'])
for vex in (Origin, TheOther, ThirdVex, FourthVex):
    print (f"VERTEX: {vex}")
    PitchRotated.loc[len(PitchRotated)] = VexRotated(vex, RotationMatrix)
    print ("RESULT:", PitchRotated)
print (PitchRotated) # Get rotated pitch vextices

# If switching current X Y is needed? 
if PitchRotated['X'].max()-PitchRotated['X'].min() < PitchRotated['Y'].max()-PitchRotated['Y'].min() :
    PitchRotated[["X","Y"]] = PitchRotated[["Y","X"]]
    print ("! Pitch X,Y Switched !")

pitch_x, pitch_y = LinearRing(zip(PitchRotated['X'], PitchRotated['Y'])).xy
fig = plt.figure()
plt.plot(pitch_x, pitch_y, "g", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

""" --- END --- """


""" Processing positional data """
'''
1) Get the initial team position dataset
2) Spruce up, Fill in the NaN value


'''

file_dir = '/Users/abcd/Documents/Liverpool/positional data/Position/'

### Getting Start & End TimeStamp for all players
### From Match Data
playernum = 5

###
team1 = 'TEAM_B'
team2 = 'TEAM_C'

StartTS = float(match_info.loc[0:playernum-1, ['Split Start Time']].max().round(6))
EndTS = float(match_info.loc[0:playernum-1, ['Split End Time']].min().round(6))
RM = RotationMatrix

Team1 = TeamTracking(file_dir, team1, StartTS, EndTS, RM)
# Cast [Timestamp] into (int), for following merging
Team1["Timestamp"] = Team1["Timestamp"].map(lambda x: x*1000000).astype(int)
#Team1.to_csv("{}_tracking.csv".format(team1), index = 0, na_rep = "NA")

Team2 = TeamTracking(file_dir, team2, StartTS, EndTS, RM)
# Cast [Timestamp] into (int), for following merging
Team2["Timestamp"] = Team2["Timestamp"].map(lambda x: x*1000000).astype(int)
#Team2.to_csv("{}_tracking.csv".format(team2), index = 0, na_rep = "NA")

##
Team1.reset_index(drop = True, inplace = True)
Team2.reset_index(drop = True, inplace = True)

## Create a timeline of 10 Hz, the generated float not rigid, errors occur when merging
Time = pd.DataFrame({"Timestamp": list(range(int(StartTS*1000000), int(EndTS*1000000)+1, 1))})
## Create a new timeline start from 0.1s (10 Hz)
Time["Start [s]"] = Time.index.map(lambda x: (x+1)*0.1)

### Fill the NaN value
## count the number of rows with NaN value (not completely missing sampling)
print (f"Non-completely missing sampling of {team1}: {len(Team1[Team1.isnull().T.any()])}")
print (f"Non-completely missing sampling of {team2}: {len(Team2[Team2.isnull().T.any()])}")

## How many rows of missing data (completely missing sampling)
print (f"missed sampling of {team1}: {len(Time) - len(Team1)}")
print (f"missed sampling of {team2}: {len(Time) - len(Team2)}")
### How many rows of missing data (completely missing sampling)
#print (f"missed sampling of {team1}: {int(EndTS * 1000000 - StartTS * 1000000) + 1 - len(Team1)}")
#print (f"missed sampling of {team2}: {int(EndTS * 1000000 - StartTS * 1000000) + 1 - len(Team2)}")

## Merging with new timeline
Team1_10Hz = pd.merge(Time, Team1, on = "Timestamp", how = "outer")
Team2_10Hz = pd.merge(Time, Team2, on = "Timestamp", how = "outer")

## Interpolation
Team1_10Hz.interpolate(method="linear", limit_direction="forward", inplace=True, axis=0)
Team2_10Hz.interpolate(method="linear", limit_direction="forward", inplace=True, axis=0)

## Preprocessing end, output
Team1_10Hz.to_csv("{}_10Hz.csv".format(team1), index=0, na_rep="NA")
Team2_10Hz.to_csv("{}_10Hz.csv".format(team2), index=0, na_rep="NA")

###

""" --- END --- """


""" --- Visualization ---"""

# plot player
first_500_TS = Team1_10Hz.iloc[[0, 100, 200, 300, 400, 500], [4, 5]]
plt.scatter(first_500_TS["Cond_x"], first_500_TS["Cond_y"])
traj_500_TS = Team1_10Hz.iloc[0: 501, [3, 4]]
plt.scatter(traj_500_TS["Cond_x"], traj_500_TS["Cond_y"])
