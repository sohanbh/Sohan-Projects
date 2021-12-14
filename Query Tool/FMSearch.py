#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Lasso

df = pd.read_csv(r"C:\Users\sohan\OneDrive\Desktop\FM.csv")
#df = df.drop(columns = ['NationID', 'Born', 'IntCaps', 'IntGoals', 'U21Caps', 'U21Goals', 'Controversy', 'Temperament', 'Dirtiness', 'ImportantMatches',
       #'InjuryProness', 'Versatility', 'Adaptability', 'Ambition', 'Loyalty',
       #'Pressure', 'Professional', 'Sportsmanship'])
GK = df.loc[df['Goalkeeper'] >= 12]
Sweeper = df.loc[df['Sweeper'] >= 12]
ST = df.loc[df['Striker'] >= 12]
AMC = df.loc[df['AttackingMidCentral'] >= 12]
AML = df.loc[df['AttackingMidLeft'] >= 12]
AMR = df.loc[df['AttackingMidRight'] >= 12]
DC = df.loc[df['DefenderCentral'] >= 12]
DL = df.loc[df['DefenderLeft'] >= 12]
DR = df.loc[df['DefenderRight'] >= 12]
DM = df.loc[df['DefensiveMidfielder'] >= 12]
MC = df.loc[df['MidfielderCentral'] >= 12]
MR = df.loc[df['MidfielderRight'] >= 12]
ML = df.loc[df['MidfielderLeft'] >= 12]
WBL = df.loc[df['WingBackLeft'] >= 12]
WBR = df.loc[df['WingBackRight'] >= 12]

GK = GK.drop(columns = ['Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
Sweeper = Sweeper.drop(columns = ['Goalkeeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
ST = ST.drop(columns = ['Goalkeeper', 'Sweeper', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
AMC = AMC.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
AML = AML.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidRight', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
AMR = AMR.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'DefenderCentral', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
DC = DC.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderLeft', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
DL = DL.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderRight', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
DR = DR.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefensiveMidfielder', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
DM = DM.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'MidfielderCentral', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
MC = MC.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'DefensiveMidfielder', 'MidfielderLeft', 'MidfielderRight', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
MR = MR.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'DefensiveMidfielder', 'MidfielderLeft', 'MidfielderCentral', 
                        'MidfielderLeft', 'WingBackLeft', 'WingBackRight'])
ML = ML.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'DefensiveMidfielder', 'MidfielderRight', 'MidfielderCentral', 
                       'WingBackLeft', 'WingBackRight'])
WBL = WBL.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'DefensiveMidfielder', 'MidfielderRight', 'MidfielderCentral', 
                       'MidfielderLeft', 'WingBackRight'])
WBR = WBR.drop(columns = ['Goalkeeper', 'Sweeper', 'Striker', 'AttackingMidCentral', 'AttackingMidLeft', 'AttackingMidRight', 'DefenderCentral', 
                       'DefenderLeft', 'DefenderRight', 'DefensiveMidfielder', 'MidfielderRight', 'MidfielderCentral', 
                       'MidfielderLeft', 'WingBackLeft'])

PositionTables = [GK, Sweeper, ST, AMC, AML, AMR, DC, DL, DR, DM, MC, MR, ML, WBL, WBR]
dictionary = {'GK': GK, 'Sweeper' : Sweeper, 'ST':ST, 'AMC':AMC, 'AML':AML, 'AMR':AMR, 'DC':DC, 'DL':DL, 'DR':DR, 
             'DM':DM, 'MC':MC, 'MR':MR, 'ML':ML, 'WBL':WBL, 'WBR':WBR}

regrdata = dict()
for key in dictionary:
    X = dictionary[key].drop(columns = ['UID', 'Name', 'Age', 'Height', 'Weight', 'PositionsDesc', 'LeftFoot', 'RightFoot', 'NationID', 'Born', 'IntCaps', 'IntGoals', 'U21Caps', 'U21Goals', 'Controversy', 'Temperament', 'Dirtiness', 'ImportantMatches',
                                        'InjuryProness', 'Versatility', 'Adaptability', 'Ambition', 'Loyalty',
           'Pressure', 'Professional', 'Sportsmanship'])
    Y = X.iloc[:,-1:]
    X = X.iloc[:, :-1] 
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    lin = Lasso(alpha=0.0001,precompute=True,max_iter = 10000,
            positive=True, random_state=9999, selection='random')
    regrdata[key] = lin.fit(X,Y)


for key in dictionary:
    X = dictionary[key].drop(columns = ['UID', 'Name', 'Age', 'Height', 'Weight', 'PositionsDesc', 'LeftFoot', 'RightFoot', 'NationID', 'Born', 'IntCaps', 'IntGoals', 'U21Caps', 'U21Goals', 'Controversy', 'Temperament', 'Dirtiness', 'ImportantMatches',
       'InjuryProness', 'Versatility', 'Adaptability', 'Ambition', 'Loyalty',
       'Pressure', 'Professional', 'Sportsmanship'])
    Y = X.iloc[:,-1:]
    X = X.iloc[:, :-1]
    X.shape[0]
    values = np.zeros(X.shape[0])
    for x in range(0, X.shape[0]):
        values[x] = regrdata[key].predict([X.iloc[x].to_numpy()])
    dictionary[key]['values'] = values


# In[122]:


def GetTopK(Position = 'GK', k = 4, Attributes = 'values', Age = 100, Height = 0, IntCaps = 0, IntGoals = 0, 
           U21Caps = 0, U21Goals = 0, Weight = 500):
    arr = Attributes.split(', ')
    table = dictionary[Position].sort_values(['values'], ascending = [False])
    table = table[(table.Age <= Age) & (table.Height >= Height) & (table.IntCaps >= IntCaps) & (table.IntGoals >= IntGoals)
                 & (table.U21Caps >= U21Caps) & (table.U21Goals >= U21Goals) & (table.Weight <= Weight)]
    table['newvalue'] = 0
    table['newvalue'] = table['values'] + (table[arr].sum(axis = 1))/len(arr)
    table = table.sort_values(['newvalue'], ascending = [False])
    return table[:k].drop(columns = ['values', 'newvalue'])


# In[110]:


import matplotlib.pyplot as plt
from sklearn import preprocessing

Abbreviations = {'GK': 'Goalkeeper', 'Sweeper' : 'Sweeper', 'ST': 'Striker', 'AMC': 'AttackingMidCentral'
               , 'AML': 'AttackingMidLeft', 'AMR': 'AttackingMidRight', 'DC': 'DefenderCentral', 'DL': 'DefenderLeft', 
               'DR': 'DefenderRight', 
             'DM':'DefensiveMidfielder', 'MC': 'MidfielderCentral', 'MR': 'MidfielderRight', 'ML': 'MidfielderLeft', 
               'WBL': 'WingBackLeft', 'WBR': 'WingBackRight'}


def getDistribution(WhichPosition = ''):
    
    g = GetTopK(Position = WhichPosition, k = 200)
    
    players = g.drop(['Name','UID', 'NationID', 'Born', 'IntCaps', 'IntGoals', 'U21Caps', 'U21Goals', 'PositionsDesc',
                   'Consistency', 'Dirtiness', 'ImportantMatches', 'Versatility', 'Adaptability', 'Ambition',
                   'Loyalty', 'Pressure', 'Professional', 'Sportsmanship', 'Temperament', 'Controversy',
                   'Age', 'Weight', 'InjuryProness'
            ], axis=1)
    X = players.loc[:,:'Strength'].drop(['RightFoot', 'LeftFoot'], axis=1)
    X_foot = players.loc[:, ['RightFoot', 'LeftFoot']]
    y = players.loc[:,Abbreviations[WhichPosition]:]
    scaler = preprocessing.MinMaxScaler()
    vectors = X.values
    scaled_rows = scaler.fit_transform(vectors.T).T
    X_normalized = pd.DataFrame(data = scaled_rows, columns = X.columns)
    fig, axes = plt.subplots(len(X_normalized.columns)//3, 3, figsize=(12, 48))

    i = 0
    for triaxis in axes:
        for axis in triaxis:
            X.hist(column = X_normalized.columns[i], bins = 100, ax=axis)
            i = i+1


# In[119]:


getDistribution('ST')


# In[139]:


GetTopK(Position = 'ST', k = 10)


# In[140]:


GetTopK(Position = 'ST', k = 10, Attributes = 'Pace, Acceleration')


# In[141]:


GetTopK(Position = 'ST', k = 10, Attributes = 'Finishing, LongShots')


# In[145]:


GetTopK(Position = 'ST', k = 10, Attributes = 'Bravery, Anticipation, Composure', Age = 29)

