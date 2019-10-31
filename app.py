#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:43:44 2019

@author: fabiocampos
"""

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction

import pandas as pd
import numpy
from sklearn import cluster
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

enem = pd.read_csv("enemescola_assig8.csv")
enem.head()
train_data=enem[enem['ano']==2014]
test_data=enem[enem['ano']==2015]

train_data = train_data.dropna(axis=0)
test_data = test_data.dropna(axis=0)

predictors_train = train_data[['formadoc','perm','aprov','aband']]
label_train = train_data[['FIES']].values

predictors_test = test_data[['formadoc','perm','aprov','aband']]
label_test = test_data[['FIES']].values

predictors_train = predictors_train.dropna(axis=0)
predictors_test = predictors_test.dropna(axis=0)

from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()                       
model.fit(predictors_train, label_train.ravel())                  
predicted_labels = model.predict(predictors_test)

from sklearn.metrics import accuracy_score
accuracy_score(label_test, predicted_labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns;

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

enem1415 = pd.merge(train_data,test_data,on='co_escola',how='inner')
predictors = enem1415[['formadoc_x','perm_x','aprov_x','aband_x']]
label = enem1415[['FIES_y']].values.ravel()

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, predictors, label.ravel(), cv=10)
scores.mean()

from sklearn.svm import SVC 
model = SVC(gamma='scale')
model.fit(predictors,label)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.config.suppress_callback_exceptions = False

app.layout = html.Div([
        
    html.H1('Predicting FIES funding from student performance'),
        
    html.Div([   
#    html.Label('Municipality'),
#    dcc.Slider(id='p1',
#            min=0, max=100, step=5, value=0,
#               marks={
#        0: {'label': '0'},
#        25: {'label': '25'},
#        100: {'label': '100'},
#        50: {'label': '50'},
#        75: {'label': '75'}                                
#    }),

    html.Div([   
    html.Label('% of teachers trained'),
    dcc.Slider(id='p2',
            min=0, max=100, step=1, value=0,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        100: {'label': '100'},
        50: {'label': '50'},
        75: {'label': '75'}                                
    }),
]),

    html.Div([   
    html.Label('% of permanencia'),
    dcc.Slider(id='p3',
            min=0, max=100, step=1, value=0,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        100: {'label': '100'},
        50: {'label': '50'},
        75: {'label': '75'}                                
    }),
]),

    html.Div([   
    html.Label('Approval Rate (%)'),
    dcc.Slider(id='p4',
            min=0, max=100, step=1, value=0,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        100: {'label': '100'},
        50: {'label': '50'},
        75: {'label': '75'}                                
    }),
]),

#    html.Div([   
#    html.Label('Failure Rate (%)'),
#    dcc.Slider(id='p5',
#            min=0, max=100, step=1, value=0,
#               marks={
#        0: {'label': '0'},
#        25: {'label': '25'},
#        100: {'label': '100'},
#        50: {'label': '50'},
#        75: {'label': '75'}                                
#    }),
#]),

    html.Div([   
    html.Label('Drop-Out Rate (%)'),
    dcc.Slider(id='p6',
            min=0, max=100, step=1, value=0,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        100: {'label': '100'},
        50: {'label': '50'},
        75: {'label': '75'}                                
    }),
]),

],className="pretty_container four columns"),

  html.Div([ 

    daq.Gauge(
        id='t',
        showCurrentValue=True,
        color={"gradient":True,"ranges":{"red":[0,0.4],"yellow":[0.4,0.7],"green":[0.7,1]}},
        label="1=Yes // 0=No",
        max=1,
        min=0,
        value=1
    ),
])
    ])


@app.callback(
    Output('t', 'value'),
    [Input('p2', 'value'),
     Input('p3', 'value'),
     Input('p4', 'value'),
     Input('p6', 'value'),
     ])
def update_output_div(pa2,
                      pa3,
                      pa4,
                      pa6):
   X_case =pd.DataFrame({'formadoc':[pa2],'perm':[pa3],'aprov':[pa4],'aband':[pa6]})
   Y_case = model.predict(X_case)
   print("Test")
   return Y_case[0]

if __name__ == '__main__':
    app.run_server()