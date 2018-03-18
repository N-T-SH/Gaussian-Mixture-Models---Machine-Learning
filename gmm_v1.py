# Original Imports

#%matplotlib inline
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import stats
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import random
import math
from scipy.stats import ks_2samp

# Plotly

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

noMoves = 0
moves = 0
surprises = 0
i = 0

# Set initial values
random.seed(1)
max_iter = 100
# Nitesh - Should we randomise this?
agentR = (-25, -25) # initial location of agentR
agentB = (25, -25) # initial location of agentB
my_centers = ((-5, -5),(5, 5))
gmm = None
ks_df_OLD = None
X_old = None

def estimateMeanConvergence(noMoveClick, moveClick, surpriseMe):
    # move distributions?
    global noMoves
    global moves
    global surprises
    global i
    global my_centers
    global agentR
    global agentB
    global max_iter
    global gmm
    global ks_df_OLD
    global X_old
    
    if noMoveClick>noMoves:
        move = 0
        noMoves = noMoveClick
    elif moveClick>moves:
        move = 1
        moves = moveClick
    elif surpriseMe>surprises:
        move = random.randint(0, 1)
        surprises = surpriseMe
    else:
        print("Button press not detected")
        move = 0
    
    print(f"Did actually move? {(lambda move:'YES' if move == 1 else 'NO')(move)}")
    my_centers = ((my_centers[0][0] + 1*move, my_centers[0][1] + 4*move),
                  (my_centers[1][0] - 3*move, my_centers[1][1] - 1*move))
    print("iteration: %d"%i)
    # draw samples
    X, y_true = make_blobs(n_samples=10000, centers=my_centers,
                       cluster_std=1.5, random_state=i)
    
    
    # stack observations if new data from same distribution (in order to incorporate more data into estimate of mean).
    # otherwise, only use new data.
    #
    # note: assumes independence of x and y
    if i!= 0:
        # fit GMM using previous means as initial means
        gmm = GaussianMixture(n_components=2, means_init = gmm.means_,
                          max_iter=max_iter).fit(X)
        # extract predicted labels
        labels = gmm.predict(X)
        ks_df = pd.DataFrame(np.column_stack((X, labels)))
        ks_df.columns=['x1', 'x2', 'labels']
        print(ks_2samp(ks_df[ks_df['labels'] == 1]['x1'], ks_df_OLD[ks_df_OLD['labels'] == 1]['x1'])) # TEMP
        
        # if estimated underlying distributions have not changed, stack data.
        #
        # note: assumes independence of x and y (i.e. uses univariate ks test)
        if ks_2samp(ks_df[ks_df['labels'] == 1]['x1'], ks_df_OLD[ks_df_OLD['labels'] == 1]['x1'])[1] > .99 and\
            ks_2samp(ks_df[ks_df['labels'] == 1]['x2'], ks_df_OLD[ks_df_OLD['labels'] == 1]['x2'])[1] > .99 and\
            ks_2samp(ks_df[ks_df['labels'] == 0]['x1'], ks_df_OLD[ks_df_OLD['labels'] == 0]['x1'])[1] > .99 and\
            ks_2samp(ks_df[ks_df['labels'] == 0]['x2'], ks_df_OLD[ks_df_OLD['labels'] == 0]['x2'])[1] > .99:
                # stack observations
                X = np.vstack((X_old, X)) 
        
    # fit GMM
    gmm = GaussianMixture(n_components=2, init_params='kmeans',
                          max_iter=max_iter).fit(X)
    
    # extract predicted labels
    labels = gmm.predict(X)
    
    # create ks df for next iteration
    ks_df_OLD = pd.DataFrame(np.column_stack((X, labels)))
    ks_df_OLD.columns=['x1', 'x2', 'labels']
    
    # update target means for agents to seek
    B_mean_estimate, R_mean_estimate = tuple(gmm.means_[0]), tuple(gmm.means_[1])
    
    
    ## move agents towards respective estimated means 
    #
    # calc distances from agents to respective means
    distanceR = (sum((np.array(R_mean_estimate) - np.array(agentR))**2))**(1/2) # unweighted....no log prob...
    distanceB = (sum((np.array(B_mean_estimate) - np.array(agentB))**2))**(1/2) # unweighted....no log prob...
    
    # calculate angles
    angle_degreeR = math.degrees(math.atan2(R_mean_estimate[1] - agentR[1],
                                            R_mean_estimate[0] - agentR[0]))
    angle_degreeB = math.degrees(math.atan2(B_mean_estimate[1] - agentB[1],
                                            B_mean_estimate[0] - agentB[0]))
    
    # scale (if set to 1, agent will move all the way to mean of respective distribution) 
    scaleR = .5
    scaleB = .5
    
    # set new agent location (red)
    agentR = agentR[0] + scaleR * distanceR * math.cos(angle_degreeR * math.pi / 180),\
             agentR[1] + scaleR * distanceR * math.sin(angle_degreeR * math.pi / 180)
    # set new agent location (blue)
    agentB = agentB[0] + scaleB * distanceB * math.cos(angle_degreeB * math.pi / 180),\
             agentB[1] + scaleB * distanceB * math.sin(angle_degreeB * math.pi / 180)
    
    # make data copy for next iteration
    X_old = np.copy(X) 
    #Increment Counter
    i+=1

def plotData(ks_df_OLD, agentR, agentB):
    # Function that, given the distribution dataframe, and the agents
    # location returns plot information.
    
    dataRPlot = go.Scatter(
        x=ks_df_OLD[ks_df_OLD['labels'] == 0]['x1'],
        y=ks_df_OLD[ks_df_OLD['labels'] == 0]['x2'],
        #text=df[df['continent'] == i]['country'],
        mode='markers',
        opacity=0.7,
        marker={
            'color': 'pink',
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
        name='Red'
    )
    dataBPlot = go.Scatter(
        x=ks_df_OLD[ks_df_OLD['labels'] == 1]['x1'],
        y=ks_df_OLD[ks_df_OLD['labels'] == 1]['x2'],
        #text=df[df['continent'] == i]['country'],
        mode='markers',
        opacity=0.7,
        marker={
            'color': 'turquoise',
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
        name='Blue'
    )
    agentRPlot = go.Scatter(
        x=[agentR[0]],
        y=[agentR[1]],
        #text=df[df['continent'] == i]['country'],
        mode='markers',
        #opacity=0.7,
        marker={
            'color': 'red',
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
        name='Red Agent'
    )
    agentBPlot = go.Scatter(
        x=[agentB[0]],
        y=[agentB[1]],
        #text=df[df['continent'] == i]['country'],
        mode='markers',
        #opacity=0.7,
        marker={
            'color': 'blue',
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
        name='Blue Agent'
    )
    
    data = [dataRPlot, dataBPlot, agentRPlot, agentBPlot]
    
    layout =  go.Layout(
        xaxis={'title': 'X1'},
        yaxis={'title': 'X2'},
        height = 500, width =500,
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    
    fig = dict(data=data, layout=layout)
    return fig


# Invoking Dash app
app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(
        id='GMM with Agent Model'
    ),
    html.Button(id='no-move', n_clicks=0, children='Same Distribution'),
    html.Button(id='move', n_clicks=0, children='Different Distribution'),
    html.Button(id='surprise-move', n_clicks=0, children='I\'m Feeling Lucky!')
])
    
@app.callback(
    dash.dependencies.Output('GMM with Agent Model', 'figure'),
    [dash.dependencies.Input('no-move', 'n_clicks'),
     dash.dependencies.Input('move', 'n_clicks'),
     dash.dependencies.Input('surprise-move', 'n_clicks')]
)
def generateData(noMoveClick, moveClick, surpriseMe):
    # First Run
    print(noMoveClick, moveClick, surpriseMe)
    estimateMeanConvergence(noMoveClick, moveClick, surpriseMe)
    # Rename dataframe columns
    return plotData(ks_df_OLD, agentR, agentB)

if __name__ == '__main__':
    app.run_server(debug = False)