from flask import Flask
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model

################################################################################
##################################### FUNC #####################################
################################################################################

def load_df():
    '''
    Returns a pandas df of the iris dataset
    '''
    data = datasets.load_iris().data
    cols = datasets.load_iris().feature_names
    df = pd.DataFrame(data=data, columns=cols)
    df['target'] = datasets.load_iris().target
    return df

################################################################################
##################################### HTML #####################################
################################################################################

server = Flask(__name__)
app = dash.Dash(name=__name__, server=server)
df = load_df()

app.layout = html.Div([
    html.H2('Iris Visualization'),
    html.Div([
        # Feature Selector 1
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-1',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[0]
            )
        ], style={'width': '50%', 'display': 'inline-block'}
        ),
        # Feature Selector 2
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-2',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[1]
            )
        ], style={'width': '50%', 'float': 'right'}
        ),
        # Regression Checklist
        html.Div([
            html.Label('Regression Target'),
            dcc.Checklist(
                id='regression-checks',
                options=[{'label': i, 'value': i} for i in range(3)],
                values=[0]
            )
        ]),
        # Main Plot
        dcc.Graph(id='main-graph', config={'displayModeBar': False})
    ],
    style={'width': '70%',
           'margin-left': 'auto',
           'margin-right': 'auto'
    })
])

################################################################################
################################### CALLBACK ###################################
################################################################################

@app.callback(
    Output('main-graph', 'figure'),
    [Input('graph-dropdown-1', 'value'),
    Input('graph-dropdown-2', 'value'),
    Input('regression-checks', 'values')])
def graph_maker(col1, col2, checks):
    '''
    Returns the figure dict for main plot
    '''

    colors = ['red', 'green', 'blue']
    names = datasets.load_iris().target_names
    data = []

    # markers
    for target in df['target'].unique():
        trace = go.Scatter(
            x=df[df['target'] == target][col1],
            y=df[df['target'] == target][col2],
            mode='markers',
            name=names[target],
            marker=dict(size=15,
                    opacity= 0.7,
                    color=colors[target],
                    line={'color': 'black', 'width': 0.5}
            )
        )
        data.append(trace)

    # regression line
    for val in checks:
        model = linear_model.LinearRegression()
        x = df[df['target'] == val][col1]
        y = df[df['target'] == val][col2]
        model.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))
        y_pred = model.predict(x.values.reshape(-1,1))
        y_pred = [y[0] for y in y_pred]
        trace = go.Scatter(
            x=x,
            y=y_pred,
            mode='line',
            line={'color': colors[val]},
            showlegend=False
        )
        data.append(trace)

    layout = go.Layout(
        xaxis={'title': col1},
        yaxis={'title': col2},
        hovermode='closest',
        margin={'l': '5%', 'b': '5%', 't': 0, 'r': 0},
    )

    return {'data': data, 'layout': layout}
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model

################################################################################
##################################### FUNC #####################################
################################################################################

def load_df():
    '''
    Returns a pandas df of the iris dataset
    '''
    data = datasets.load_iris().data
    cols = datasets.load_iris().feature_names
    df = pd.DataFrame(data=data, columns=cols)
    df['target'] = datasets.load_iris().target
    return df

################################################################################
##################################### HTML #####################################
################################################################################

app = dash.Dash()
df = load_df()

app.layout = html.Div([
    html.H2('Iris Visualization'),
    html.Div([
        # Feature Selector 1
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-1',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[0]
            )
        ], style={'width': '50%', 'display': 'inline-block'}
        ),
        # Feature Selector 2
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-2',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[1]
            )
        ], style={'width': '50%', 'float': 'right'}
        ),
        # Regression Checklist
        html.Div([
            html.Label('Regression Target'),
            dcc.Checklist(
                id='regression-checks',
                options=[{'label': i, 'value': i} for i in range(3)],
                values=[0]
            )
        ]),
        # Main Plot
        dcc.Graph(id='main-graph', config={'displayModeBar': False})
    ],
    style={'width': '70%',
           'margin-left': 'auto',
           'margin-right': 'auto'
    })
])

################################################################################
################################### CALLBACK ###################################
################################################################################

@app.callback(
    Output('main-graph', 'figure'),
    [Input('graph-dropdown-1', 'value'),
    Input('graph-dropdown-2', 'value'),
    Input('regression-checks', 'values')])
def graph_maker(col1, col2, checks):
    '''
    Returns the figure dict for main plot
    '''

    colors = ['red', 'green', 'blue']
    names = datasets.load_iris().target_names
    data = []

    # markers
    for target in df['target'].unique():
        trace = go.Scatter(
            x=df[df['target'] == target][col1],
            y=df[df['target'] == target][col2],
            mode='markers',
            name=names[target],
            marker=dict(size=15,
                    opacity= 0.7,
                    color=colors[target],
                    line={'color': 'black', 'width': 0.5}
            )
        )
        data.append(trace)

    # regression line
    for val in checks:
        model = linear_model.LinearRegression()
        x = df[df['target'] == val][col1]
        y = df[df['target'] == val][col2]
        model.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))
        y_pred = model.predict(x.values.reshape(-1,1))
        y_pred = [y[0] for y in y_pred]
        trace = go.Scatter(
            x=x,
            y=y_pred,
            mode='line',
            line={'color': colors[val]},
            showlegend=False
        )
        data.append(trace)

    layout = go.Layout(
        xaxis={'title': col1},
        yaxis={'title': col2},
        hovermode='closest',
        margin={'l': '5%', 'b': '5%', 't': 0, 'r': 0},
    )

    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
