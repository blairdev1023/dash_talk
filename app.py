import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from sklearn import datasets

################################################################################
##################################### FUNCS ####################################
################################################################################

def load_df():
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
    dcc.Dropdown(
        id='graph-dropdown-1',
        options=[{'label': i, 'value': i} for i in df.columns],
        value=df.columns[0]
    ),
    dcc.Dropdown(
        id='graph-dropdown-2',
        options=[{'label': i, 'value': i} for i in df.columns],
        value=df.columns[1]
    ),
    dcc.Graph(id='main-graph')
])

################################################################################
################################### CALLBACK ###################################
################################################################################

@app.callback(
    Output('main-graph', 'figure'),
    [Input('graph-dropdown-1', 'value'),
    Input('graph-dropdown-2', 'value')])
def graph_maker(col1, col2):
    '''
    Returns the figure dict for main plot
    '''

    data = []
    trace = go.Scatter(
        x=df[col1],
        y=df[col2],
        mode='markers',
        # name=col,
        # line={'shape': 'spline'}
    )
    data.append(trace)

    layout = go.Layout(
        xaxis={'title': col1},
        yaxis={'title': col2},
    )
    return {'data': data, 'layout': layout}


if __name__ == '__main__':
    app.run_server(debug=True)
