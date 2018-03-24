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
    data = datasets.load_boston().data
    cols = datasets.load_boston().feature_names
    df = pd.DataFrame(data=data, columns=cols)
    df['target'] = datasets.load_boston().target
    return df

################################################################################
##################################### HTML #####################################
################################################################################

app = dash.Dash()
df = load_df()

app.layout = html.Div([
    html.H2('Boston Visualization'),
    dcc.Dropdown(
        id='graph-dropdown',
        options=[{'label': i, 'value': i} for i in df.columns],
        multi=True,
        value=df.columns[0]
    ),
    dcc.Graph(id='main-graph')
])

################################################################################
################################### CALLBACK ###################################
################################################################################

@app.callback(
    Output('main-graph', 'figure'),
    [Input('graph-dropdown', 'value')])
def graph_maker(cols):
    '''
    Main graph for app.
    '''
    data = []
    for col in cols:
        trace = go.Scatter(
            x=[n for n in range(df[col].size)],
            y=df[col],
            mode='lines',
            name=col,
            line={'shape': 'spline'}
        )
        data.append(trace)

    layout = go.Layout(
        yaxis={'title': 'thingy'},
    )
    return {'data': data, 'layout': layout}


if __name__ == '__main__':
    app.run_server()
