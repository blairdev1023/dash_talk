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

def regression_line(target):
    '''
    Returns the x and y-prediction points of the best-fit linear regression
    line for the given iris target
    '''

################################################################################
##################################### HTML #####################################
################################################################################

app = dash.Dash()
df = load_df()

app.layout = html.Div([
    html.H2('Iris Visualization'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-1',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[0]
            )
        ], style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div([
            dcc.Dropdown(
                id='graph-dropdown-2',
                options=[{'label': i, 'value': i} for i in df.columns[:4]],
                value=df.columns[1]
            )
        ], style={'width': '50%', 'float': 'right'}
        ),
        dcc.Checklist(
            id='regression-checks',
            options=[{'label': i, 'value': i} for i in range(3)],
            values=[1]
        ),
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

    color_dict = {0: 'red', 1: 'yellow', 2: 'blue'}
    data = []

    # markers
    trace = go.Scatter(
        x=df[col1],
        y=df[col2],
        mode='markers',
        marker={'size': 15,
                'opacity': 0.7,
                'color': [color_dict[i] for i in df['target']],
                'line': {'color': 'black', 'width': 0.5}
        }
    )
    data.append(trace)

    # regression line
    for val in checks:
        model = linear_model.LinearRegression()
        x = df[df['target'] == val]['sepal width (cm)'].values.reshape(-1,1)
        y = df[df['target'] == val]['petal length (cm)'].values.reshape(-1,1)
        model.fit(x, y)
        y_pred = model.predict(x)
        trace = go.Scatter(
            x=x,
            y=y_pred,
            mode='line'
        )
        print(trace)
        data.append(trace)

    layout = go.Layout(
        xaxis={'title': col1},
        yaxis={'title': col2},
        hovermode='closest',
        margin={'l': 0, 'b': 0, 't': 0, 'r': 0}
    )
    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
