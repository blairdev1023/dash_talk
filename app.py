import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np





################################################################################
##################################### HTML #####################################
################################################################################

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        html.H2('Iris Visualization'),
    ]),
])




################################################################################
################################### CALLBACK ###################################
################################################################################





if __name__ == '__main__':
    app.run_server(debug=True)
