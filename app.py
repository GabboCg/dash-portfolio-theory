#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:17:46 2022

@author: gabrielcabrera
"""

# =============================================================================
# libraries
# =============================================================================

import yfinance as yf
import numpy as np
import pandas as pd 
from datetime import date

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash import dash, dash_table
from dash.exceptions import PreventUpdate

from utils.functions import *

# =============================================================================
# Dash app
# =============================================================================

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H3("Portfolio Theory"),
    dcc.Dropdown(['FB', 'AAPL', 'AMZN', 'NFLX', 'TSLA'],  
                 multi = True, 
                 id = 'dropdown', 
                 style = {"width": "50%"}, 
                 placeholder = "Select a ticker..."),
    html.Br(),
    dcc.RadioItems(['Daily', 'Weekly','Monthly'], 
                   'Daily', 
                   inline = True, 
                   id='items'),
    html.Br(),
    dcc.Input(type = 'number',
              placeholder = "Risk Free",
              debounce=True,
              id='input-1'),
    html.Br(),
    html.Br(),
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed = date(2000, 1, 1),
        max_date_allowed = date.today(),
    ),
    html.Br(),
    html.Br(),
    html.Button(id = 'submit-val', n_clicks = 0, children = 'Submit'),
    html.Br(),
    html.Div(children = [dcc.Graph(id = 'fig-1', 
                                   style = {'width': '850px', 'height': '600px'})],
             style = {'display': 'inline-block', 'margin-left': '50px'}),
    html.Div(children = [dcc.Graph(id = 'fig-2', 
                                   style = {'width': '850px', 'height': '600px'})],
             style={'display': 'inline-block', 'margin-left': '50px'}),
    dcc.Graph(id = 'fig-3', 
              style = {'height': '600px'}),
    html.Div(id = 'container-button-basic'),
])

# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output('fig-1', 'figure'),
    Output('fig-2', 'figure'),
    Output('fig-3', 'figure'),
    Input('dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('items', 'value'),
    Input('input-1', 'value'),
    Input('submit-val', 'n_clicks')
    )
def update_figure(tickers, start_date, end_date, interval, rfree, n_clicks):
    
    if n_clicks == 0:
        raise PreventUpdate
    else:
        if interval == 'Daily':
            frequency = '1d'
        elif interval == 'Weekly':
            frequency = '1wk'
        else:
            frequency = '1mo'
        fig1, fig2, fig3 = generate_figure(tickers, start_date, end_date, rfree, frequency)
        return(fig1, fig2, fig3) 

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
