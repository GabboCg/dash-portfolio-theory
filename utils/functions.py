#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:23:26 2022

@author: gabrielcabrera
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf

def GlobalMinVar(mu, cov):
    
    sigma_inv = np.linalg.inv(cov)
    ones = np.ones(len(mu)).reshape(-1, 1)
    
    m = sigma_inv.dot(ones) / ones.T.dot(sigma_inv).dot(ones)
    
    return(m)

def EfficientPortfolio(mu, cov):
    
    top_mat = np.concatenate((2 * cov.to_numpy(), np.array(mu).reshape(-1,1), np.ones(len(mu)).reshape(-1,1)), axis=1)
    mid_vec = np.concatenate((mu.to_numpy().reshape(1,-1), np.zeros((1, 2))), axis = 1)
    bot_vec = np.concatenate((np.ones((1, len(mu))), np.zeros((1, 2))), axis=1)
        
    A_mat = np.concatenate((top_mat, mid_vec, bot_vec), axis=0)
    
    b_vec = np.concatenate((np.zeros((1, len(mu))), np.array([mu.max(), 1]).reshape(1,-1)), axis=1)
    z = np.linalg.inv(A_mat).dot(b_vec.T)
    
    return(z[0:len(mu)])

def EfficientFrontier(mu, cov, m, z):
   
   frontier = [] 
   
   for i in list(range(-2, 21)):
       
       a = i / 10 
       
       zvec = a * m + (1 - a) * z
       
       mu_mz = zvec.T.dot(mu)[0]
       sigma_mz = np.sqrt(zvec.T.dot(cov).dot(zvec))[0][0]
    
       frontier.append([mu_mz, sigma_mz])

   frontier = pd.DataFrame(frontier, columns=['mu','sigma'])
   
   return(frontier)

def TangecyPortfolio(mu, cov, rf):
    
    sigma_inv = np.linalg.inv(cov)
    mu = mu.to_numpy().reshape(-1, 1)
    ones = np.ones(len(mu)).reshape(-1, 1)
    
    t = sigma_inv.dot(mu - rf) / ones.T.dot(sigma_inv).dot(mu - rf)
    
    return(t)

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope[0] * x_vals
    plt.plot(x_vals, y_vals, '--')
    
    return(x_vals, y_vals)

def generate_figure(tickers, start_date, end_date, rfree, frequency):
    
    stocks = yf.download(tickers, start_date, end_date, interval=frequency)

    ret = stocks['Close'].pct_change().dropna()

    mu = ret.mean()
    sigma2 = ret.var()
    cov =  ret.cov()
    
    m = GlobalMinVar(mu, cov)

    rp_ret_gmin = mu.dot(m)
    rp_var_gmin = m.T.dot(cov).dot(m)

    z = EfficientPortfolio(mu, cov)    

    frontier = EfficientFrontier(mu, cov, m, z)
    
    t = TangecyPortfolio(mu, cov, rfree)    

    rp_ret_tang = mu.dot(t)
    rp_var_tang = t.T.dot(cov).dot(t)
     
    slope = (rp_ret_tang - rfree) / np.sqrt(rp_var_tang)
          
    plt.plot(frontier.sigma, frontier.mu)
    plt.plot(np.sqrt(sigma2), mu, 'b*')
    plt.plot(np.sqrt(rp_var_gmin), rp_ret_gmin, 'go')
    plt.plot(np.sqrt(rp_var_tang), rp_ret_tang, 'ro')
    x_vals, y_vals = abline(slope, rfree) 
         
    symbol_template = go.layout.Template()
    
    fig1 = go.Figure(go.Scatter(x = frontier.sigma, 
                                y = frontier.mu, 
                                mode = 'lines',
                                line = dict(color = 'royalblue', 
                                            width = 3, 
                                            dash = 'solid'),
                                name = 'Efficient Frontier',
                                hovertemplate = 'Ret.: %{y:.2%}<extra></extra><br>Std.: %{x:.2f}'))
    
    fig1.add_trace(go.Scatter(x = np.sqrt(sigma2), 
                              y = mu,
                              text= list(mu.index), 
                              textposition = "bottom center",
                              marker = dict(symbol = "star", 
                                            size = 16,
                                            line = dict(width = 1,
                                                        color = 'DarkSlateGrey')),
                              mode = 'markers+text',
                              name = 'Assets',
                              hovertemplate = 'Ret.: %{y:.2%}<extra></extra><br>Std.: %{x:.2f}'))
    
    fig1.add_hline(y = np.array(rp_ret_gmin[0]), 
                   line_dash = "dot",
                   line_color = 'gray',
                   annotation_text = "Inefficient Portfolios", 
                   annotation_position = "bottom right")
    
    fig1.add_trace(go.Scatter(x = np.array(np.sqrt(rp_var_gmin)[0]), 
                              y = np.array(rp_ret_gmin[0]),
                              marker = dict(symbol = "diamond", 
                                            size = 16,
                                            line = dict(width = 1,
                                                        color = 'DarkSlateGrey')),
                              mode = 'markers',
                              name = 'Min. Variance',
                              hovertemplate = 'Ret.: %{y:.2%}<extra></extra><br>Std.: %{x:.2f}'))
    
    fig1.add_hrect(y0 = np.array(rp_ret_gmin[0]), 
                   y1 = -np.round(rp_ret_gmin[0], 2), 
                   line_width = 0, 
                   fillcolor = "red", 
                   opacity = 0.2)
    
    fig1.add_trace(go.Scatter(x = np.array(np.sqrt(rp_var_tang)), 
                              y = np.array(rp_ret_tang),
                              marker = dict(symbol = "square", 
                                            size = 16,
                                            line = dict(width = 1,
                                                        color = 'DarkSlateGrey')),
                              mode = 'markers',
                              name = 'Tangency',
                              hovertemplate = 'Ret.: %{y:.2%}<extra></extra><br>Std.: %{x:.2f}'))
    
    fig1.add_trace(go.Scatter(x = x_vals, 
                              y = y_vals, 
                              mode = 'lines',    
                              line = dict(color = 'firebrick', 
                                          width = 3, 
                                          dash = 'dash'),
                              name = 'Capital Market Line',
                              hovertemplate = 'Ret.: %{y:.2%}<extra></extra><br>Std.: %{x:.2f}'))
    
    fig1.update_layout(template = symbol_template,
                       xaxis_title = r'Std.',
                       yaxis_title = r'Return',
                       xaxis = dict(zeroline = False,
                                    showline = False),
                       yaxis = dict(zeroline = False,
                                    showline = False))
    
    fig1.update_traces(textposition = 'top center')
    fig1.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
    fig1.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)

    symbol_template = go.layout.Template()
    
    symbol_template = go.layout.Template()

    # Create traces
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = ret.dot(m).index, 
                              y = ret.dot(m).iloc[:,0],
                              mode = 'lines+markers',
                              name = 'Min. Variance'))
    
    fig2.add_trace(go.Scatter(x = ret.dot(z).index, 
                              y = ret.dot(z).iloc[:,0],
                              mode = 'lines+markers',
                              name = 'Efficient'))
    
    fig2.add_trace(go.Scatter(x = ret.dot(t).index, 
                              y = ret.dot(t).iloc[:,0],
                              mode = 'lines+markers', 
                              name = 'Tangecy'))
    
    fig2.update_traces(mode="markers+lines", hovertemplate='%{y:.2%}')
    fig2.update_layout(hovermode="x unified")
    fig2.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
    fig2.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
    
    df = pd.DataFrame(np.concatenate((m.T, z.T, t.reshape(1,-1)), axis = 0), columns = mu.index, index = ['Min. Variance', 'Efficient', 'Tangecy'])
    df = df.melt(ignore_index=False)
    df['port'] = df.index
    
    items = df['variable'].unique()
    data = []
    for item in items:
        group_item = df[df['variable']==item]
        data.append(
            go.Bar(
                x = group_item['port'],
                y = group_item['value'],
                name = str(item),
                hovertemplate = '%{y:.2%}'
            ) 
        )
    layout = go.Layout(barmode='group')
    fig3 = go.Figure(data = data, layout = layout)
    fig3.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
    fig3.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
    
    return(fig1, fig2, fig3)
