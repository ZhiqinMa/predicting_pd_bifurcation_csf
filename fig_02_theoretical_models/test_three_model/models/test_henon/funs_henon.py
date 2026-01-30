#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:51:19 2022

Generic functions stored here

@author: tbury
"""



# import python libraries
import numpy as np
import pandas as pd


def iterate_model(xn, yn, a, b, sigma_1):
    '''
    Iterate the model one time step using the Euler-Maruyama scheme
    
    Parameters:
        a : control parameter
        b : constant = 0.3
    '''
    
    zeta_1 = np.random.normal()
    
    
    # Subsequent state variables
    xn_next = yn + 1.0 - a*xn**2 + sigma_1*zeta_1
    yn_next = b*xn
    
    return xn_next, yn_next



def simulate_model(x0, y0, avals, sigma_1):
    '''
    Simulate a single realisaiton of model
    
    Parameters:
        x0 : initial condition of memory variable
        y0 : intitial condition of apd
        avals: list of consecutive values of bifurcation parameter
        sigma_1 : noise amplitude

    Returns
    -------
    df_traj: pd.DataFrame
        Realisation fo model

    '''
    
    # Model parameters
    b = 0.3
    
    # Simulation parameters
    tburn = 100
        
    # Initialise arrays to store single time-series data
    t = np.arange(len(avals))
    x_vals = np.zeros(len(avals))
    y_vals = np.zeros(len(avals))
    
    # Run burn-in period
    for i in range(int(tburn)):
        x0, y0 = iterate_model(x0,y0,avals[0],b,sigma_1)
        
    # Initial condition post burn-in period
    x_vals[0]=x0
    y_vals[0]=y0
    
    # Run simulation
    for i, a in enumerate(avals[:-1]):
        x_vals[i+1], y_vals[i+1] = iterate_model(
            x_vals[i],y_vals[i],a,b,sigma_1)

    # Put data in df
    df_traj = pd.DataFrame(
        {'time': t,
         'x': x_vals,
         'y': y_vals})

    return df_traj
    

def sim_rate_forcing(sigma, rate_forcing=0.3/500):
    '''
    Run a simulation with the bifurcation parameter varying at some defined rate
    
    Parameters:
        sigma: noise ampltiude
        rate_forcing: (change in a)/(change in time)
    
    Note change in T from start to bifurcation is 100.
    '''
    
    x0 = 1.21
    y0 = 1.21

    astart= 0.1
    acrit = 0.4
    afinal = 1.5*acrit  # afinal: 1.5*0.4 = 0.6
    avals = np.arange(astart, afinal, rate_forcing)
    
    # Take transition time as time at bifurcation
    transition = int((acrit-astart)/rate_forcing)

    df_forced = simulate_model(x0, y0, avals, sigma)
    series_forced = df_forced.set_index('time')['x']
    
    # Simulate a null trajectory with the same length as the pretransition section
    df_null = simulate_model(x0, y0, [astart]*transition, sigma)
    series_null = df_null.set_index('time')['x']
    
    return series_forced, transition, series_null
    

