#-*- coding: utf-8 -*-
"""
File: bs_2dim_helper.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import sys
sys.path.append("..")

import numpy as np
from scipy.stats import norm
from typing import Callable

def index_function(N : int, d : int, coord : tuple) -> int:
    '''
    Converts from the coordinate system along each each dimension to the overall
    index of the linear system to be solved. 

    Inputs:
    -------

    N  : (int) number of grid points along each dimension.

    d  : (int) number of dimensions in the problem.

    coord  : (tuple) array of coordinates along each dimension.

    The ordering of the coord tuple is as follows:

    coord = (j_{d},j_{d-1},...,j_{1})

    where j_{i} is the coordinate along the i-th dimension.

    Outputs:
    --------

    (int) :  index of the linear system to be solved.

    '''

    index = 0
    for m in range(1,d+1):
        index += ((N-1)**(m-1))*(coord[m-1]-1)
    return index + 1

def convert_to_x1(j1 : int, x1_lower : int, x1_upper : int, x1_n_grid : int) -> float:
    '''
    Converts the index j1 to the corresponding x1 value

    Inputs:
    -------

    j1 : (int) The index of the x1 value

    x1_lower : (int) The lower bound of the x1 value

    x1_upper : (int) The upper bound of the x1 value

    x1_n_grid : (int) The number of grid points in the x1 direction

    Outputs:
    --------

    (float) : The x1 value corresponding to the index j1

    '''

    return ((x1_upper-x1_lower)/(x1_n_grid - 2))*(j1-1) + x1_lower

def convert_to_x2(j2 : int, x2_lower : int, x2_upper : int, x2_n_grid : int) -> float:
    '''
    Converts the index j2 to the corresponding x2 value

    Inputs:
    -------

    j2 : (int) The index of the x2 value

    x2_lower : (int) The lower bound of the x2 value

    x2_upper : (int) The upper bound of the x2 value

    x2_n_grid : (int) The number of grid points in the x2 direction

    Outputs:
    --------

    (float) : The x2 value corresponding to the index j2

    '''

    return ((x2_upper-x2_lower)/(x2_n_grid - 2))*(j2-1) + x2_lower

## \vec{B}_{(\partial^{2} x_{1})} ##

def B_x1_x1(tau : float, x1_n_grid : int, x2_n_grid : int,
             spatial_step : float, Lx_1 : Callable, Ux_1 : Callable) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{(\partial^{2} x_{1})}, the boundary conditions
    for the second derivative along the x1 axis. The latex representation is:

    $$
    \left(\\vec{B}_{(\partial^{2} x_{1})}(t)\\right)_{l} = 
    \\frac{1}{h^{2}}\sum_{j_{2}}\left[\delta_{l,\mathcal{N}(j_{2},1)}L_{x_{1}}(t,j_{2})
    +\delta_{l,\mathcal{N}(j_{2},N-1)}U_{x_{1}}(t,j_{2}) \\right]
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    x1_n_grid : (int) The number of grid points in the x1 direction

    x2_n_grid : (int) The number of grid points in the x2 direction

    spatial_step : (float) The spatial step size

    Lx_1 : (Callable) The function that returns the lower boundary condition

    Ux_1 : (Callable) The function that returns the upper boundary condition

    We assume that x1_n_grid = x2_n_grid. If this is modified, unexpected 
    behaviour may occur.
    
    Outputs:
    --------

    (np.ndarray) : The vector \vec{B}_{(\partial^{2} x_{1})}
    
    '''

    vec = np.zeros((x1_n_grid -1)*(x2_n_grid-1))
    for l in range(1,(x1_n_grid - 1)*(x2_n_grid) + 1):
        for j2 in range(1,x2_n_grid):
            if index_function(x2_n_grid,2,(j2,1)) == l:
                vec[l-1] = (1/(spatial_step**2))*Lx_1(tau,j2)
            elif index_function(x2_n_grid,2,(j2,x2_n_grid-1)) == l:
                vec[l-1] = (1/(spatial_step**2))*Ux_1(tau,j2)
            else:
                break 
    return vec

## \vec{B}_{(\partial^{2} x_{2})} ##

def B_x2_x2(tau : float, x1_n_grid : int, x2_n_grid : int,
             spatial_step : float, Lx_2 : Callable, Ux_2 : Callable) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{(\partial^{2} x_{2})}, the boundary conditions
    for the second derivative along the x2 axis. The latex representation is:
    
    $$
    \left(\\vec{B}_{(\partial^{2} x_{2})}(t)\\right)_{l} = 
    \\frac{1}{h^{2}}\sum_{j_{2}}\left[\delta_{l,\mathcal{N}(1,j_{1})}L_{x_{2}}(t,j_{2})
    +\delta_{l,\mathcal{N}(N-1,j_{1})}U_{x_{2}}(t,j_{1}) \\right]
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    x1_n_grid : (int) The number of grid points in the x1 direction

    x2_n_grid : (int) The number of grid points in the x2 direction

    spatial_step : (float) The spatial step size

    Lx_2 : (Callable) The function that returns the lower boundary condition

    Ux_2 : (Callable) The function that returns the upper boundary condition

    We assume that x1_n_grid = x2_n_grid. If this is modified, unexpected
    behaviour may occur.

    Outputs:
    --------

    (np.ndarray) : The vector \vec{B}_{(\partial^{2} x_{2})}
    '''

    vec = np.zeros((x1_n_grid -1)*(x2_n_grid-1))
    for l in range(1,(x1_n_grid - 1)*(x2_n_grid) + 1):
        for j1 in range(1,x1_n_grid):
            if index_function(x1_n_grid,2,(j1,1)) == l:
                vec[l-1] = (1/(spatial_step**2))*Lx_2(tau,j1)

            elif index_function(x1_n_grid,2,(j1,x2_n_grid-1)) == l:

                vec[l-1] = (1/(spatial_step**2))*Ux_2(tau,j1)
            else:
                break 
    return vec    
    
## \vec{B}_{(\partial x_{1}\partial x_{2})} ##

def B_x1_x2(tau : float, x1_n_grid : int, x2_n_grid : int,
             spatial_step : float, Lx_1 : Callable, Lx_2 : Callable,
             Ux_1 : Callable, Ux_2 : Callable) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{(\partial x_{1}\partial x_{2})}, the boundary conditions
    for the mixed derivative along the x1 and x2 axis. The latex representation is:
    
    $$
    \left(\\vec{B}_{(\partial x_{1}\partial x_{2})}(t)\\right)_{l} =
    \\frac{1}{4h^{2}}\sum_{j_{1}}\sum_{j_{2}}\left[-\delta_{l,\mathcal{N}(j_{2},1)}L_{x_{1}}(t,j_{2})
    -\delta_{l,\mathcal{N}(j_{1},1)}L_{x_{2}}(t,j_{1})+\delta_{l,\mathcal{N}(j_{2},N-1)}U_{x_{1}}(t,j_{2})
    +\delta_{l,\mathcal{N}(j_{1},N-1)}U_{x_{2}}(t,j_{1}) \\right]
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    x1_n_grid : (int) The number of grid points in the x1 direction

    x2_n_grid : (int) The number of grid points in the x2 direction

    spatial_step : (float) The spatial step size

    Lx_1 : (Callable) The function that returns the lower boundary condition

    Lx_2 : (Callable) The function that returns the lower boundary condition

    Ux_1 : (Callable) The function that returns the upper boundary condition

    Ux_2 : (Callable) The function that returns the upper boundary condition

    We assume that x1_n_grid = x2_n_grid. If this is modified, unexpected
    behaviour may occur.

    Outputs:
    --------

    (np.ndarray) : The vector \\vec{B}_{(\partial x_{1}\partial x_{2})}

    '''

    vec = np.zeros((x1_n_grid -1)*(x2_n_grid-1))
    for l in range(1,(x1_n_grid - 1)*(x2_n_grid) + 1):
        for j1 in range(1,x1_n_grid):
            for j2 in range(1,x2_n_grid):
                if index_function(x2_n_grid,2,(j2,1)) == l:
                    vec[l-1] = -(1/(4*spatial_step**2))*Lx_1(tau,j2)
                
                elif index_function(x1_n_grid,2,(1,j1)) == l:	
                    vec[l-1] = -(1/(4*spatial_step**2))*Lx_2(tau,j1)
                
                elif index_function(x2_n_grid,2,(j2,x2_n_grid-1)) == l:
                    vec[l-1] = (1/(4*spatial_step**2))*Ux_1(tau,j2)

                elif index_function(x1_n_grid,2,(x1_n_grid-1,j1)) == l:
                    vec[l-1] = (1/(4*spatial_step**2))*Ux_2(tau,j1)  

                else:
                    break
    return vec

## \vec{B}_{(\partial x_{1})} ##

def B_x1(tau : float, x1_n_grid : int, x2_n_grid : int,
          spatial_step : float, Lx_1 : Callable, Ux_1 : Callable) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{(\partial x_{1})}, the boundary conditions
    for the first derivative along the x1 axis. The latex representation is:
    
    $$
    \left(\\vec{B}_{(\partial x_{1})}(t)\\right)_{l} =
    \\frac{1}{2h}\sum_{j_{2}}\left[-\delta_{l,\mathcal{N}(j_{2},1)}L_{x_{1}}(t,j_{2})
    +\delta_{l,\mathcal{N}(j_{2},N-1)}U_{x_{1}}(t,j_{2}) \\right]
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    x1_n_grid : (int) The number of grid points in the x1 direction

    x2_n_grid : (int) The number of grid points in the x2 direction

    spatial_step : (float) The spatial step size

    Lx_1 : (Callable) The function that returns the lower boundary condition

    Ux_1 : (Callable) The function that returns the upper boundary condition

    We assume that x1_n_grid = x2_n_grid. If this is modified, unexpected
    behaviour may occur.

    Outputs:
    --------

    (np.ndarray) : The vector \\vec{B}_{(\partial x_{1})}

    '''

    vec = np.zeros((x1_n_grid -1)*(x2_n_grid-1))
    for l in range(1,(x1_n_grid - 1)*(x2_n_grid) + 1):
        for j2 in range(1,x2_n_grid):
            if index_function(x2_n_grid,2,(j2,1)) == l:
                vec[l-1] = -(1/(2*spatial_step))*Lx_1(tau,j2)

            elif index_function(x2_n_grid,2,(j2,x2_n_grid-1)) == l:
                vec[l-1] = (1/(2*spatial_step))*Ux_1(tau,j2)

            else:
                break 
    return vec

## \vec{B}_{(\partial x_{2})} ##

def B_x2(tau : float, x1_n_grid : int, x2_n_grid : int,
          spatial_step : float, Lx_2 : Callable, Ux_2 : Callable) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{(\partial x_{2})}, the boundary conditions
    for the first derivative along the x2 axis. The latex representation is:
    
    $$
    \left(\\vec{B}_{(\partial x_{2})}(t)\\right)_{l} =
    \\frac{1}{2h}\sum_{j_{1}}\left[-\delta_{l,\mathcal{N}(1,j_{1})}L_{x_{2}}(t,j_{1})
    +\delta_{l,\mathcal{N}(N-1,j_{1})}U_{x_{2}}(t,j_{1}) \\right]
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    x1_n_grid : (int) The number of grid points in the x1 direction

    x2_n_grid : (int) The number of grid points in the x2 direction

    spatial_step : (float) The spatial step size

    Lx_2 : (Callable) The function that returns the lower boundary condition

    Ux_2 : (Callable) The function that returns the upper boundary condition

    We assume that x1_n_grid = x2_n_grid. If this is modified, unexpected
    behaviour may occur.

    Outputs:
    --------

    (np.ndarray) : The vector \vec{B}_{(\partial x_{2})}

    '''

    vec = np.zeros((x1_n_grid -1)*(x2_n_grid-1))
    for l in range(1,(x1_n_grid - 1)*(x2_n_grid) + 1):
        for j1 in range(1,x1_n_grid):
            if index_function(x1_n_grid,2,(j1,1)) == l:
                vec[l-1] = -(1/(2*spatial_step))*Lx_2(tau,j1)

            elif index_function(x1_n_grid,2,(j1,x2_n_grid-1)) == l:
                vec[l-1] = (1/(2*spatial_step))*Ux_2(tau,j1)
                
            else:
                break 
    return vec    

def B_total(tau : float, vol_1 : float, vol_2 : float, rho : float,
             r : float, B_x1_x1 : Callable, B_x2_x2 : Callable,
               B_x1_x2 : Callable, B_x1,B_x2 : float) -> np.ndarray:
    '''
    Returns the vector \\vec{B}_{\\text{total}}, the total boundary conditions.
    The latex representation is:

    $$
    \\vec{B}_{\\text{total}}(t) = \\frac{\sigma_{1}^{2}}{2}\\vec{B}_{\partial^{2}x_{1}}(t)
    +\\frac{\sigma_{2}^{2}}{2}\\vec{B}_{\partial^{2}x_{2}}(t) 
    +\\rho \sigma_{1}\sigma_{2}\\vec{B}_{(\partial x_{1}\partial x_{2})}(t) 
    +\left(r-\\frac{\sigma_{1}^{2}}{2}\\right)\\vec{B}_{(\partial x_{1})}(t)
    +\left(r-\\frac{\sigma_{2}^{2}}{2}\\right)\\vec{B}_{(\partial x_{2})}(t)
    $$

    Inputs:
    -------

    tau : (float) The time to maturity of the option

    vol_1 : (float) The volatility of the first asset

    vol_2 : (float) The volatility of the second asset

    rho : (float) The correlation between the two assets

    r : (float) The risk free rate

    B_x1_x1 : (Callable) The function that returns the boundary conditions for the
                        second derivative along the x1 axis

    B_x2_x2 : (Callable) The function that returns the boundary conditions for the
                        second derivative along the x2 axis   

    B_x1_x2 : (Callable) The function that returns the boundary conditions for the
                        mixed derivative along the x1 and x2 axis

    B_x1 : (Callable) The function that returns the boundary conditions for the
                        first derivative along the x1 axis

    B_x2 : (Callable) The function that returns the boundary conditions for the
                        first derivative along the x2 axis

    Outputs:
    --------

   (np.ndarray) : The vector \\vec{B}_{\\text{total}}

    '''

    vec = ((vol_1**2)/2)*B_x1_x1(tau)
    vec += ((vol_2**2)/2)*B_x2_x2(tau)
    vec += rho*vol_1*vol_2*B_x1_x2(tau)
    vec += (r-((vol_1**2)/2))*B_x1(tau)
    vec += (r-((vol_2**2)/2))*B_x2(tau)
    return vec

def maghrabe_pricing(S_1 : float, S_2 : float, vol_1 : float, vol_2 : float, rho : float, T : float) -> float:
    """
    Calculate the price of a spread option using the Maghrabe formula for European exchange options.

    Inputs:
    -------

    S_1 : (float) The current price of the underlying asset 1

    S_2 : (float) The current price of the underlying asset 2

    vol_1 : (float) The volatility of the underlying asset 1

    vol_2 : (float) The volatility of the underlying asset 2

    rho : (float) The correlation between the two underlying assets

    T : (float) The time to maturity of the option

    Outputs:
    --------

    (float) : The price of the European exhchange option

    """

    # Calculate the parameters of the spread option
    mu = np.log(S_1 / S_2)
    sigma = np.sqrt(vol_1**2 + vol_2**2 - 2 * rho * vol_1 * vol_2)
    d_1 = (mu + (0.5 * sigma**2 * T)) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    # Calculate the price of the spread option
    return S_1 * norm.cdf(d_1) - S_2 * norm.cdf(d_2) 

if __name__ == "__main__":
    print(maghrabe_pricing(170,90,0.3,0.4,0.1,2))