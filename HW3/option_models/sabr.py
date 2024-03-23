# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import pyfeng as pf
import abc

class ModelABC(abc.ABC):
    beta = 1   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr = None, None

    ### Numerical Parameters
    dt = 0.1
    n_path = 10000

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0.0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.beta = beta
        self.intr = intr

    def base_model(self, sigma=None):
        if sigma is None:
            sigma = self.sigma

        if self.beta == 0:
            return pf.Norm(sigma, intr=self.intr)
        elif self.beta == 1:
            return pf.Bsm(sigma, intr=self.intr)
        else:
            raise ValueError(f'0<beta<1 not supported')

    def vol_smile(self, strike, spot, texp=1.0):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(strike, spot, texp, cp=1)
        iv = self.base_model().impvol(price, strike, spot, texp, cp=1)
        return iv

    @abc.abstractmethod
    def price(self, strike, spot, texp=1.0, cp=1):
        """
        Vanilla option price

        Args:
            strike:
            spot:
            texp:
            cp:

        Returns:

        """
        return NotImplementedError

    def sigma_path(self, texp):
        """
        Path of sigma_t over the time discretization

        Args:
            texp:

        Returns:
            sigma_t
        """
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]
        
        Z_1 = np.random.standard_normal((n_dt, self.n_path))
        Z_t = np.cumsum(Z_1 * np.sqrt(dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))  # 此处是通过Z_t的cumsum来进行每步的累积的
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)

        return Z_1, sigma_t

    def intvar_normalized(self, sigma_path):
        """
        Normalized integraged variance I_t = \int_0^T sigma_t dt / (sigma_0^2 T)

        Args:
            sigma_path: sigma path

        Returns:

        """

        weight = np.ones(sigma_path.shape[0])
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        intvar = np.sum(weight[:, None] * sigma_path, axis=0)
        return intvar

class ModelBsmMC(ModelABC):
    """
    MC for Bsm SABR (beta = 1)
    """

    beta = 1.0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        # (1) Generate the paths of sigma_t.
        Z_1, vol_path = self.sigma_path(texp)  # normal variable Z_1, the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        
        # (2) Simulate S_0, ...., S_T.
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        X_1 = np.random.standard_normal((n_dt, self.n_path)) # normal variable X_1, indepent with Z_1
        W_1 = self.rho*Z_1 + np.sqrt(1-self.rho**2)*X_1  # normal variable W_1, correlated with Z_1 of rho
        #这里有一个问题，sigma_t和S_t会差一期，导致会有浪费的sigma_t
        sigma_times_W_accumulation =  np.cumsum(W_1 * np.sqrt(dt) * self.sigma * vol_path[:-1], axis=0)
        sigma_squared_accumulation = np.cumsum(np.square(self.sigma * vol_path[:-1]), axis=0)
        S_t = spot * np.exp(sigma_times_W_accumulation - 1/2*sigma_squared_accumulation*dt)
        S_t = np.insert(S_t, 0, spot*np.ones(S_t.shape[1]), axis=0)
        
        
        # (3) Calculate option prices (vector) for all strikes
        S_T = S_t[-1,:]
        df = np.exp(-self.intr * texp)
        p = df * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelNormMC(ModelBsmMC):
    """
    MC for Normal SABR (beta = 0)
    """

    beta = 0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        # (1) Generate the paths of sigma_t.
        Z_1, vol_path = self.sigma_path(texp)  # normal variable Z_1, the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        # (2) Simulate S_0, ...., S_T.
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        X_1 = np.random.standard_normal((n_dt, self.n_path)) # normal variable X_1, indepent with Z_1
        W_1 = self.rho*Z_1 + np.sqrt(1-self.rho**2)*X_1  # normal variable W_1, correlated with Z_1 of rho
        #这里有一个问题，sigma_t和S_t会差一期，导致会有浪费的sigma_t
        #原路径初值为1，此处的vol_path需要乘sigma_0
        sigma_times_W_accumulation = np.cumsum(W_1 * np.sqrt(dt) * self.sigma *  vol_path[:-1], axis=0) 
        S_t = spot + sigma_times_W_accumulation
        S_t = np.insert(S_t, 0, spot*np.ones(S_t.shape[1]), axis=0)

        # (3) Calculate option prices (vector) for all strikes
        S_T = S_t[-1,:]
        df = np.exp(-self.intr * texp)
        p = df * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelBsmCondMC(ModelBsmMC):
    """
    Conditional MC for Bsm SABR (beta = 1)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the BS model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''
        # (1) Generate the paths of sigma_t and normalized integrated variance
        Z_1, vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = self.sigma * vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path)

        # (2) Calculate the equivalent spot and volatility of the BS model
        vol = self.sigma * np.sqrt((1-self.rho**2) * I_t)
        spot_equiv = spot * np.exp(self.rho/self.vov*(sigma_t-self.sigma) - 
                                  self.rho**2*self.sigma**2 * texp/2 * I_t)

        m = self.base_model(vol)  #BSM model
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p


class ModelNormCondMC(ModelNormMC):
    """
    Conditional MC for Bsm SABR (beta = 1)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the Bachelier model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''
        # (1) Generate the paths of sigma_t and normalized integrated variance
        Z_1, vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = self.sigma * vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path)

        # (2) Calculate the equivalent spot and volatility of the BS model
        vol = self.sigma * np.sqrt((1-self.rho**2) * I_t)
        spot_equiv = spot + self.rho/self.vov * (sigma_t-self.sigma)

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p