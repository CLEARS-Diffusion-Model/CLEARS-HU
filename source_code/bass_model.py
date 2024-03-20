# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:11:06 2023

@author: adh
"""


import pandas as pd
import numpy as np
import os
import itertools

# os.chdir("C:/Users/adh/OneDrive - Cambridge Econometrics/ADH CE/Phd/ÚNKP_2023/data")


def Bass_param_estimation(data, titles):

    hist_years = [int(year) for year in titles['hist_year']]
    # Create datafrane for PVs
    pv = pd.DataFrame(data['pv_nr'][:, :, 0, 0], index = titles['nuts3'], columns = hist_years)
    # Annual PV installations
    pv_diff = pv.diff()
    # Total potential population for PVs
    nr_houses = pd.Series(data['nr_houses'][:, 0, 0, 0], index = titles['nuts3'])

    # Create grid for parameter tuning
    # Getting all permutations of list_1
    # with length of list_2
    q_list = [i / 1000 for i in list(range(1, 400, 5))]
    p_list = [i / 1000000 for i in range(1, 200, 5)]
    c = list(itertools.product(q_list, p_list))

    p_list = []
    q_list = []

    for i, reg_pv in pv.iterrows():
        # Change t0 depending on the number of PVs in first year
        first_year = min(hist_years)
        if (reg_pv[first_year] >= pv.iloc[:, 1].mean()) & (reg_pv[first_year] < pv.iloc[:, 2].mean()):
            t0 = 1
        elif (reg_pv[first_year] >= pv.iloc[:, 2].mean()) & (reg_pv[first_year] < pv.iloc[:, 3].mean()):
            t0 = 2
        elif reg_pv[first_year] >= pv.iloc[:, 3].mean():
            t0 = 3
        else:
            t0 = 0

        m = nr_houses[i]

        df = pd.DataFrame(0, columns = range(2008, 2023), index = c)


        for ix, ind in enumerate(df.index):
            # p is the coefficient of innovation
            # q is the coefficient of imitation
            q = ind[0]
            p = ind[1]

            pred = []
            for t in hist_years:
                t = int(t) - first_year + 1 + t0
                # Estimate diffusion with p, q combination
                pred_t = (m * (1 - np.exp(-(p+q)*(t-t0)))/(1+q/p*np.exp(-(p+q)*(t-t0))))
                pred = pred + [pred_t]

            df.iloc[ix, :] = pred

        # Find paa
        min_ind = np.argmin((df.subtract(reg_pv, axis = 1) ** 2).sum(axis = 1))
        min_values = df.index[min_ind]
        q = min_values[0]
        p = min_values[1]
        q_list = q_list + [q]
        p_list = p_list + [p]

        print('    Innovation and imitation parameters for', i)
        print('      p:', p)
        print('      q:', q)

        # pred = []
        # for t in range(2008, 2023):
        #     t = t - 2007
        #     pred_t = (m * (1 - np.exp(-(p+q)*(t-t0)))/(1+q/p*np.exp(-(p+q)*(t-t0))))
        #     pred = pred + [pred_t]

        # test = pd.Series(pred, index = range(2008, 2023))
        # test.plot()


        # pred = []
        # for t in range(2008, 2051):
        #     t = t - 2007 + t0
        #     pred_t = (m*(1 - np.exp(-(p+q)*(t-t0)))/(1+q/p*np.exp(-(p+q)*(t-t0))))
        #     pred = pred + [pred_t]

        # test = pd.Series(pred, index = range(2008, 2051))
        # test.plot()
        # reg_pv.plot()

    data['p'][:, 0, 0, 0] = p_list
    data['q'][:, 0, 0, 0] = q_list

    return data


def simulate_diffusion(data, titles, simulation_start, year, period):

    battery_cum_lag = data['battery_cum'][:, :, :, period - 1]
    t0 = simulation_start
    t_current = year - simulation_start

    for i, nuts3 in enumerate(titles['nuts3']):
        p = data['p'][i, 0, 0, 0]
        q = data['q'][i, 0, 0, 0]

        m = data['potential_pop'][i, :, :, period]
        nuts3_bat_lag = battery_cum_lag[i, :, :].sum()
        # Find diffusion year
        t_sim = np.array(range(0, 500)) / 10
        m_total = m.sum()


        pred_sim = (m_total * (1 - np.exp(-(p+q)*(t_sim)))/(1+q/p*np.exp(-(p+q)*(t_sim))))

        # Find closest value
        t_prev = np.argmin(abs(pred_sim - nuts3_bat_lag))
        t = t_prev / 10 + 1

        # if nuts3 == 'Pest':
            # print('    Potential population:', nuts3, ':', m_total)
            # print('    Diffusion year', nuts3, ':', t)

        # Estimate diffusion
        pred_t = (m * (1 - np.exp(-(p+q)*(t)))/(1+q/p*np.exp(-(p+q)*(t))))
        # if t > t0:
        #     pred_lag = (m * (1 - np.exp(-(p+q)*(t_lag-t0)))/(1+q/p*np.exp(-(p+q)*(t_lag-t0))))
        # else:
        #     pred_lag = np.zeros(len(titles['profile_type']))
        # Calculate new batteries
        data['battery_new'][i, :, :, period] = pred_t - battery_cum_lag[i, :, :]

    # Remove batteries over their lifetime
    lt_idx = titles['battery_data'].index('lifetime')
    lifetime = data['battery_specs'][lt_idx, 0, 0, 0]
    scrap_year = int(period - lifetime)
    battery_new = data['battery_new'][:, :, :, period]
    if scrap_year > 0:
        battery_scrap = data['battery_new'][:, :, :, scrap_year]
        data['battery_scrap'][:, :, :, period] = battery_scrap
        data['battery_cum'][:, :, :, period] = battery_cum_lag + battery_new #- battery_scrap
    else:
        data['battery_cum'][:, :, :, period] = battery_cum_lag + battery_new
    # Calculate share of battery owners
    data['battery_share'][:, :, :, period] = data['battery_cum'][:, :, :, period] / data['nr_houses'][:, :, :, 0]
    return data
