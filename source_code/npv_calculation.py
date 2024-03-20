# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:17:54 2023

@author: adh
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import norm

def npv_calculation(data, titles, subsidy, lump_sum, period):

    # Assume no subsidy before 2024
    if period < 16:
        # subsidy = 0.5
        subsidy = 0
        lump_sum = 0
    # elif period < 19:
    #     subsidy = 0.66
    #     lump_sum = 0

    # Set main parameters
    yearly_cons = data['consumption'][:, :, :, :] * 12
    cons_size = [1.5, 1, 0.75]
    battery_size = np.multiply(cons_size, 4)
    pl_idx = titles['battery_data'].index('elec_price_low')
    price_low = data['battery_specs'][pl_idx, 0, 0, 0]
    ph_idx = titles['battery_data'].index('elec_price_high')
    price_high =  data['battery_specs'][ph_idx, 0, 0, 0]
    plim_idx = titles['battery_data'].index('elec_price_limit')
    price_limit = data['battery_specs'][plim_idx, 0, 0, 0]
    pgr_idx = titles['battery_data'].index('elec_price_growth')
    price_gr = data['battery_specs'][pgr_idx, 0, 0, 0]
    eff_idx = titles['battery_data'].index('efficiency')
    battery_eff = data['battery_specs'][eff_idx, 0, 0, 0]
    dod_idx = titles['battery_data'].index('depth_of_discharge')
    dod = data['battery_specs'][dod_idx, 0, 0, 0]
    disc_idx = titles['battery_data'].index('discount_rate')
    discount_rate = data['battery_specs'][disc_idx, 0, 0, 0]
    bc_idx = titles['battery_data'].index('battery_cost')
    battery_cost = data['battery_specs'][bc_idx, 0, 0, 0]
    lc_idx = titles['battery_data'].index('labour_cost')
    labour_cost = data['battery_specs'][lc_idx, 0, 0, 0]
    lt_idx = titles['battery_data'].index('lifetime')
    lifetime = data['battery_specs'][lt_idx, 0, 0, 0]

    sc_idx = titles['battery_data'].index('self_consumption')
    self_consumption = data['battery_specs'][sc_idx, 0, 0, 0]
    fit_idx = titles['battery_data'].index('feed_in_tariff')
    feed_in_tariff = data['battery_specs'][fit_idx, 0, 0, 0]
    battery_price_chng = data['battery_price'][1, 0, 0, period]
    battery_cost = battery_cost * battery_price_chng

    # Adjust profiles with consumption
    yearly_cons_size = np.repeat(yearly_cons, len(titles['cons_size']), axis = 1)
    yearly_cons_size = yearly_cons_size * np.expand_dims(cons_size, axis = (0, 2, 3))
    profile_sum = data['profiles'][0, 0, :, :].sum(axis = 1).sum(axis = 0)

    for i, nuts3 in enumerate(titles['nuts3']):
        # Adjust profile with consumption profiles
        adj_profile = np.repeat(data['profiles'], len(titles['cons_size']), axis = 0)
        adj_profile = adj_profile * np.expand_dims(yearly_cons_size[i, :, :, :], axis = (1)) / profile_sum

        # Adjust solar profile to meet annual consumption
        pv_sum = data['pv_gen'][i, :, :, :].sum(axis = 1).sum(axis = 1)
        pv_size = yearly_cons_size[i, :, 0, 0] / pv_sum
        adj_pv_gen = data['pv_gen'][i, :, :, :] * np.expand_dims(pv_size, axis = (1, 2))
        # Extend PV array with NUTS3 dimension
        adj_pv_gen = np.repeat(np.expand_dims(adj_pv_gen, axis = (0)), len(titles['profile_type']), axis = 0)
        # Calculate PV overproduction
        overprod = adj_pv_gen - adj_profile
        overprod[overprod < 0] = 0
        # data['pv_overprod'] = overprod

        # Calculate actual price
        price_low_cons = yearly_cons_size[i, :, :, :].copy()
        price_low_cons[price_low_cons > price_limit] = price_limit
        price_high_cons = yearly_cons_size[i, :, :, :].copy() - price_low_cons
        price_high_cons[price_high_cons < 0] = 0

        price = (price_low_cons * price_low + price_high_cons * price_high) / yearly_cons_size[i, :, :, :]
        # Annuity factor for NPV
        annuity_factor = (1 - ((1 + price_gr) / (1 + discount_rate))**lifetime)
        # Price difference between electricity price and feed-in-tariff (benefit from 1 kWh energy stored)
        price_diff = price - feed_in_tariff
        # Battery output adjusted for efficiency and self-consumption
        daily_storage = overprod.sum(axis = 3)
        for s, size in enumerate(titles['cons_size']):
            daily_storage[:, s, :][daily_storage[:, s, :] > battery_size[s]] = battery_size[s]
        battery_output = daily_storage.sum(axis = 2) * battery_eff * dod # * self_consumption
        # Total benefits from battery
        npv_benefit = battery_output * price_diff[:, :, 0] / (discount_rate - price_gr) * annuity_factor
        data['battery_benefit'][i, :, :, period] = npv_benefit

        # Adjustment of labour cost with nuts3 income
        inc_ratio = data['income'][i, :, 0, 0] / data['income'][:, :, 0, 0].mean()
        inv = (battery_cost + labour_cost * inc_ratio) * battery_size * (1 - subsidy) - lump_sum

        # Calculae subsidy
        # Assume that realtive subsidy provides subsidy until NPV = 0
        # Therefore covers the subsidy % of the benefits
        subs = npv_benefit / (1 - subsidy) * subsidy + lump_sum
        # subs = (battery_cost + labour_cost * inc_ratio) * battery_size * subsidy + lump_sum
        # Extend investment array with profile_type dimension
        inv_2d = np.repeat(np.expand_dims(inv, axis = (0)), len(titles['profile_type']), axis = 0)
        data['battery_investment'][i, :, :, period] = inv_2d
        # subs_2d = np.repeat(np.expand_dims(subs, axis = (0)), len(titles['profile_type']), axis = 0)


        # NPV
        npv = np.subtract(npv_benefit, inv_2d)
        data['npv'][i, :, :, period] = npv
        data['subsidy'][i, :, :, period] = subs

    return data


def potential_population(data, titles, period):

    cons_size_w = np.expand_dims([0.25, 0.5, 0.25], axis = (0,1))
    p_ratios = data['p'][:, 0, 0, 0] / data['p'][:, :, 0, 0].mean()
    innovators = 0.025 * p_ratios

    b_rel_std_idx = titles['battery_data'].index('battery_cost_std')
    battery_cost_rel_std = data['battery_specs'][b_rel_std_idx, 0, 0, 0]

    # Gather relevant variables
    inv_3d = data['battery_investment'][:, :, :, period]
    cost_std = inv_3d * battery_cost_rel_std
    benefits = data['battery_benefit'][:, :, :, period]

    # Calculate the potential population
    # Use the cumulative distribution function of normal distribution
    # To find the probability of finding a battery with investment cost
    # where NPV is positive
    # + add the share of innovators to the potential population
    pot_population_share = norm.cdf(benefits, inv_3d, cost_std)
    for reg, nuts3 in enumerate(titles['nuts3']):
        reg_pop_share = pot_population_share[reg, :, :]
        reg_pop_share[reg_pop_share < innovators[reg]] = innovators[reg]
        reg_pop_share[reg_pop_share > 1] = 1
        data['potential_pop_share'][reg, :, :, period] = reg_pop_share
    # pot_population_share[pot_population_share < innovators] = innovators
    pot_population_share[pot_population_share > 1] = 1
    data['potential_pop_share'][:, :, :, period] = pot_population_share


    # Calculate number of households by profile_type
    # Extend house nurmber array with profile_type dimension
    nr_houses_3d = np.repeat(data['nr_houses'][:, :, :, 0], len(titles['profile_type']), axis = 1)
    nr_houses_3d = np.repeat(data['nr_houses'][:, :, :, 0], len(titles['cons_size']), axis = 2)
    nr_houses_profile = data['profile_shares'][:, :, :, 0] * nr_houses_3d
    nr_houses_profile_size = cons_size_w * nr_houses_profile

    data['nr_houses_profile'][:, :, :, period] = nr_houses_profile_size

    # Potential population where NPV is positive, so they might buy battery
    pot_population = pot_population_share * nr_houses_profile_size
    data['potential_pop'][:, :, :, period] = pot_population

    return data

