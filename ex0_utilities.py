"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 0: Discount Factors Bootstrap
"""

import numpy as np
import pandas as pd
import datetime as dt
from .date_functions import (
    business_date_offset,
    year_frac_act_x,
    year_frac_30e_360
)
from typing import Iterable, Union, List, Tuple

def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    
    effDates, effDf = dates, discount_factors
    
    if isinstance(effDates, pd.DatetimeIndex) or hasattr(effDates[0], 'date'):   
        effDates = [year_frac_act_x(effDates[0], d, 365) for d in effDates]  
        
    effDates = effDates[1:]
    effDf = list(discount_factors)[1:]

    zero_rates = []
    for i in range(len(effDates)):
        if effDates[i] > 0:
            zero_rates.append(-np.log(np.abs(effDf[i])) / effDates[i])
        else:
            zero_rates.append(0.0) 

    return zero_rates


def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")
    
    df = np.array(discount_factors, dtype=float)
    year_fractions = np.array([year_frac_act_x(reference_date, d, 360) for d in dates])
    
    zero_rates = np.zeros_like(df)
    
    valid = year_fractions > 0
    zero_rates[valid] = -np.log(np.abs(df[valid])) / year_fractions[valid]
    
    if not valid[0] and len(zero_rates) > 1:
        zero_rates[0] = zero_rates[1]
    
    target_yf = year_frac_act_x(reference_date, interp_date, 360)
    interp_zero_rate = np.interp(target_yf, year_fractions, zero_rates)
    
    discount = np.exp(-interp_zero_rate * target_yf)
    return float(discount)


def bootstrap(
    reference_date: dt.datetime,
    depo: pd.DataFrame,
    futures: pd.DataFrame,
    swaps: pd.DataFrame,
    shock: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:

    termDates, discounts = [reference_date], [1.0]
    
    depo.index = pd.to_datetime(depo.index)
    futures.index = pd.to_datetime(futures.index)
    swaps.index = pd.to_datetime(swaps.index)

    #### DEPOS
    depoDates = depo.index[0:3].to_list()    
    
    depoRates = depo.iloc[:, -2:].loc[depoDates].mean(axis=1).values 
    depoRates = depoRates + (shock if isinstance(shock, float) else shock[depoDates].values)

    termDates += depoDates
    depoFrac = [year_frac_act_x(reference_date, depoDates[i], 360) for i in range(len(depoDates))]
    
    for i in range(0, len(depoDates)):
        discounts.append(1/(1 + depoRates[i] * depoFrac[i])) 
    
    #### FUTURES
    futures.columns = futures.columns.str.strip().str.capitalize()
    futures_of_interest = futures.iloc[0:7, :]

    for index, rowFut in futures_of_interest.iterrows():
        t_i = rowFut['Settle']         
        t_i_plus_1 = rowFut['Expiry']  

        current_price = (rowFut['Bid'] + rowFut['Ask']) / 2.0
        fwd_rate = (100.0 - current_price) / 100.0
        
        yf_fut = year_frac_act_x(t_i, t_i_plus_1, 360)  
        fwd_disc = 1.0 / (1.0 + fwd_rate * yf_fut)

        discount_t_i = get_discount_factor_by_zero_rates_linear_interp(
            reference_date=reference_date,
            interp_date=t_i,
            dates=termDates,       
            discount_factors=discounts  
        )

        discount_t_i_plus_1 = discount_t_i * fwd_disc
        
        termDates.append(t_i_plus_1)
        discounts.append(discount_t_i_plus_1)

   
    #### SWAPS
    swapDate = swaps.index[0] 
    swapYearFrac = [year_frac_30e_360(reference_date, swapDate)]
    
    df_1 = get_discount_factor_by_zero_rates_linear_interp(reference_date, swapDate, termDates, discounts)
    swap_old = swapDate
    BPV = df_1 * swapYearFrac[0]

    swapRates = (swaps.iloc[:, -2:].mean(axis=1).values / 100.0)
    swapRates = swapRates + (shock if isinstance(shock, float) else shock[swaps.index].values)
    
    for idx in range(1, len(swaps.index)):
        swapDate = swaps.index[idx]
        rate = swapRates[idx]
        yf = year_frac_30e_360(swap_old, swapDate)
        
        swapYearFrac.append(yf)
        df = (1 - rate * BPV) / (1 + rate * yf)
        
        termDates.append(swapDate)
        discounts.append(df)

        BPV += df * yf
        swap_old = swapDate

    #### FINAL CURVES
    discount_factors = pd.Series(index=termDates, data=discounts)
    
    zero_partial = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    
    zero_rates = pd.Series(index=termDates[1:], data=zero_partial)
    
    return discount_factors, zero_rates