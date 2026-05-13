"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (
    year_frac_act_x,
    date_series,
    year_frac_30e_360,
    schedule_year_fraction,
)
from utilities.ex0_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
)

from scipy.stats import norm

from typing import Union, List, Tuple


class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"


def swaption_price_calculator(
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Value date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying forward starting swap expiry.
        sigma_black (float): Swaption implied volatility.
        freq (int): Number of times a year the fixed leg pays the coupon.
        discount_factors (pd.Series): Discount factors.
        swaption_type (SwapType): Swaption type, default to receiver.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price (and possibly delta).
    """

    ttm = year_frac_act_x(ref_date, expiry, 365)
    d1 = (np.log(S0 / strike) + 0.5 * sigma_black ** 2 * ttm) / (sigma_black * np.sqrt(ttm))  
    d2 = d1 - sigma_black * np.sqrt(ttm)

    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)
    bpv = basis_point_value(fixed_leg_payment_dates[1:], discount_factors, fixed_leg_payment_dates[0])  

    if swaption_type == SwapType.RECEIVER:
        price = bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        delta = bpv * (norm.cdf(d1) - 1)
    elif swaption_type == SwapType.PAYER:
        price = bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        delta = bpv * norm.cdf(d1)
    else:
        raise ValueError("Invalid swaption type.")

    if compute_delta:
        return price, delta
    else:
        return price


def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
) -> float:
    """
    Given the specifics of an interest rate swap (IRS), return its rate sensitivity calculated as
    the duration of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.

    Returns:
        (float): Swap duration.
    """

    # RMK: in a Swap the Notional is NOT EXCHANGED at maturity

    # As Required in the provided document, Date convention is 30/360 for fixed
    yf = np.array([year_frac_30e_360(ref_date, d) for d in fixed_leg_payment_dates])

    # index.[0] gives me the settlement dates of the bootstrapped curve
    df_swap_dates = np.array([
        get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0], 
            d, 
            discount_factors.index, 
            discount_factors.values
        ) for d in fixed_leg_payment_dates
    ])

    P = np.sum(df_swap_dates) * swap_rate + df_swap_dates[-1]  
    Numerator = np.sum(df_swap_dates * yf) * swap_rate + yf[-1] * df_swap_dates[-1] 
    Du = Numerator/P

    return Du



def basis_point_value(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    settlement_date: dt.datetime | None = None,
) -> float:
    """
    Given a swap fixed leg payment dates and the discount factors, return the basis point value.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        settlement_date (dt.datetime | None): Settlement date, default to None, i.e. to today.
            Needed in case of forward starting swaps.

    Returns:
        float: Basis point value.
    """

    # discount_factors.index[0] = today
    if settlement_date is None:
        settlement_date = discount_factors.index[0]

    start_dates = [settlement_date] + fixed_leg_schedule[:-1]
    end_dates = fixed_leg_schedule
    yf = np.array([year_frac_30e_360(s, e) for s, e in zip(start_dates, end_dates)])

    df = np.array([
    get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], 
        date, 
        discount_factors.index, 
        discount_factors.values
    ) for date in fixed_leg_schedule
])

    bpv = np.sum(yf * df)

    return float(bpv)


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
) -> float:
    """
    Given a fixed leg payment schedule and the discount factors, return the swap par rate. If a
    forward start date is provided, a forward swap rate is returned.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        fwd_start_date (dt.datetime | None): Forward start date, default to None.

    Returns:
        float: Swap par rate.
    """
    
    # Here we compute the discount factor B(t0,tn). The if and else condition is 
    # needed since if no fwd_start_date is provided, then the discount factor we 
    # need is actually B(t0,t0), which is equal to 1.

    discount_factor_t0 = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fwd_start_date,
        discount_factors.index,
        discount_factors.values,
    ) if fwd_start_date is not None else 1.0

    bpv =  basis_point_value(fixed_leg_schedule, discount_factors, fwd_start_date)


    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )
    float_leg = discount_factor_t0 - discount_factor_tN

    return float(float_leg / bpv)


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule and the discount factors, return the swap
    mark-to-market.

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default to 'payer'.

    Returns:
        float: Swap mark-to-market.
    """

    # Single curve framework, returns price and basis point value
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)

    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )

    float_leg = 1.0 - P_term
    fixed_leg = swap_rate * bpv

    if swap_type == SwapType.PAYER:
        multiplier = 1
    elif swap_type == SwapType.RECEIVER:
        multiplier = -1
    else:
        raise ValueError("Unknown swap type.")

    return multiplier * (float_leg - fixed_leg)
