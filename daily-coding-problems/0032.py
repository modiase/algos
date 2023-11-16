"""
Suppose you are given a table of currency exchange rates, represented as a 2D array.
Determine whether there is a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount 
A of any currency, so that you can end up with some amount greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.
"""
from typing import Mapping

def find_arbitrages(rates: Mapping[str, Mapping[str, float]]):
    """
    Finds possible arbitrages in a mapping of exhange rates.
    
    Arguments:
    ----------
    rates: An mapping of exchange rates where each key is the
    source exchange rate and the value is a mapping from the 
    target currency to the target excahnge rate.

    returns: a list of sequences of currencies that result 
    in an arbitrage.
    """
    arbitrages = []
    for source_currency, exchange_rates in rates.items():
        arbitrages += find_arbitrages_for_currency(source_currency, exchange_rates)
    return arbitrages
    
def find_arbitrages_for_currency(source_currency: str, exchange_rates: Mapping[str, float]):
    """
    Finds possible arbitrages for a given currency
    
    Arguments:
    ----------
    source_currency: the starting currency
    exchange_rates: A mapping of target currency to exchange rate

    returns: a list of sequences of currencies that result 
    in an arbitrage.
    """
    arbitrages = []
    for source in rates:
        arbitrages += find_arbitrages_for_currency(source)
    return arbitrages
