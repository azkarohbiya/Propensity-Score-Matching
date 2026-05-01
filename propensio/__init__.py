"""
propensio
=========
Causal inference using Propensity Score Matching and Euclidean LCG method.

Quick Start
-----------
Load a sample dataset and run PropensityScoreMatch in seconds:

    import propensio

    df = propensio.load_dataset('stroke')   # built-in sample data

    psm = propensio.PropensityScoreMatch(
        df        = df,
        features  = ['age', 'bmi', 'hypertension'],
        treatment = 'treatment_column',
        outcome   = 'outcome_column'
    )

    # Treatment effects
    print(psm.ATT)          # Average Treatment Effect on the Treated
    print(psm.ATE)          # Average Treatment Effect
    print(psm.ATC)          # Average Treatment Effect on the Control

    # Results
    psm.df_matched          # Matched dataframe
    psm.df_smd              # Standardized Mean Difference table
    psm.df_TE               # Treatment effect dataframe

    # Plots
    psm.plot_smd()          # SMD chart (before vs after matching)

Classes
-------
PropensityScoreMatch
    Main class. Matches treated and control groups using propensity scores
    and estimates treatment effects via linear regression.

EuclideanMethod
    LCG selection using Euclidean Distance and t-test p-value optimization.
    Requires a dataframe with columns [msisdn, rev].

MapEuclideanMethod
    Runs EuclideanMethod on multiple takers/non-takers pairs at once.

EuclideanMethodAscDesc
    Runs MapEuclideanMethod with both ascending and descending sort,
    then selects the best result per group.

Install
-------
    pip install propensio
"""

from .matching import PropensityScoreMatch
from .lcgeuclideanmethod import EuclideanMethod, MapEuclideanMethod, EuclideanMethodAscDesc, dfpack2arr
from .datasets import load_dataset
