import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.optimize import curve_fit
from jbgp_fit import get_experimentError
from pandas import DataFrame
import pdb

def drill2obs(df):
    """get the average distance between drill location and sample locations
       for every trial (row) in the dataframe"""
    mu_trial =\
        df.apply(lambda row: np.mean(np.abs(row[field] - row.obsX)), axis=1)
    return np.mean(mu_trial)


def dBetween(df, f1, f2):
    """get the average distance between df fields 'f1' and 'f2' for
       for every trial (row) in the dataframe"""
    mu_trial =\
        df.apply(lambda row: np.mean(np.abs(row[f1] - row[f2])), axis=1)
    return np.mean(mu_trial)


def dBetween_analysis(df, f1, f2):
    """this is for the dfs after being loaded with jbload.(experiment name)
    Gets the average distance between fields f1 and f2 in df for each subject.
    Then groups subjects and finds average distance by location"""
    # get avg dist for every trial for every sub
    mutrial_d = df.groupby('workerid').apply(lambda df0: dBetween(df0, f1, f2))\
                                      .reset_index()
    # get each sub's experiment avg
    mu_d = mutrial_d.groupby('workerid').mean().reset_index()
    # flatten and label grouped df
    mu_d = mu_d.rename(columns={0: 'mu_d'})
    # get and merge subject conditions (lengthscales)
    exp_lenscale = df.groupby('workerid').apply(lambda df0: df0.lenscale.iat[0])
    exp_lenscale = pd.DataFrame(exp_lenscale).reset_index()\
                    .rename(columns={0: 'exp_lenscale'})
    mu_d = mu_d.merge(exp_lenscale, on='workerid')
    # get condition averages
    cond_mu_d = mu_d.groupby('exp_lenscale').mean()
    return {'worker_mu_d': mu_d,
            'cond_mu_d': cond_mu_d}


def drill2obs_analysis(df):
    """this is for the noChoice_exp0 analysis df after going through
    preprocessing found in analysis0.py.  Gets the average distance
    from sample locations to drill location for each subject.  Then
    groups subjects and finds average distance by location"""
    # get avg dist for every trial for every sub
    mutrial_d2obs = df.groupby('workerid').apply(drill2obs).reset_index()
    # get each sub's experiment avg
    mu_d2obs = mutrial_d2obs.groupby('workerid').mean().reset_index()
    # flatten and label grouped df
    mu_d2obs = mu_d2obs.rename(columns={0: 'mu_d2obs'})
    # get and merge subject conditions (lengthscales)
    exp_lenscale = df.groupby('workerid').apply(lambda df0: df0.lenscale.iat[0])
    exp_lenscale = pd.DataFrame(exp_lenscale).reset_index()\
                    .rename(columns={0: 'exp_lenscale'})
    mu_d2obs = mu_d2obs.merge(exp_lenscale, on='workerid')
    # get condition averages
    cond_mu_d2obs = mu_d2obs.groupby('exp_lenscale').mean()
    return {'worker_mu_drill2obs': mu_d2obs,
            'cond_mu_drill2obs': cond_mu_d2obs}


def fit_lenscale_analysis_opt(df):
    fitfcn = lambda df0: minimize_scalar(lambda ls:\
            single_exp_err(df0, ls), bounds=(0.0001, 0.5), method="bounded")
    dfw = df.groupby('workerid')
    fits = dfw.apply(fitfcn)
    fits = fits.to_dict()
    fits = pd.DataFrame(fits)
    fits = fits.T
    fits = fits.x
    exp_lenscales = dfw.apply(lambda df0: df0.lenscale.iat[0])
    optfits = pd.concat([fits, exp_lenscales], axis=1)\
                .rename(columns={0:'fit_lenscale',1:'actual_lenscale'})
    return optfits


def single_exp_err(df, lenscale):
    print 'workerid: ' + df.workerid.iat[0]
    X = np.linspace(0, 1, 1028)
    expfits = get_experimentError(df, lenscale, X)
    return expfits[:, 0].mean()


def fit_lenscale(df, lenscalepool):
    """searches over the lenscales in lenscale pool for the ls that best
    predicts drilling behavior given the samples in the trial for each
    trial (row) in the df"""
    print 'workerid: ' + df.workerid.iat[0]
    X = np.linspace(0, 1, 1028)
    trialFits = [get_experimentError(df, lenscale, X)
                   for lenscale in lenscalepool]
    trialFits = np.asarray(trialFits)
    xerr = trialFits[:,:,0]
    mu_xerr_byLenscale = xerr.mean(axis=1)
    iBestLen = mu_xerr_byLenscale.argmin()
    fit_lenscale = lenscalepool[iBestLen]
    return fit_lenscale


def fit_lenscale_analysis(df, lenscalepool, go=0):
    """fits lengthscales for each worker using jbstats.fit_lenscale.
    Adds subject conditions and cleans up merged dataframe"""
    if not go:
        continue_warning = ('WARNING!  THIS TAKES A LONG TIME!'
                'Input 1 to continue, 0 to cancel'
                '(set param go=1 to skip this message): ')
        go = input(continue_warning)
    assert go in [0, 1], "must enter 0 or 1"
    if go:
        dfw = df.groupby('workerid')
        # WARNING!: THIS PART TAKES ABOUT 15-45 MINUTES!!
        # fit lenscale to best ls in lenscalepool
        lenscaleFits = dfw.apply(lambda df0: fit_lenscale(df0, lenscalepool))
        # get experiment lengethscales
        exp_lenscales = dfw.apply(lambda df0: df0.lenscale.iat[0])
        # merge fit and actual lengthscales into single df
        return pd.concat([exp_lenscales, lenscaleFits], axis=1)\
                        .rename(columns={0:'exp_lenscale', 1:'fit_lenscale'})\
                        .reset_index()

def cumprob_bestYtodrill_analysis(df, nbins=20):
    """this only works for 'chooseSam_noNorm' experiment at the moment.  gets
    the prob of deciding to drill given the best observed y so far is at least
    p for p in 0-100 points.  Then fits sigmoids for each subject and each
    condition."""
    # get highest yval when decided to drill
    subfcn = cumprob_bestYtodrill
    hsac(df, [condfcn, subfcn], ['condition', 'workerid'], fcnLabels=fcnLabels)
    max_samY = df.groupby(['workerid', 'round']).apply(
            lambda df0: df0.samY.iat[0].max())




def cumprob_bestYtodrill(dfw, nbins=20):
    """this only works for 'chooseSam_noNorm' experiment at the moment.  gets
    the prob of deciding to drill given the best observed y so far is at least
    p for p in 0-100 points, then fits a sigmoid.  This is intended for a single
    subject"""
    max_samY = dfw.groupby('round').apply(lambda df0: df0.samY.iat[0].max())
    # convert yval to score (see chooseSam_noNorm_e for the javascript conversion)
    MAXALLOWED = 0.5
    YRANGE = 1.
    fromMin = lambda samY0: (MAXALLOWED - samY0) / YRANGE
    earned = lambda fromMin: (1.-fromMin) * 100.
    y2score = lambda samY0: earned(fromMin(samY0))
    yScore = [y2score(samY0) for samY0 in max_samY]
    pScoreVals = np.linspace(0, 100, nbins+1)

    # get p(decide2drill for yValScores of at least cutoffVal
    pdrill = lambda samYvec, cutoffVal:\
        np.sum(samYvec < cutoffVal) / len(samYvec)
    cumprob_drill = [pdrill(yScore, v) for v in pScoreVals]

    # fit data with a sigmoid
    def sigmoid(xdata, x0, k):
        y = np.exp(-k*(xdata-x0)) / (1 + np.exp(-k*(xdata-x0)))
        return y
    init_guess = [50., 1.]
    popt, pcov = curve_fit(sigmoid, pScoreVals, cumprob_drill, p0=init_guess)
    xmodel = np.linspace(0, 100, 101)
    ymodel = sigmoid(xmodel, *popt)
    # compare fit to ideal with no knowledge of smoothness or penalties for sampling
    pideal = [55., 1.]  # ideal is a switch function with switch at 55
    yideal = sigmoid(xmodel, *pideal)
    return {'pScoreVals': pScoreVals,
            'cumprob_drill': cumprob_drill,
            'popt': popt,
            'xmodel': xmodel,
            'ymodel': ymodel,
            'pideal': pideal,
            'yideal': yideal,
            'gpsd': dfw.gpsd.iat[0],
            'condition': dfw.condition.iat[0],
            'workerid': dfw.workerid.iat[0]}

