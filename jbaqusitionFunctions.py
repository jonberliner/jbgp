from jb.jbdrill import jbload
from pandas import DataFrame, merge, concat
from numpy import log, pi, mean, linspace, zeros_like, diag, append,
from numpy import array as a
from pylab import find
from numpy.random import RandomState
from scipy.stats import norm
rng = RandomState()

#TODO:
#   decide if aqfcns should return value for every point in the domain, or only the chosen point
#   need to make above point consistent bt functions
#

#FIXME: not yet tested
def emep(domainbounds, xObs, yObs, lenscale, sigvar, noisevar2, xres, yres, ysdbounds):
    """expected value of the max expected value of the posterior"""
    out = {}
    xcandidates = linspace(domainbounds[0], domainbounds[1], xres)
    out['x'] = xcandidates
    out['emep'] = zeros_like(xcandidates)
    ySDcandidates = linspace(ysdbounds[0], ysdbounds[1], yres)
    pdfySDcandidates = norm.pdf(ySDcandidates)
    cdfySDcandidates = norm.cdf(ySDcandidates)
    for ix0, xx0 in enumerate(xcandidates):
        # print 'x: ' + str(ix0)
        x0 = a([xx0])
        # get ymu and ysd for x0 so know what points to consider for generating prob-weighted ev of max of posterior
        ymu0 = jbgp.conditioned_mu(x0, xObs, yObs, lenscale, sigvar, noisevar2)
        xcmpri0 = jbgp.K_se(x0, x0, lenscale, sigvar)  # get covmat for xSam
        ycm0 = jbgp.conditioned_covmat(x0, atleast_2d(xcmpri0), xObs, lenscale, sigvar, noisevar2)
        ysd0 = diag(ycm0)
        # y-vals to consider with probs pysdcandidates
        ycands = a([ymu0 + (ysd0 * d) for d in ySDcandidates])
        xObsPlusX0 = append(xObs, x0)  # add considered point to xObs locations
        # run simulations of what happens with certain y-vals
        mep = zeros_like(cdfySDcandidates)

        for iy0, y0 in enumerate(ycands):
            # print 'y: ' + str(iy0)
            yObsPlusY0 = append(yObs, y0)
            py0 = pdfySDcandidates[iy0]
            mu0 = jbgp.conditioned_mu(domain, xObsPlusX0, yObsPlusY0, lenscale, sigvar, noisevar2)
            mep[iy0] = mu0.max()

        out['emep'][ix0] = trapz(maxevpost, cdfySDcandidates)
    return out


def exploit(mu, domain):
    iymax = mu.argmax()
    ymax = mu.max()
    xymax = domain[iymax]
    return {'imax': iymax,
            'fmax': ymax,
            'xmax': xymax}


def infomax(covmat, domain):
    sd = diag(covmat)
    cvXv = dot(covmat, sd)
    infoGain = sum(cvXv, axis=0) # get col sums
    iInfomax = infoGain.argmax()
    xInfomax = domain[iInfoGain]
    finfomax = infoGain[iInfomax]

    return {'imax': iInfomax,
            'fmax': fInfomax,
            'xmax': xInfomax}


def PI(yBest, mu, sd, domain, ksi):
    PI = norm.cdf( (mu - yBest - ksi) / sd )
    ipimax = PI.argmax()
    pimax = PI.max()
    xpimax = domain[ipimax]
    return {'imax': ipimax,
            'fmax': pimax,
            'xmax': xpimax}


def EI(yBest, mu, sd, domain):
    Z = (mu - yBest) / sd
    EI = (mu - yBest) * norm.cdf(Z) + sd * norm.pdf(Z)
    ieimax = EI.argmax()
    eimax = EI.max()
    xeimax = domain[ieimax]
    return {'imax': ieimax,
            'fmax': eimax,
            'xmax': xeimax}


def GPUCB(mu, sd, domain, t, v=1., delta=0.1):
    d = len(domain.shape)
    # from brochu et al 2010, delta from srivinas et al 2010
    tau = 2. * log( (t**(d/2.+2.) * pi**2.) / (3.*delta) )
    GPUCB = mu + sqrt(v * tau) * sd
    igpucbmax = GPUCB.argmax()
    gpucbmax = GPUCB.max()
    xgpucbmax = domain[igpucbmax]
    return {'imax': igpucbmax,
            'fmax': gpucbmax,
            'xmax': xgpucbmax}


def optimal_myopic(domain, domain_mu_prior, domain_cm_prior, xObs, yObs,\
                   xres, yres, ysdbounds, lenscale, sigvar, noisevar2):
    out = {}
    bounds = [domain[0], domain[-1]]
    xcandidates = linspace(bounds[0], bounds[1], xres)
    out['x'] = xcandidates
    out['ev'] = zeros_like(xcandidates)
    ysdcandidates = linspace(ysdbounds[0], ysdbounds[1], yres)
    pysdcandidates = norm.pdf(ysdcandidates)

    for ix0, x0 in enumerate(xcandidates):
        x0 = a([x0])
        xObsPlusX0 = append(xObs, x0)
        xcmpri0 = jbgp.K_se(x0, x0, lenscale, sigvar)
        ymu0 = jbgp.conditioned_mu(x0, xObs, yObs, lenscale, sigvar, noisevar2)
        ycm0 = jbgp.conditioned_covmat(x0, atleast_2d(xcmpri0), xObs, lenscale, sigvar, noisevar2)
        ysd0 = diag(ycm0)
        ycands = a([ymu0 + (ysd0 * n) for n in ysdcandidates])
        for iy0, y0 in enumerate(ycands):
            yObsPlusY0 = append(yObs, y0)
            py0 = pysdcandidates[iy0]
            mu0 = jbgp.conditioned_mu(domain, xobsPlusX0, yObsPlusY0, lenscale, sigvar, noisevar2)
            evmax = mu0.max()
            out['ev'][ix0] += evmax * py0

    return out
