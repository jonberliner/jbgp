from numpy import linspace, zeros_like, diag, isscalar, ndarray
from jb import jbgp
from numpy.random import RandomState
rng = RandomState()
from matplotlib import pyplot as plt
from pylab import find


def check_obs(obs):
    if obs is None: obs_huh = False
    elif isscalar(obs): obs_huh = not obs==0
    elif type(obs) is ndarray: obs_huh = any(obs)
    elif type(obs) is list:
        obs_huh = obs[0] is not None
    else: raise ValueError('obs must be None, >=int, ndarray, or lo 2 ndarrays')
    return obs_huh


def gp_quick_condition(lenscale,
                       obs=None,
                       hidfcn=None,
                       domain=linspace(0, 1, 1028),
                       sigvar=1.,
                       noisevar2=1e-7):

    obs_huh = check_obs(obs)

    mu_prior = zeros_like(domain)
    covmat_prior = jbgp.K_se(domain, domain, lenscale, sigvar)

    if obs_huh:
        if isscalar(obs) and obs>0:
            if hidfcn is None:
                hidfcn = jbgp.sample(domain, mu_prior, covmat_prior, noisevar2)
            iobs = rng.randint(len(domain), size=obs)
            xobs = domain[iobs]
            yobs = hidfcn[iobs]
        elif type(obs) is ndarray:
            if hidfcn is None:
                hidfcn = jbgp.sample(domain, mu_prior, covmat_prior, noisevar2)
            iobs = obs
            xobs = domain[iobs]
            yobs = hidfcn[iobs]
        elif type(obs) is list:
            assert len(obs)==2, 'if list, must be list of two ndarrays w x and y vals'
            assert not hidfcn, 'cannot provide yvals if passing a hidfcn'
            iobs = None
            xobs = obs[0]
            yobs = obs[1]
        else:
            raise ValueError('obs must be an integer, a nparray, or a list of two nparrays')

        post_mu = jbgp.conditioned_mu(domain, xobs, yobs, lenscale, sigvar, noisevar2)
        post_covmat = jbgp.conditioned_covmat(domain, covmat_prior, xobs, lenscale, sigvar, noisevar2)
    else:
        post_mu = mu_prior
        post_covmat = covmat_prior
        xobs = None
        yobs = None
        iobs = None

    return {'mu': post_mu,
            'covmat': post_covmat,
            'xobs': xobs,
            'yobs': yobs,
            'iobs': iobs}


def gp_quick_sample(lenscale,
                    obs=None,
                    nsam=1,
                    hidfcn=None,
                    domain=linspace(0, 1, 1028),
                    sigvar=1.,
                    noisevar2=1e-7):

    mu_prior = zeros_like(domain)
    covmat_prior = jbgp.K_se(domain, domain, lenscale, sigvar)

    post = gp_quick_condition(lenscale, obs, hidfcn, domain, sigvar, noisevar2)

    samples = [jbgp.sample(domain, post['mu'], post['covmat'], noisevar2)
               for _ in xrange(nsam)]

    return {'samples': samples,
            'xobs': post['xobs'],
            'yobs': post['yobs'],
            'iobs': post['iobs']}


def gp_plot_demo(lenscale,
                 obs=None,
                 nsam=None,
                 hidfcn=None,
                 domain=linspace(0, 1, 1028),
                 sigvar=1.,
                 noisevar2=1e-7,
                 col='0.5',
                 show_mu=True,
                 show_sd=True,
                 show_legend=True,
                 ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    obs_huh = check_obs(obs)

    mu_prior = zeros_like(domain)
    covmat_prior = jbgp.K_se(domain, domain, lenscale, sigvar)
    post = gp_quick_condition(lenscale, obs, hidfcn, domain, sigvar, noisevar2)

    if nsam:
        # pdb.set_trace()
        samples = gp_quick_sample(lenscale, [post['xobs'], post['yobs']], nsam,
                                  None, domain, sigvar, noisevar2)['samples']

    h = {}
    # plot output
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    if show_sd:
        h['sd'] = ax.fill_between(domain, post['mu']+diag(post['covmat'])**2. * 1.96,\
                                post['mu']-diag(post['covmat'])**2. * 1.96,\
                                color=col, alpha=0.2)
    if show_mu:
        h['mu'] = ax.plot(domain, post['mu'], color=col, lw=2.5, alpha=0.8, label='mu')

    sams = []
    if nsam:
        for isam, sam in enumerate(samples):
            if isam==0:
                sams.append(ax.plot(domain, sam, color=col, lw=1, alpha=0.3, label='samples'))
                h['samples'] = sams[0]
            else:
                sams.append(ax.plot(domain, sam, color=col, lw=1, alpha=0.3))
    if obs_huh:
        h['observations'] = ax.plot(post['xobs'], post['yobs'], color=col,\
                                    marker='o', alpha=1.0, ms=8, mec='None',\
                                    mfc=col, ls='None', label='observations')
    if show_legend:
        order = []
        if show_mu: order.append('mu')
        if obs_huh: order.append('observations')
        if nsam: order.append('samples')
        labs = []
        hands = []
        handles, labels = ax.get_legend_handles_labels()
        for lab in order:
            ilab = find([l==lab for l in labels])[0]
            labs.append(labels[ilab])
            hands.append(handles[ilab])
        # sort both labels and handles by labels
        leg = ax.legend(hands, labs)
        leg.draw_frame(False)
    plt.axis('off')
    return ax

