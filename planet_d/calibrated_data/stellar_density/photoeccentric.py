import numpy as np

def photoeccentric_maxprob(rho_circ, err_rho_circ, rho_star, err_rho_star, rho_star_max=20., npts=120, nsamp=0, plotfig=False, retvals=False, verbose=False):
    """Determine e, omega, and rho_* via a Photoeccentric analysis.

    :INPUTS:
      rho_circ, err_rho_circ : scalars
        Best-fit stellar density and uncertainty from a circular-orbit
        fit to the transit light curve.

      rho_star, err_rho_star : scalars
        Best-fit stellar density and uncertainty from other
        information, e.g. spectroscopic or asteroseismic analysis.

      rho_star_max : scalar
        Maximum density of star, in same units as above.

      npts : positive int
        Number of points in finite Maximimum-likelihood analysis.

      nsamp : int
        Number of samples to draw from 3D posterior
        distribution. Don't set it too high without testing it first;
        maybe ~10000 at most. If negative, directly display posteriors
        but don't draw samples.

      retvals : bool
        If True, return the (eccentricity, omega, g, rho_star) samples.
        If True and nsamp<1, return (nsamp, nsamp, nsamp, nsamp)

    :EXAMPLE:
      ::

        import transit

        # Quick:
        transit.photoeccentric_maxprob(1.5, 1, 4.18, 2.9,nsamp=-1,npts=240, plotfig=True)

        # Slow, but return samples:
        ecc, om, g, rho = transit.photoeccentric_maxprob(1.5, 1, 4.18, 2.9,nsamp=3000,npts=240, plotfig=True, retvals=True)


    :NOTES:
      Follows Sec. 3.4 of Dawson & Johnson (2012)
        """
    # 2015-07-21 13:57 IJMC: Created

    import tools
    #if plotfig:
    #    import corner
    import pylab as py
    import analysis as an

    inplines = ['', 'Inputs:']
    inplines.append('')

    if hasattr(rho_circ, '__iter__') and len(rho_circ)>1 and nsamp>0:
        inplines.append("rho*_circ:  (posterior input)")
        inplines.append("rho*_meas:  %1.3f +/- %1.3f" % (rho_star, err_rho_star))
        ecc0,om0,gs0,rho0 = photoeccentric_maxprob(2.0, 999999, rho_star, err_rho_star, nsamp=nsamp,npts=npts, plotfig=False, retvals=True, verbose=False)
        bins = np.linspace(0, 30, npts*2)
        #bincens = np.vstack((bins[1:], bins[0:-1])).mean(0)
        #rhostar_spec_prob, junk  = np.histogram(rho0, bins, normed=True)
        #rhostar_circ_prob, junk  = np.histogram(rho_circ, bins, normed=True)

        if rho_circ.size >= rho0.size:
            nstep = (1.0*rho_circ.size/rho0.size)
            nsamp0 = (np.arange(rho0.size)*nstep).astype(int)
        else:
            nsamp0 = np.random.uniform(0, rho_circ.size, rho0.size).astype(int)
        rhostar_fake = rho_circ[nsamp0]/gs0**3 
        MC = np.exp(-((rhostar_fake - rho_star) / err_rho_star)**2) > np.random.uniform(size=rhostar_fake.size)
        samp_ecc, samp_om, samp_gs, samp_rhostar = ecc0[MC], om0[MC], gs0[MC], rho0[MC]
        best_ewp = np.median(samp_ecc), np.median(samp_om), np.median(samp_rhostar)
        best_g = np.median(samp_gs)

    else:
        inplines.append("rho*_circ:  %1.3f +/- %1.3f" % (rho_circ, err_rho_circ))        
        inplines.append("rho*_meas:  %1.3f +/- %1.3f" % (rho_star, err_rho_star))

        om = np.linspace(0,2*np.pi, npts-1)
        ecc = np.linspace(0, .99, npts)
        rhostar = np.linspace(0, 20, npts+1)
        #om1, ecc1 = np.meshgrid(om, ecc)
        #g1 = (1. + ecc1 * np.sin(om1)) / (1. - ecc1**2)**0.5
        om2, ecc2, rhostar2 = np.meshgrid(om, ecc, rhostar)
        g2 = (1. + ecc2 * np.sin(om2)) / (1. - ecc2**2)**0.5
        loglik = -0.5 * (((g2**3 * rhostar2 - rho_circ)/err_rho_circ)**2 + ((rhostar2-rho_star)/err_rho_star)**2)
        best_indices = (loglik==loglik.max()).nonzero()
        best_ewp = ecc[best_indices[0]][0], om[best_indices[1]][0], rhostar[best_indices[2]][0]
        best_g = g2[best_indices[0],best_indices[1],best_indices[2]][0]

        if nsamp>0:
            samp_ecc, samp_om, samp_rhostar = np.array(tools.sample_3dcdf(np.exp(loglik), ecc, om, rhostar, nsamp=nsamp))

    tlines = ['', 'Max-Likelihood Outputs:','']
    tlines.append("eccentricity is: %1.2f" % best_ewp[0])
    tlines.append("omega is:        %1.2f " % best_ewp[1])
    tlines.append("g(e, omega) is:  %1.2f" % best_g)
    tlines.append("stellar density: %1.2f" % best_ewp[2])
    tlines.append("")

        
    if nsamp>0:
        nsamp1 = samp_ecc.size
        samp_g = (1. + samp_ecc * np.sin(samp_om)) / (1. - samp_ecc**2)**0.5
        lohi = np.round(nsamp1*np.array([.1587, .8413])).astype(int)
        sigma_ecc = np.diff(np.sort(samp_ecc)[lohi])
        sigma_om = np.diff(np.sort(samp_om)[lohi])
        sigma_g = np.diff(np.sort(samp_g)[lohi])
        sigma_rhostar = np.diff(np.sort(samp_rhostar)[lohi])
        vals = np.concatenate((best_ewp[0:2], (best_g, best_ewp[-1]))).ravel()
        limits = np.sort(np.vstack((samp_ecc, samp_om, samp_g, samp_rhostar)), axis=1)[:,lohi]
        lowers = vals - limits[:,0]
        uppers = limits[:,1] - vals

        for kk in range(4):
            tlines[kk+3] += '$,^{+%1.2f}_{-%1.2f}$' % (uppers[kk], lowers[kk])

        if plotfig:
            labs = ['$e$', '$\omega$', '$g(e, \omega)$', '$\\rho_{star}$']
            fig=corner.corner(np.vstack((samp_ecc, samp_om, samp_g, samp_rhostar)).T, plot_datapoints=False, labels=labs)
            for child in fig.get_children():
                if hasattr(child, 'get_xaxis'):
                    child.get_xaxis().get_label().set_fontsize(16)
                    child.get_yaxis().get_label().set_fontsize(16)

    else:
        samp_ecc, samp_om, samp_g, samp_rhostar = (nsamp, nsamp, nsamp, nsamp)
        cmap = py.cm.cubehelix
        cmap_r = py.cm.cubehelix_r
        linecol = 'k'
        linewid = 2

        prob = np.exp(loglik)
        conflevels = [[an.confmap(prob.sum(ii), val) for val in [.6827,.9545,.9973][::-1]] for ii in range(3)]

        
        if plotfig:
            fig = py.figure()
            ax_ecc = py.subplot(3,3,1)
            py.plot(ecc, prob.sum(2).sum(1), color=linecol, linewidth=linewid)
            ax_eccom = py.subplot(3,3,4)
            py.contourf(ecc, om, prob.sum(2).T, cmap=cmap)
            py.contour(ecc, om, prob.sum(2).T, conflevels[2], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_om = py.subplot(3,3,5)
            py.plot(om, prob.sum(2).sum(0), color=linecol, linewidth=linewid)
            ax_eccrho = py.subplot(3,3,7)
            py.contourf(ecc, rhostar, prob.sum(1).T, cmap=cmap)
            py.contour(ecc, rhostar, prob.sum(1).T, conflevels[1], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_omrho = py.subplot(3,3,8)
            py.contourf(om, rhostar, prob.sum(0).T, cmap=cmap)
            py.contour(om, rhostar, prob.sum(0).T, conflevels[0], linestyles=['solid', 'dashed', 'dashdot'], linewidths=linewid, cmap=cmap_r)
            ax_rho = py.subplot(3,3,9)
            py.plot(rhostar, prob.sum(1).sum(0), color=linecol, linewidth=linewid)

            rho_cumsum  = np.cumsum(prob.sum(1).sum(0)/prob.sum())
            rho_5sighi = py.find(rho_cumsum>0.9999994)
            rho_5siglo = py.find(rho_cumsum<(1.-0.9999994))
            if rho_5sighi.size>0:
                rho_upper = rhostar[rho_5sighi[0]]
            else:
                rho_upper = rhostar.max()
            if rho_5siglo.size>0:
                rho_lower = rhostar[rho_5siglo[-1]]
            else:
                rho_lower = rhostar.min()


            omlim = 0, 2*np.pi
            [ax.set_xlim(omlim) for ax in [ax_om, ax_omrho]]
            ax_eccom.set_ylim(omlim)
            [ax.set_ylim([rho_lower, rho_upper]) for ax in [ax_eccrho, ax_omrho]]
            ax_rho.set_xlim([rho_lower, rho_upper])

            [ax.set_xticklabels([]) for ax in [ax_ecc, ax_eccom, ax_om]]
            [ax.set_yticklabels([]) for ax in [ax_ecc, ax_om, ax_omrho, ax_rho]]
            [[tick.set_rotation(45) for tick in ax.get_xaxis().get_ticklabels()] \
                 for ax in [ax_eccrho, ax_omrho, ax_rho]]
            [[tick.set_rotation(45) for tick in ax.get_yaxis().get_ticklabels()] \
                 for ax in [ax_eccrho, ax_eccom, ax_ecc]]
            ax_rho.set_xlabel('$\\rho_*$')
            ax_omrho.set_xlabel('$\omega$')
            ax_eccrho.set_xlabel('$e$')
            ax_eccrho.set_ylabel('$\\rho_*$')
            ax_eccom.set_ylabel('$\omega$')
        
    
    if plotfig:
        ax = fig.add_axes([.4, .8, .57, .16])
        tools.textfig(inplines, ax=ax, fig=fig, fontsize=12)
        ax = fig.add_axes([.57, .57, .4, .2])
        tools.textfig(tlines, ax=ax, fig=fig, fontsize=14)
    

        


    if verbose:
        for line in inplines: print( line)
        for line in tlines: print( line)

    if retvals:
        ret = samp_ecc, samp_om, samp_g, samp_rhostar
    else:
        ret = None

    return ret

