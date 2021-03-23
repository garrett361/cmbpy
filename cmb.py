# simple 2d power spectra generation with some simple post-analysis

import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import scipy.stats as stats
import corner


class cmb():

    """
    class for generating power-law power spectra on a 2^(size_exponent) by 2^(size_exponent) grid
    currently only doing dimensions=2 case, but may generalize later
    """

    def __init__(self, amplitude=1, power=1.75, size_exponent=3, dimensions=2):
        self.amplitude = amplitude
        self.power = power
        self.size = 2**(size_exponent)
        self.dimensions = dimensions

    # generating the spectrum

    def generate(self):
        """
        generating the spectrum
        """
        # creating a grid w/ each point pulled from std normal
        gaussianseed = np.random.normal(
            size=[self.size for _ in range(self.dimensions)])
        # can fourier transform as below
        gaussianseedfourier = np.fft.fft2(gaussianseed)
        # numpy's fft algorithm automatically indexes with negative values on right half
        # positive on left half, as desired

        # relevant momenta vectors with desired fft indexing
        kvector = [(2*np.pi)*i for i in range(int(self.size/2+1))] + \
            [(2*np.pi)*i for i in range(int(-self.size/2+1), 0)]
        # power spectrum function

        def powerspectrum(i, j):
            if i == j == 0:
                return 0
            else:
                ktotal = np.sqrt(kvector[i]**2+kvector[j]**2)
                return self.amplitude*(ktotal**(-self.power))
        # create grid of relevant (square-root of) power spectrum values
        powergridsqrt = np.array([[np.sqrt(powerspectrum(i, j))
                                   for i in range(self.size)] for j in range(self.size)])
        # multiply by the transformed white noise
        gaussianseedfourierwithspectra = gaussianseedfourier*powergridsqrt
        # transform back and take the real part to get the spectrum
        self.spectrum = np.real(np.fft.ifft2(gaussianseedfourierwithspectra))
        # imaginary parts are from numerical errors; can check they're very small

    def spectrumplot(self):
        """
        plotting the spectrum
        """
        if hasattr(self, 'spectrum'):
            _, ax = plt.subplots()
            ax.imshow(self.spectrum)
            plt.show()
            plt.close()
        else:
            print('Run generate to create spectrum first')

    def hist(self):
        if not hasattr(self, 'spectrum'):
            return print('Run generate to create spectrum first')

        data = self.spectrum.flatten()
        std = np.std(data)
        self.std = std
        # plot data
        _, ax = plt.subplots()
        ax.hist(data, bins=100, density=True)
        ax.set_xlabel(f'$\sigma$={std:.5f}')
        ax.set_title('Normalized distribution of generated points')
        # plot fit
        x = np.linspace(-5*std, 5*std, 100)
        y = stats.norm.pdf(x, 0, std)
        ax.plot(x, y)
        plt.show()
        plt.close()

    def get_pair_data(self, maxlen=None):
        """
        getting the various relevant data points from the position-space spectrum
        data points are tuples containing an array with the two signals
        and two more arrays with their locations
        maxlen throws out all data points whose distance is larger than maxlen
        (bottleneck, should improve)
        """
        # dynamically choosing maxlen:
        # (should probably be adding 1 to maxlen everywhere due to how we later round for binning)
        if maxlen == None:
            maxlen = int(self.size//10)
        data = []
        for i in range(maxlen, self.size):
            for j in range(maxlen, self.size):
                for a in range(max(i-maxlen, 0), i):
                    for b in range(max(j-maxlen, 0), j):
                        if (i-a)**2+(j-b)**2 < maxlen**2:
                            data.append((np.array([self.spectrum[i][j], self.spectrum[a][b]]), np.array(
                                [i, j]), np.array([a, b])))

        print('Independent data pairs analyzed:', len(data))
        self.pair_data = data

    def bin_pair_data(self):
        """
        bin data by distances
        """
        if not hasattr(self, 'pair_data'):
            return print('Run get_pair_data first')

        if not hasattr(self, 'binned_pair_data'):
            def dist(p1, p2):
                return np.sqrt(np.dot(p1-p2, p1-p2))
            # create dictionary whose keys are the distance between points, rounded down
            datadict = {}
            for d in self.pair_data:
                d_dist = int(round(dist(d[1], d[2])))
                if d_dist not in datadict:
                    datadict[d_dist] = [np.prod(d[0])]
                else:
                    datadict[d_dist].append(np.prod(d[0]))
            self.binned_pair_data = datadict

        # plot all data points first:
        _, ax = plt.subplots()

        # then create means of data points
        # data points
        binnedxpoints = [key for key in self.binned_pair_data]
        meanbinnedydata = [np.mean(self.binned_pair_data[key])
                           for key in self.binned_pair_data]
        # standard deviation on y measurements
        # just for plotting purposes, only doing rough estimate for LR fitting
        meanbinnedystds = [np.std(self.binned_pair_data[key])/np.sqrt(len(self.binned_pair_data[key]))
                           for key in self.binned_pair_data]

        ax.errorbar(binnedxpoints, meanbinnedydata,yerr=meanbinnedystds, marker='^', color='g')

        # perform simple lin-reg analysis on binned data
        # for scaling ansatz that P(x)=A*x^B
        lrx = np.array([[1, np.log(value)] for value in binnedxpoints])
        # take abs of the values, just to prevent ln errors
        # should only affect some end points and this is a rough analysis anyway
        lry = np.array([np.log(np.abs(value)) for value in meanbinnedydata])
        lry = lry.reshape([-1, 1])
        xxmatrix = lrx.T.dot(lrx)
        xxinv = np.linalg.inv(xxmatrix)
        betafit = xxinv.dot(lrx.T).dot(lry)
        # so, the predicted function is
        def ypredictfn(x): return np.exp(betafit[0])*(x**(betafit[1]))
        xpredictions = np.linspace(
            binnedxpoints[0], binnedxpoints[-1], num=100)
        ypredictions = ypredictfn(xpredictions)
        ax.plot(xpredictions, ypredictions, linestyle='--', color='r')
        xpower = betafit[1][0]
        kpower = -xpower-self.dimensions
        ax.set_title(
            f'$y\\approx  {np.exp(betafit[0][0]):2f} \\cdot x^{{{xpower:2f}}}\\longleftrightarrow P(k)\\sim k^{{{kpower:2f}}} $')

        plt.show()
        plt.close()

        # save fits2f
        self.lr_fits = {'amplitude': np.exp(betafit[0][0]),
                        'xpower': betafit[1][0]}

    def _log_likelihood_xspace(self, theta, data):
        """
        the log-likelihood for the model, to be called in various places
        theta=(ampltiude,power)
        data= binned values from self.binned_pair_data dictionary
        """
        # we want to compare against power law models
        # probabilities only depend on the distance
        # for fixed distance at two points i,j, the prob for
        # with p(i,j)\propto exp[s[i]s[j]/(amplitude*dist(i,j)**power)]
        # for points i,j on grid whose respective signals are s[i],s[j]
        amplitude, power = theta
        # distance function

        def dist(p1, p2):
            return np.sqrt(np.dot(p1-p2, p1-p2))

        # inverse variance matrix
        # need to introduce some large cutoff on the diagonal elements
        def xspaceinversevariance(p1, p2):
            cutoff = 10**5
            var = amplitude * np.array([[cutoff, dist(p1, p2)**(power-self.dimensions)],
                                        [dist(p2, p1)**(power-self.dimensions), cutoff]])
            varinv = np.linalg.inv(var)
            return varinv

        # individual likelihoods from data point d
        def individualll(
            d): return -.5*(np.log(np.linalg.det(xspaceinversevariance(d[1], d[2])))-.5*np.linalg.multi_dot([d[0], xspaceinversevariance(d[1], d[2]), d[0]]))

        return sum(map(individualll, data))

    def MLE(self):
        """
        MLE analysis using scikit's minimize
        """
        if not hasattr(self, 'data'):
            return print('Run get_pair_data first')
        # maximizing log-likelihood (by minimizing its negative)
        negll = lambda *args: -1*self._log_likelihood_xspace(*args)
        initial = np.array([self.amplitude, self.power])
        MLEsoln = minimize(negll, initial, args=self.pair_data)
        self.MLEsoln = MLEsoln

    def bayes_xspace(self, amplitude_max=10,
                     power_max=2,
                     nsteps=10**4,
                     walkers=32):
        """
        running bayesian analysis w/ flat priors
        """

        if not hasattr(self, 'data'):
            return print('Run get_pair_data first')

        # set flat priors on the amplitude and power

        def log_prior(theta):
            amplitude, power = theta
            if 0 < amplitude < amplitude_max and 0 < power < power_max:
                return 0.0
            return -np.inf

        # total log-prob needed for MCMC
        def log_prob(theta, data):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp+self._log_likelihood_xspace(theta, data)

        # run MCMC with walkers starting at random points in the prior range
        pos = np.concatenate((amplitude_max*np.random.rand(walkers, 1),
                              power_max*np.random.rand(walkers, 1)), axis=1)

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, args=[self.pair_data])
        sampler.run_mcmc(pos, nsteps, progress=True)

        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["amplitude", "power"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()

        # check auto-correlation time
        # tau = sampler.get_autocorr_time()
        # get samples, after burning and thinning
        # flat_samples = sampler.get_chain(discard=int(
        #     np.mean(tau)*3), thin=int(np.mean(tau)//2), flat=True)

        flat_samples = sampler.get_chain(discard=int(nsteps//10), flat=True)
        self.MCMC_samples = flat_samples

    def corner(self):
        if not hasattr(self, 'MCMC_samples'):
            return print('Run bayes_xspace first')
        labels = ["amplitude", "power"]
        fig = corner.corner(self.MCMC_samples, labels=labels,
                            truths=[1, self.power])
        plt.show()
        plt.close()


if __name__ == '__main__':
    c = cmb(size_exponent=8, amplitude=1, power=1.2)
    c.generate()
    c.spectrumplot()
    c.hist()
    c.get_pair_data(maxlen=10)
    c.bin_pair_data()
    # c.MLE()
    # print(c.MLEsoln)
    # c.bayes_xspace(nsteps=500,walkers=100)
    # c.corner()
