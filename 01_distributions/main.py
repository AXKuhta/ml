import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

#
# Experimental and theoretical cumulative probability + density
#
def plot_distributions(d, title, n=1000):
	v = d.rvs(1000)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	fig.suptitle(title)
	ax1.set_title("Probability density")
	ax2.set_title("Cumulative distribution")

	lower_t = d.ppf(0.025)
	upper_t = d.ppf(0.975)

	lower_e = np.quantile(v, 0.025)
	upper_e = np.quantile(v, 0.975)

	print(lower_t, upper_t)
	print(lower_e, upper_e)

	print(f"95% CI Experimental: {lower_e:.2f}..{upper_e:.2f}")
	print(f"95% CI Theoretical: {lower_t:.2f}..{upper_t:.2f}")

	bins, edges, _ = ax1.hist(v, bins=int(n**0.5), density=True, label="Experimental")
	bins, edges, _ = ax2.hist(v, bins=int(n**0.5), density=True, cumulative=True)

	x = edges

	if "pdf" in dir(d):
		ax1.step(x, d.pdf(x))

	ax2.step(x, d.cdf(x), label="Theoretical")

	fig.legend()

	fig.show()

#
# mu and sigma vs. hparam
#
def mu_sigma_sweep_uniform():
	a_ = np.linspace(0.0, 1.0, 10)
	b_ = np.linspace(0.0, 1.0, 3)
	avg = []
	std = []

	for b in b_:
		avg_ = []
		std_ = []

		for a in a_:
			d = ss.uniform(a, b)
			x = d.rvs(1000)

			avg_.append( np.mean(x) )
			std_.append( np.std(x) )

		avg.append(avg_)
		std.append(std_)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	fig.suptitle("uniform(a, b)")
	ax1.set_title("mu")
	ax2.set_title("sigma")
	ax1.set_xlabel("a")
	ax2.set_xlabel("a")

	for i, b in enumerate(b_):
		ax1.plot(a_, avg[i], label=f"b={b:.1f}")
		ax2.plot(a_, std[i])

	fig.legend()
	fig.show()

def mu_sigma_sweep_norm():
	a_ = np.linspace(0.0, 1.0, 10)
	b_ = np.linspace(0.0, 1.0, 3)
	avg = []
	std = []

	for b in b_:
		avg_ = []
		std_ = []

		for a in a_:
			d = ss.norm(a, b)
			x = d.rvs(1000)

			avg_.append( np.mean(x) )
			std_.append( np.std(x) )

		avg.append(avg_)
		std.append(std_)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	fig.suptitle("norm(a, b)")
	ax1.set_title("mu")
	ax2.set_title("sigma")
	ax1.set_xlabel("a")
	ax2.set_xlabel("a")

	for i, b in enumerate(b_):
		ax1.plot(a_, avg[i], label=f"b={b:.1f}")
		ax2.plot(a_, std[i])

	fig.legend()
	fig.show()

def mu_sigma_sweep_chi2():
	a_ = np.linspace(0.0, 1.0, 10)
	b_ = np.arange(3, 6)
	avg = []
	std = []

	for b in b_:
		avg_ = []
		std_ = []

		for a in a_:
			d = ss.norm(a, b)
			x = d.rvs(1000)

			avg_.append( np.mean(x) )
			std_.append( np.std(x) )

		avg.append(avg_)
		std.append(std_)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	fig.suptitle("chi2(df, a)")
	ax1.set_title("mu")
	ax2.set_title("sigma")
	ax1.set_xlabel("a")
	ax2.set_xlabel("a")

	for i, b in enumerate(b_):
		ax1.plot(a_, avg[i], label=f"df={b}")
		ax2.plot(a_, std[i])

	fig.legend()
	fig.show()

#plot_distributions(ss.uniform(1), "uniform(1)")
#plot_distributions(ss.bernoulli(0.5), "bernoulli(0.5)")
#plot_distributions(ss.poisson(0.5), "poisson(0.5)")
#plot_distributions(ss.binom(10, 0.5), "binom(10, 0.5)")
#plot_distributions(ss.norm(1, 1), "norm(1, 1)")
#plot_distributions(ss.chi2(5), "chi2(5)")
#plot_distributions(ss.t(999), "t(999)")

#mu_sigma_sweep_uniform()
#mu_sigma_sweep_norm()
mu_sigma_sweep_chi2()
