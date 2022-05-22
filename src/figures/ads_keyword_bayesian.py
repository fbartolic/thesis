import numpy as np
from matplotlib import pyplot as plt


data = np.loadtxt("../data/misc/ads_keyword_bayes.csv", delimiter=",")
year, count, count_ref = data.T

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(year, count, color="white", edgecolor="k", linewidth=1.5)
ax.set_xlim(1980, 2021)
ax.set_ylabel("Number of entries for the keyword\n 'Bayesian' on NASA ADS")
ax.set_xlabel("year")

# Save as pdf
fig.savefig("ads_keyword_bayesian.pdf", bbox_inches="tight")

