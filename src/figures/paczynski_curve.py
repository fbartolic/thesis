import numpy as np
import matplotlib.pyplot as plt


tau = np.linspace(-1.1, 1.1, 1000)

u0 = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
colors = [
    "C0",
]

fig, ax = plt.subplots(figsize=(10, 8))

for i, u0_ in enumerate(u0):
    u = np.sqrt(u0_ ** 2 + tau ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    color = "C" + str(i)
    ax.plot(tau, A, color, label=r"$u_0=" + str(u0_) + "$")

# Inset plot
a = plt.axes([0.15, 0.5, 0.32, 0.32])
# a.grid(True)
a.set_xlim(-1.2, 1.2)
a.set_ylim(-1.2, 1.2)
circle = plt.Circle(
    (0, 0), 1.0, facecolor="white", edgecolor="k", linestyle="dashed", lw=1
)
a.add_artist(circle)
circle2 = plt.Circle((0, 0), 0.03, facecolor="black")
a.add_artist(circle2)
a.set_aspect(1)
x_ = np.linspace(0, 0.79)
a.plot(x_, 0.85 * x_, "k--", lw=1)
a.scatter([0.0], [0.0], marker="o", color="black")

plt.text(-0.8, -1.3, r"source trajectory", fontsize=16)
plt.text(0.8, 0.7, r"$\theta_E$", fontsize=16)
plt.text(-0.2, 0.1, r"L", fontsize=16)
plt.axis("off")

for i, u0_ in enumerate(u0):
    u = np.sqrt(u0_ ** 2 + tau ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    color = "C" + str(i)
    a.axhline(-u0_, color=color)

ax.grid(True)
ax.set_xlabel(r"$\tau=(t-t_0)/t_E$")
ax.set_ylabel(r"$A(u)$")
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.legend(prop={"size": 16})

# Save to disk
fig.savefig("paczynski_curve.pdf", bbox_inches="tight")
