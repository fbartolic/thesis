import starry
import matplotlib.pyplot as plt

starry.config.lazy = False
starry.config.quiet = True

ydeg = 5
fig, ax = plt.subplots(ydeg + 1, 2 * ydeg + 1, figsize=(12, 6))
fig.subplots_adjust(hspace=0)
for axis in ax.flatten():
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
for l in range(ydeg + 1):
    ax[l, 0].set_ylabel(
        "l = %d" % l,
        rotation="horizontal",
        labelpad=20,
        y=0.38,
        fontsize=10,
        alpha=0.5,
    )
for j, m in enumerate(range(-ydeg, ydeg + 1)):
    ax[-1, j].set_xlabel("m = %d" % m, labelpad=10, fontsize=10, alpha=0.5)

# Loop over the orders and degrees
map = starry.Map(ydeg=ydeg)
for i, l in enumerate(range(ydeg + 1)):
    for j, m in enumerate(range(-l, l + 1)):

        # Offset the index for centered plotting
        j += ydeg - l

        # Compute the spherical harmonic
        # with no rotation
        map.reset()
        if l > 0:
            map[l, m] = 1

        # Plot circle
        circ = plt.Circle((0, 0), 1.01, edgecolor="k", facecolor="none", linewidth=1.0)
        ax[i, j].add_patch(circ)

        # Plot the spherical harmonic
        ax[i, j].imshow(
            map.render(),
            cmap="OrRd",
            interpolation="none",
            origin="lower",
            extent=(-1, 1, -1, 1),
            zorder=-1,
        )
        ax[i, j].set_xlim(-1.1, 1.1)
        ax[i, j].set_ylim(-1.1, 1.1)
        ax[i, j].set_aspect(1)
        ax[i, j].set_rasterization_zorder(0)

# Save to file
plt.savefig("spherical_harmonics.pdf", bbox_inches="tight", dpi=200)
