import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# ----------------------------
# Physical parameters (pressure-driven Poiseuille)
# ----------------------------
DeltaP = 100.0     # pressure drop
mu     = 1.0       # viscosity
Lpipe  = 10.0      # pipe length (also used as visual length)

# ----------------------------
# Pipe + simulation parameters
# ----------------------------
R_values = [1.0, 2.0]  # run both at once
dt = 0.02
N  = 3000
D  = 0.002         # sideways diffusion (set 0 for perfectly layered laminar flow)

def v_poiseuille(r, R, DeltaP, mu, Lpipe):
    # v(r) = (DeltaP/(4 mu L)) * (R^2 - r^2)
    return (DeltaP / (4.0 * mu * Lpipe)) * (R**2 - r**2)

# Theory (for printing)
print("=== Theory (Poiseuille) ===")
for R in R_values:
    Q_theory = (np.pi * DeltaP * R**4) / (8.0 * mu * Lpipe)
    vmax = (DeltaP * R**2) / (4.0 * mu * Lpipe)
    vavg = vmax / 2.0
    print(f"R = {R}")
    print(f"v_max = {vmax:.4f}, v_avg = {vavg:.4f}")
    print(f"Q = pi*DeltaP*R^4/(8*mu*L) = {Q_theory:.4f}")

def project_into_circle(x, y, R):
    rr = np.sqrt(x*x + y*y)
    outside = rr > R
    if np.any(outside):
        x[outside] *= (R / rr[outside])
        y[outside] *= (R / rr[outside])
    return x, y

def init_state(R):
    # Uniform over cross-sectional area: r = R*sqrt(U)
    theta = 2*np.pi*np.random.rand(N)
    r0 = R*np.sqrt(np.random.rand(N))
    x = r0*np.cos(theta)
    y = r0*np.sin(theta)
    z = Lpipe*np.random.rand(N)

    rr0 = np.sqrt(x*x + y*y)
    v0 = v_poiseuille(rr0, R, DeltaP, mu, Lpipe)
    return x, y, z, v0

def setup_axes(ax, R, vmax):
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.set_xlim(0, Lpipe)
    ax.set_ylim(-R, R)
    ax.set_xlabel("z (along pipe)")
    ax.set_ylabel("y (radius)")
    ax.set_title(f"Laminar pipe flow tracers | R={R}")

    ax.plot([0, Lpipe], [ R,  R], color="white", linewidth=2)
    ax.plot([0, Lpipe], [-R, -R], color="white", linewidth=2)

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.0, vmax=vmax)
    return cmap, norm

# ----------------------------
# Plot (side view): z vs y  (BLACK THEME)
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="black", constrained_layout=True)

states = []
scatters = []

for ax, R in zip(axes, R_values):
    vmax = (DeltaP * R**2) / (4.0 * mu * Lpipe)
    cmap, norm = setup_axes(ax, R, vmax)
    x, y, z, v0 = init_state(R)

    sc = ax.scatter(z, y, s=6, c=v0, cmap=cmap, norm=norm, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("axial speed v(r)", color="white")
    cbar.outline.set_edgecolor("white")
    cbar.ax.set_facecolor("black")
    cbar.ax.tick_params(colors="white")
    for t in cbar.ax.get_yticklabels():
        t.set_color("white")

    states.append({"R": R, "x": x, "y": y, "z": z})
    scatters.append(sc)

def update(_):
    artists = []
    for state, sc in zip(states, scatters):
        R = state["R"]
        x = state["x"]
        y = state["y"]
        z = state["z"]

        rr = np.sqrt(x*x + y*y)
        v = v_poiseuille(rr, R, DeltaP, mu, Lpipe)

        z = z + v*dt

        if D > 0:
            x = x + np.sqrt(2*D*dt)*np.random.randn(N)
            y = y + np.sqrt(2*D*dt)*np.random.randn(N)
            x, y = project_into_circle(x, y, R)

        z = np.mod(z, Lpipe)

        state["x"] = x
        state["y"] = y
        state["z"] = z

        sc.set_offsets(np.c_[z, y])
        sc.set_array(v)
        artists.append(sc)

    return artists

ani = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()