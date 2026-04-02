import numpy as np
import matplotlib.pyplot as plt

# Plane Vanilla parameters
c_bar = 0.2        # m
b = 1.77           # m
c_h = 0.0775       # m
b_h = 0.525        # m
f_e = 0.6
l_h = 0.65         # m


S = b * c_bar
S_h = b_h * c_h

AR = b / c_bar
AR_h = b_h / c_h

V_h = (S_h * l_h) / (S * c_bar)

a_w = (2 * np.pi) / (1 + 2 / AR)
a_h = (2 * np.pi) / (1 + 2 / AR_h)

xi_e = np.arccos(1 - 2 * f_e)
a_e = 2 * (np.pi - xi_e + np.sin(xi_e)) / (1 + 2 / AR_h)

# alpha required to keep the airplane in trim when deflected by alpha_e
def alpha_required(alpha_e_rad):
    numerator = V_h * (c_bar / l_h)
    denominator = (a_w / a_e) + V_h * (c_bar / l_h) * (a_h / a_e)
    return -(numerator / denominator) * alpha_e_rad

alpha_e_deg = np.linspace(-10, 10, 400)
alpha_e_rad = np.deg2rad(alpha_e_deg)
alpha_rad = alpha_required(alpha_e_rad)
alpha_deg = np.rad2deg(alpha_rad)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alpha_e_deg, alpha_deg, color='red', linewidth=2, label=r"Required $\alpha$")
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel(r"Elevator deflection, $\alpha_e$ (deg)")
plt.ylabel(r"Required angle of attack, $\alpha$ (deg)")
plt.title(r"SM5(b): Required $\alpha$ vs. $\alpha_e$ for Plane Vanilla")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
