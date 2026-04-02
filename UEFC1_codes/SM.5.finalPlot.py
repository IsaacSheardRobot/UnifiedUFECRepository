import numpy as np
import matplotlib.pyplot as plt


c_bar = 0.2          # m
b = 1.77             # m
c_h = 0.0775         # m
b_h = 0.525          # m
f_e = 0.6
l_h = 0.65           # m
CLw_nom = 0.65
CMw_nom = -0.15

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

# CLw and CLh
CLw = CLw_nom + a_w * alpha_rad
CLh = a_h * alpha_rad + a_e * alpha_e_rad

# Required x_cg / c_bar
xcg_over_c = (
    0.25 * CLw
    + V_h * CLh * (1 + 0.25 * (c_bar / l_h))
    - CMw_nom
) / (
    CLw
    + (c_bar / l_h) * V_h * CLh
)

# Nominal CG location
xcg_nom_over_c = 0.25 - CMw_nom / CLw_nom

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alpha_e_deg, xcg_over_c, color='red', linewidth=2,
         label=r"Required $x_{cg}/\bar{c}$")

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.xlabel(r"Trim elevator deflection, $\alpha_e^{trim}$ (deg)")
plt.ylabel(r"$x_{cg}/\bar{c}$")
plt.title(r"SM5(d): Required $x_{cg}/\bar{c}$ vs. $\alpha_e^{trim}$ for Plane Vanilla")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Nominal x_cg/c_bar = {xcg_nom_over_c:.4f}")
