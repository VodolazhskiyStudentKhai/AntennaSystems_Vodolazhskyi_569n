import numpy
import matplotlib.pyplot as plt
import math

# ======================= ВХІДНІ ДАНІ =======================
# Табл. 5: варіант 5 -> λ = 3,65 см, вид антени №2 за табл. 4
variant = 5

lam_list = [3.0, 3.2, 3.4, 3.5, 3.65, 2.5, 2.6, 2.7, 2.8, 2.9,
            3.0, 3.1, 3.15, 3.0, 3.0, 2.8, 2.8, 3.1, 3.1, 3.6,
            3.4, 2.5, 3.2, 3.3, 3.4]                    # λ, см
type_list = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
             2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
             3, 1, 2, 3, 1]                             # вид антени (табл. 4)

N_list  = [14, 5, 6]
L_list  = [1.6, 2.0, 1.6]
d_list  = [2.0, 4.0, 3.2]
nu_list = [1, 0, 0]

lam = lam_list[variant - 1]
ant = type_list[variant - 1]
N   = N_list[ant - 1]
Lsh = L_list[ant - 1]
d   = d_list[ant - 1]
nu  = nu_list[ant - 1]
a   = 2.3

print(f"Варіант {variant}: λ = {lam} см, вид антени №{ant}")
print(f"N = {N};  Lщ = {Lsh} см;  d = {d} см;  ν = {nu}")

lam_kr = 2 * a
lam_v  = lam / numpy.sqrt(1 - (lam / lam_kr) ** 2)      # довжина хвилі у хвилеводі
k = 2 * numpy.pi / lam
print(f"λкр = 2a = {lam_kr:.2f} см")
print(f"λхв = {lam_v:.3f} см")
print(f"k = {k:.3f} рад/см")
print(f"ξ = 2πd/λхв = {2 * numpy.pi * d / lam_v:.3f} рад "
      f"({math.degrees(2 * numpy.pi * d / lam_v):.1f}°)")
print(f"Напрям головної пелюстки (m=0): θгол = arcsin(λ/λхв) = "
      f"{math.degrees(math.asin(lam / lam_v)):.2f}°")

print("\nНулі ДС (5.22):")
for m in range(-3 * N, 3 * N + 1):
    s = m * lam / (d * N) + lam / lam_v - nu * lam / (2 * d)
    if abs(s) <= 1:
        t0 = math.asin(s)
        psi0 = k * d * math.sin(t0) - 2 * numpy.pi * d / lam_v + nu * numpy.pi
        note = "   (напрям макс. типу 0/0)" if abs(numpy.sin(psi0 / 2)) < 1e-9 else ""
        print(f"  m = {m:+3d}:  θ0 = {math.degrees(t0):+7.2f}°{note}")

print("\nБічні максимуми (5.23), рівень Fc = 1/(N·sin((2n+1)π/2N)):")
for nn in range(-3 * N, 3 * N + 1):
    if nn in (0, -1):
        continue
    x_p = (2 * nn + 1) * math.pi / (2 * N)
    if any(abs(x_p - j * math.pi) < math.pi / N - 1e-9
           for j in range(-2 * N, 2 * N + 1)):
        continue
    s = (2 * nn + 1) * lam / (2 * d * N) + lam / lam_v - nu * lam / (2 * d)
    if abs(s) <= 1:
        lvl = 1 / (N * abs(math.sin(x_p)))
        print(f"  n = {nn:+3d}:  θmax = {math.degrees(math.asin(s)):+7.2f}°,  Fc = {lvl:.3f}")

print(f"\nШирина головної пелюстки «по нулях» (5.25): "
      f"2θ0 = {2 * math.degrees(math.asin(lam / (d * N))):.2f}°")

# ---------------- Розрахунок ДС, θ ∈ [-90°, +90°] ----------------
steps, F1H, FC, FH, FE = [], [], [], [], []

for teta_deg in numpy.arange(-89.99, 90.0, 0.01):
    teta = math.radians(teta_deg)
    f1 = abs(numpy.cos(numpy.pi / 2 * numpy.sin(teta)) / numpy.cos(teta))
    psi = k * d * numpy.sin(teta) - 2 * numpy.pi * d / lam_v + nu * numpy.pi
    if abs(numpy.sin(psi / 2)) < 1e-12:
        f2 = 1.0
    else:
        f2 = abs(numpy.sin(N * psi / 2) / (N * numpy.sin(psi / 2)))
    steps.append(teta_deg)
    F1H.append(f1)
    FC.append(f2)
    FH.append(f1 * f2)
    FE.append(1.0)

FH_max = max(FH)
i_max = FH.index(FH_max)
FH_n = [v / FH_max for v in FH]

i = i_max
while FH_n[i] > 0.707:
    i -= 1
th_l = steps[i]
i = i_max
while FH_n[i] > 0.707:
    i += 1
th_r = steps[i]
SGP_glob = th_r - th_l

th_main = math.degrees(math.asin(lam / lam_v))
i0 = min(range(len(steps)), key=lambda j: abs(steps[j] - th_main))
while FH_n[i0 + 1] > FH_n[i0]:
    i0 += 1
while FH_n[i0 - 1] > FH_n[i0]:
    i0 -= 1
lvl0 = FH_n[i0]
i = i0
while FH_n[i] > 0.707 * lvl0:
    i -= 1
th_l0 = steps[i]
i = i0
while FH_n[i] > 0.707 * lvl0:
    i += 1
th_r0 = steps[i]
SGP_main = th_r0 - th_l0

max_x_FH, max_y_FH, min_x_FH, min_y_FH = [], [], [], []
max_x_FC, max_y_FC = [], []
for i in range(1, len(FH_n) - 1):
    if FH_n[i] > FH_n[i - 1] and FH_n[i] > FH_n[i + 1]:
        max_x_FH.append(steps[i]); max_y_FH.append(FH_n[i])
    if FH_n[i] < FH_n[i - 1] and FH_n[i] < FH_n[i + 1]:
        min_x_FH.append(steps[i]); min_y_FH.append(FH_n[i])
    if FC[i] > FC[i - 1] and FC[i] > FC[i + 1]:
        max_x_FC.append(steps[i]); max_y_FC.append(FC[i])

print(f"\nГлобальний максимум ДС: θ = {steps[i_max]:.2f}° (норм. рівень 1.000)")
print(f"Головна пелюстка (m=0): θ = {steps[i0]:.2f}°, норм. рівень {FH_n[i0]:.3f}")
print(f"Ширина найбільшої пелюстки на рівні 0.707: {SGP_glob:.2f}° "
      f"({th_l:.2f}° ... {th_r:.2f}°)")
print(f"Ширина головної пелюстки (m=0) на рівні 0.707: {SGP_main:.2f}° "
      f"({th_l0:.2f}° ... {th_r0:.2f}°)")
sll = max(y for x, y in zip(max_x_FH, max_y_FH)
          if abs(x - steps[i_max]) > 3 and abs(x - steps[i0]) > 3)
print(f"Рівень найвищої бічної пелюстки: {sll:.3f} ({20 * math.log10(sll):.1f} дБ)")

def f1h_deg(td):
    t = math.radians(td)
    return abs(numpy.cos(numpy.pi / 2 * numpy.sin(t)) / numpy.cos(t))

def fc_deg(td):
    t = math.radians(td)
    psi = k * d * numpy.sin(t) - 2 * numpy.pi * d / lam_v + nu * numpy.pi
    if abs(numpy.sin(psi / 2)) < 1e-12:
        return 1.0
    return abs(numpy.sin(N * psi / 2) / (N * numpy.sin(psi / 2)))

# ---------------- Загальна таблиця аналізу ДС ----------------
th_main = math.degrees(math.asin(lam / lam_v))
w0_25   = 2 * math.degrees(math.asin(lam / (d * N)))
z_left  = max(z for z in min_x_FH if z < th_main)
z_right = min(z for z in min_x_FH if z > th_main)
w0_fact = z_right - z_left

len_value = [len(min_x_FH), len(max_x_FH)]
print("Табл. 1 - Аналіз ДС ХвЩА (нормованої) в площині H")
print("-----------------------------------------------------------------------")
print("| № | θminH,° | θmaxH,° | F1н(θmax) | Fс(θmax) | FH(θ) | FH, дБ |")
for i in range(max(len_value)):
    v2 = f"{min_x_FH[i]:7.2f}" if i < len(min_x_FH) else "      -"
    v3 = f"{max_x_FH[i]:7.2f}" if i < len(max_x_FH) else "      -"
    c1 = f"{f1h_deg(max_x_FH[i]):9.3f}" if i < len(max_x_FH) else "        -"
    c2 = f"{fc_deg(max_x_FH[i]):8.3f}" if i < len(max_x_FH) else "       -"
    c3 = f"{max_y_FH[i]:5.3f}" if i < len(max_x_FH) else "     -"
    c4 = f"{20 * math.log10(max_y_FH[i]):6.1f}" if i < len(max_x_FH) else "     -"
    print(f"| {i + 1:2d}| {v2} | {v3} | {c1} | {c2} | {c3} | {c4} |")
print("-----------------------------------------------------------------------")
print("| Зведені параметри ДС:                                               |")
print(f"| θгол = {th_main:.2f}°;  2θ0 (5.25) = {w0_25:.2f}°;  2θ0факт = {w0_fact:.2f}° |")
print(f"| 2θ0.707 (найб. пел.) = {SGP_glob:.2f}°;  2θ0.707 (гол. пел.) = {SGP_main:.2f}° |")
print(f"| FH(θгол) = {FH_n[i0]:.3f};  SLL = {sll:.3f} ({20 * math.log10(sll):.1f} дБ) |")
print("-----------------------------------------------------------------------")
# ---------------- Графік ДС у площинах H та E ----------------
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, F1H, linewidth=0.7, label="$F_{1H}(θ)$ — одиночна щілин")
ax.plot(steps, FC, linewidth=0.7, label="$F_C(θ)$ — множник системи")
ax.plot(steps, FH_n, linewidth=0.9, color="red", label="$F_H(θ)$ — ДС антени, норм.")
ax.plot(steps, FE, linewidth=0.7, label="$F_E(θ)$ = 1")

ax.plot([th_l, th_r], [0.707, 0.707], 'r--', linewidth=0.6)
ax.plot([th_l, th_l], [0, 0.707], 'r--', linewidth=0.6)
ax.plot([th_r, th_r], [0, 0.707], 'r--', linewidth=0.6)
ax.plot(th_l, 0.707, 'ro', markersize=4)
ax.plot(th_r, 0.707, 'ro', markersize=4)
ax.annotate(f'({th_l:.2f}°, 0.707)', xy=(th_l, 0.707), xytext=(th_l - 22, 0.76),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)
ax.annotate(f'({th_r:.2f}°, 0.707)', xy=(th_r, 0.707), xytext=(th_r + 3, 0.76),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)

ax.plot(max_x_FH, max_y_FH, "o", markersize=4, color="black", label="$θ_{max}$ FH")
ax.plot(min_x_FH, min_y_FH, "o", markersize=3, color="blue", label="$θ_{min}$ FH")

ax.set_xlabel('θ, °', fontsize=10)
ax.set_ylabel('|F(θ)|', fontsize=10)
plt.xticks(numpy.arange(-90, 91, 10), fontsize=8)
plt.yticks(numpy.arange(0, 1.1, 0.1), fontsize=8)
plt.ylim(-0.01, 1.05)
plt.xlim(-90, 90)
plt.legend(loc="lower right", fontsize=7)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')
fig.savefig("ДС ХвЩА у площинах H та E, лаб. 5.jpg", dpi=600)
plt.show()

# ---------------- Радіальна (полярна) ДС, передня напівплощина ----------------
fig2 = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
ax = fig2.add_subplot(111, projection='polar')
theta_rad = numpy.radians(steps)

ax.plot(theta_rad, FH_n, linewidth=1.0, color="red", label="$F_H(θ)$, норм.")
ax.plot(theta_rad, FC, linewidth=0.6, color="blue", alpha=0.5, label="$F_C(θ)$, множник")

ax.scatter([numpy.radians(steps[i_max])], [1.0], color="black", s=25,
           label=f"гратинговий макс.")
ax.scatter([numpy.radians(steps[i0])], [FH_n[i0]], color="green", s=25,
           label=f"головна пелюстка m=0 )")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_thetagrids(range(-90, 91, 15))
ax.set_title("Радіальна ДС ХвЩА (площина H, передня напівплощина)", fontsize=12)
ax.set_rlabel_position(90)
ax.grid(True, linestyle='--', linewidth=0.3)
ax.legend(loc="lower left", fontsize=8)
fig2.savefig("ДС ХвЩА радіальна, лаб. 5.jpg", dpi=600)
plt.show()