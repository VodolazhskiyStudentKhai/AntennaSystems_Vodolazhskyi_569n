import numpy
import matplotlib.pyplot as plt
import math
from scipy.special import jv   # Функції Бесселя: jv(0,u)=J0(u), jv(1,u)=J1(u), jv(2,u)=J2(u)

# ======================= ВХІДНІ ДАНІ =======================
# Табл. 7: варіант 5 -> λ = 3,65 см, D = 1,2 м, f = 0,45 м
variant = 5
lam_list = [3.0, 3.2, 3.4, 3.5, 3.65, 2.5, 2.6, 2.7, 2.8, 2.9,
            3.0, 3.1, 4.45, 3.0, 3.0, 2.8, 2.8, 3.1, 3.1, 3.6,
            3.4, 2.5, 3.2, 3.3, 3.4]                          # λ, см
D_list = [0.8, 0.9, 1.0, 1.1, 1.2, 0.5, 0.5, 0.6, 0.6, 0.7,
          0.6, 0.7, 1.2, 0.6, 0.8, 1.1, 0.7, 1.0, 1.3, 1.35,
          1.4, 0.9, 1.2, 1.3, 1.4]                            # D, м
f_list = [0.4, 0.4, 0.45, 0.45, 0.45, 0.25, 0.2, 0.25, 0.3, 0.3,
          0.2, 0.4, 0.4, 0.25, 0.3, 0.25, 0.3, 0.3, 0.4, 0.3,
          0.4, 0.25, 0.3, 0.35, 0.4]                          # f, м

lam = lam_list[variant - 1] / 100      # м
D = D_list[variant - 1]                # м
f = f_list[variant - 1]                # м
print(f"Варіант {variant}: λ = {lam_list[variant - 1]} см, D = {D} м, f = {f} м")

# ---------------- Проміжні величини ----------------
c = 3 * 10 ** 8
f0 = c / lam                # робоча частота
k = 2 * numpy.pi / lam      # хвильове число (1)
R0 = D / 2                  # радіус розкриву (2)
p = 2 * f                   # параметр параболоїда (3)
v = 3.5 * R0 / p            # аргумент ДС
print(f"f0 = c/λ = {f0 / 10 ** 9:.3f} ГГц")
print(f"k = 2π/λ = {k:.2f} рад/м")
print(f"R0 = D/2 = {R0} м;  p = 2f = {p} м;  v = 3.5·R0/p = {v:.4f}")

# п'єдестали та енергетичні параметри (для звіту)
psi0 = 2 * math.atan(R0 / p)                                                    # половина кута розкриву
T_E = math.cos(psi0 / 2) ** 2 * math.cos(psi0) * math.sin(math.pi / 2 * math.cos(psi0))   # (7.25)
T_H = math.cos(psi0 / 2) ** 2 * math.sin(math.pi / 2 * math.cos(psi0))          # (7.26)
v_a = 7.5 * (R0 / p) ** 2 / (1 + (R0 / p) ** 2) ** 4
eta = 1 - 5 / 8 * math.cos(psi0) ** 3 - 3 / 8 * math.cos(psi0) ** 5             # (7.35)
print(f"ψ0 = {math.degrees(psi0):.2f}°;  T_E = {T_E:.3f};  T_H = {T_H:.3f}")
print(f"апертурний КВП (7.31) = {v_a:.3f};  ККД «переливання» (7.35) = {eta:.3f}")

# ---------------- Нормована ДС дзеркала (7.16), (7.17) ----------------
eps = 1e-9

def F(teta_deg, plane):
    """Нормована ДС ДЗА; plane = 'E' або 'H'."""
    teta = math.radians(teta_deg)
    u = k * R0 * math.sin(teta)
    J0v, J1v = jv(0, v), jv(1, v)
    # перший дріб: при u = v розкриваємо 0/0 за правилом Лопіталя
    if abs(v ** 2 - u ** 2) < eps:
        t1 = (J1v ** 2 + J0v ** 2) / 2
    else:
        t1 = (v * J1v * jv(0, u) - u * jv(1, u) * J0v) / (v ** 2 - u ** 2)
    # J1(u)/u: при u -> 0 границя 0.5
    t2 = 0.5 if abs(u) < eps else jv(1, u) / u
    # третій дріб: при u = 1.5v розкриваємо 0/0 за правилом Лопіталя
    w = 1.5 * v
    if abs(w ** 2 - u ** 2) < eps:
        t3 = -(w * jv(0, w) * jv(2, w) - w * jv(1, w) ** 2 + 2 * jv(1, w) * jv(2, w)) / (2 * w)
    elif abs(u) < eps:
        t3 = 0.0
    else:
        t3 = (u * jv(1, u) * jv(2, w) - w * jv(2, u) * jv(1, w)) / (w ** 2 - u ** 2)
    norm = 0.74 * J1v / v + 0.13
    sign = 0.25 if plane == 'E' else -0.25
    return math.cos(teta / 2) ** 2 * (0.74 * t1 + 0.26 * t2 + sign * t3) / norm

# ---------------- Розрахунок ДС, θ ∈ [0°, +90°] ----------------
steps, FE, FH = [], [], []
for td in numpy.arange(0.0, 90.0, 0.01):
    steps.append(td)
    FE.append(F(td, 'E'))
    FH.append(F(td, 'H'))
FE_n = [x / FE[0] for x in FE]
FH_n = [x / FH[0] for x in FH]

def null_points(plane):
    """Напрями нульових випромінювань (уточнення бісекцією)."""
    vals = FE if plane == 'E' else FH
    nulls = []
    for i in range(1, len(vals)):
        if vals[i - 1] * vals[i] < 0:
            a, b, fa = steps[i - 1], steps[i], vals[i - 1]
            for _ in range(25):
                mid = 0.5 * (a + b)
                fm = F(mid, plane)
                if fa * fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            nulls.append(0.5 * (a + b))
    return nulls

def hpbw(plane):
    """Ширина головної пелюстки на рівні 0.707 (ДС симетрична, максимум при θ=0)."""
    vals = FE_n if plane == 'E' else FH_n
    i = 0
    while i < len(vals) - 2 and vals[i] > 0.707:
        i += 1
    x1, x2, y1, y2 = steps[i - 1], steps[i], vals[i - 1], vals[i]
    th = x1 + (0.707 - y1) * (x2 - x1) / (y2 - y1)   # лінійна інтерполяція
    return 2 * th, th

def sidelobes(plane):
    """Максимури |F| бічних пелюсток після першого нуля."""
    vals = FE if plane == 'E' else FH
    n0 = null_points(plane)[0]
    sl = []
    for i in range(1, len(vals) - 1):
        if steps[i] > n0 and abs(vals[i]) > abs(vals[i - 1]) and abs(vals[i]) > abs(vals[i + 1]) \
                and abs(vals[i]) > 1e-6:
            sl.append((steps[i], abs(vals[i])))
    return sl

nulls_E, nulls_H = null_points('E'), null_points('H')
sl_E, sl_H = sidelobes('E'), sidelobes('H')
HPBW_E, th_E = hpbw('E')
HPBW_H, th_H = hpbw('H')
SLL_E, SLL_H = sl_E[0][1], sl_H[0][1]

print(f"\nШирина головної пелюстки на рівні 0.707:")
print(f"  площина H: {HPBW_H:.2f}° (θ0.707 = {th_H:.3f}°)")
print(f"  площина E: {HPBW_E:.2f}° (θ0.707 = {th_E:.3f}°)")
# оцінка за табл. 7.1 методички для порівняння
rat = R0 / p
if 0.4 <= rat <= 1.0:
    cH = 61 + (63 - 61) * (rat - 0.4) / 0.2 if rat < 0.6 else (63 + (70 - 63) * (rat - 0.6) / 0.2 if rat < 0.8 else 70 + (79 - 70) * (rat - 0.8) / 0.2)
    print(f"  оцінка за табл. 7.1: (2θ0.5)_H ≈ {cH:.1f}·λ/2R0 = {cH * lam / (2 * R0):.2f}°")
print(f"\nПерші нулі ДС, H: {[round(x, 2) for x in nulls_H[:7]]}")
print(f"Перші нулі ДС, E: {[round(x, 2) for x in nulls_E[:7]]}")
print(f"Рівень 1-ї бічної пелюстки: H = {SLL_H:.4f} ({20 * math.log10(SLL_H):.1f} дБ), "
      f"E = {SLL_E:.4f} ({20 * math.log10(SLL_E):.1f} дБ)")

# ---------------- Загальна таблиця ----------------
def plane_data(plane):
    nulls = nulls_H if plane == 'H' else nulls_E
    sl = sl_H if plane == 'H' else sl_E
    return nulls, sl

len_value = max(len(nulls_H), len(nulls_E), len(sl_H), len(sl_E))
print("\nТабл. 1 - Аналіз ДС ДЗА в площинах H та E")
print("--------------------------------------------------------------------------")
print("| № | θ0H,°  | θ0E,°  | θmaxH,° | FH(θ) | θmaxE,° | FE(θ) |")
for i in range(len_value):
    v2 = f"{nulls_H[i]:6.2f}" if i < len(nulls_H) else "   -  "
    v3 = f"{nulls_E[i]:6.2f}" if i < len(nulls_E) else "   -  "
    v4 = f"{sl_H[i][0]:7.2f}" if i < len(sl_H) else "    -  "
    h4 = f"{sl_H[i][1]:5.3f}" if i < len(sl_H) else "  -  "
    v6 = f"{sl_E[i][0]:7.2f}" if i < len(sl_E) else "    -  "
    h6 = f"{sl_E[i][1]:5.3f}" if i < len(sl_E) else "  -  "
    print(f"{i + 1:2d}| {v2} | {v3} | {v4} | {h4} | {v6} | {h6} |")
print("--------------------------------------------------------------------------")

FH_abs = [abs(x) for x in FH_n]
FE_abs = [abs(x) for x in FE_n]

# ---------------- 1. Графік нормованої ДС, θ ∈ [0°, +90°] ----------------
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, FH_abs, linewidth=0.9, color="red", label="$|F_H(θ)|$")
ax.plot(steps, FE_abs, linewidth=0.9, color="blue", label="$|F_E(θ)|$")

# Рівень 0.707 для обох площин
ax.plot([0, th_E], [0.707, 0.707], 'k--', linewidth=0.6)
ax.plot([th_H, th_H], [0, 0.707], 'r--', linewidth=0.6)
ax.plot([th_E, th_E], [0, 0.707], 'b--', linewidth=0.6)
ax.plot(th_H, 0.707, 'ro', markersize=4)
ax.plot(th_E, 0.707, 'bo', markersize=4)
ax.annotate(f'H: $2θ_{{0.5}}={HPBW_H:.2f}°$\nE: $2θ_{{0.5}}={HPBW_E:.2f}°$',
            xy=(th_E, 0.707), xytext=(th_E + 3.5, 0.72),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7), fontsize=8)

# Нулі та максимуми бічних пелюсток (перші 12, щоб не захаращувати вісь)
N_pts = 12
ax.plot(nulls_H[:N_pts], [0] * N_pts, "o", markersize=2.5, color="darkgreen", label="$θ_0$ нулі H")
ax.plot(nulls_E[:N_pts], [0] * N_pts, "x", markersize=3, color="teal", label="$θ_0$ нулі E")
ax.plot([p_[0] for p_ in sl_H[:N_pts]], [p_[1] for p_ in sl_H[:N_pts]], "o", markersize=2.5, color="black", label="$θ_{max}$ бічн. H")
ax.plot([p_[0] for p_ in sl_E[:N_pts]], [p_[1] for p_ in sl_E[:N_pts]], "s", markersize=2.5, color="dimgray", label="$θ_{max}$ бічн. E")

ax.set_xlabel('θ, grad', fontsize=10)
ax.set_ylabel('|F(θ)|', fontsize=10)
ax.set_title("Нормована ДС ДЗА у площинах H та E", fontsize=11)
plt.xticks(numpy.arange(0, 91, 5), fontsize=8)
plt.yticks(numpy.arange(0, 1.1, 0.1), fontsize=8)
plt.ylim(-0.01, 1.03)
plt.xlim(0, 90)
plt.legend(loc="upper right", fontsize=8)
plt.grid(which='both', linestyle='--', linewidth=0.25, color='gray')
fig.savefig("ДС ДЗА у площинах H та E, лаб. 7.jpg", dpi=600)
plt.show()

# ---------------- 2. Полярна (радіальна) ДС ----------------
fig2 = plt.figure(figsize=(16 / 2.54, 16 / 2.54))
ax2 = fig2.add_subplot(111, projection='polar')
theta_rad = numpy.radians(steps)
ax2.plot(theta_rad, FH_abs, linewidth=0.9, color="red", label="$|F_H(θ)|$")
ax2.plot(theta_rad, FE_abs, linewidth=0.9, color="blue", label="$|F_E(θ)|$")
ax2.plot(-theta_rad, FH_abs, linewidth=0.9, color="red", alpha=0.8)   # симетрична половина
ax2.plot(-theta_rad, FE_abs, linewidth=0.9, color="blue", alpha=0.8)
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_thetagrids(range(0, 360, 15))
ax2.set_title("Полярна ДС ДЗА (симетрична відносно θ = 0°)", fontsize=11, pad=15)
ax2.set_rlabel_position(90)
ax2.set_ylim(0, 1.02)
ax2.grid(True, linestyle='--', linewidth=0.3)
ax2.legend(loc="lower left", fontsize=8)
fig2.savefig("ДС ДЗА радіальна, лаб. 7.jpg", dpi=600)
plt.show()