import numpy
import matplotlib.pyplot as plt
import math

# ---------------- Дано ----------------
variant = 5

lam_table = [3, 3.2, 3.4, 3.5, 3.6, 3.7, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.45,
             3, 3, 2.8, 2.8, 3.1, 3.1, 3.6, 3.4, 2.5, 3.2, 3, 3]   # cm

h_table = [5, 8, 10, 12, 14, 15, 16, 15, 14, 12, 10, 8, 5,
           10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]         # cm

# ------------- Параметри стрижня (п. 4.7 завдання) -------------
eps_r = 2.2          # відносна діелектрична проникність (фторопласт)
tg_delta = 2e-4      # тангенс кута діелектричних втрат
d_max = 2.9          # cm
d_min = 1.7          # cm
d_ser = 2.3          # cm
l = 23.7             # cm - довжина стрижня

lam = lam_table[variant - 1]
h = h_table[variant - 1]

print(f"Варіант {variant}: lambda = {lam} (cm), h = {h} (cm)")
print(f"Стрижень: eps_r = {eps_r}, tg_delta = {tg_delta}, l = {l} (cm), d_сер = {d_ser} (cm)")

k = 2 * numpy.pi / lam                    # rad/cm
xi = 1 + lam / (2 * l)                    # оптимальне уповільнення (4.7)
A = numpy.pi * l / lam                    # = k*l/2
u0 = A * (xi - 1)                         # = pi/2 при xi_opt

print(f"k = {k:.4f} (rad/cm)")
print(f"l/lambda = {l / lam:.2f}")
print(f"xi_opt = 1 + lambda/(2l) = {xi:.4f}   (V_ф = C/xi = {299792458 / xi / 1e6:.1f} Мм/с)")

# КСД за (4.8) та (4.15)
D1 = 7.5 * l / lam
print(f"КСД однострижневої ДСА (4.8): D = (7...8)*l/lambda = {7 * l / lam:.1f}...{8 * l / lam:.1f}"
      f"  ->  {10 * numpy.log10(D1):.1f} dB")
print(f"КСД двострижневої ДСА (4.15): D = 2*D1 = {2 * D1:.1f}  ->  {10 * numpy.log10(2 * D1):.1f} dB")

# Оптимальна довжина хвилі стрижня (4.9)-(4.10)
s1 = 1 / numpy.sqrt(numpy.pi * (eps_r - 1))
s2 = 1 / numpy.sqrt(2.5 * numpy.pi * (eps_r - 1))
lam_opt = 2 * d_ser / (s1 + s2)
print(f"\nРозрахункова оптимальна довжина хвилі стрижня: lambda = {lam_opt:.2f} cm "
      f"(f = {299792458 / (lam_opt / 100) / 1e9:.2f} GHz)")


# ------- функція Бесселя J1 -------
def bessel_j1(x):
    x = numpy.asarray(x, dtype=float)
    half = x / 2.0
    term = half.copy()
    s = term.copy()
    q = half * half
    for m in range(1, 30):
        term = -term * q / (m * (m + 1))
        s = s + term
    return s


def lambda_func1(x):
    """Лямбда-функція першого порядку L1(x) = 2*J1(x)/x (табл. П.1)."""
    x = numpy.asarray(x, dtype=float)
    out = numpy.where(numpy.abs(x) < 1e-9, 1.0, 2 * bessel_j1(x) / numpy.where(numpy.abs(x) < 1e-9, 1.0, x))
    return numpy.abs(out)


# ---------------- Розрахунок ДС ----------------
steps = []
F1H = []
F1E = []
F2H = []
F2E = []

for teta in numpy.arange(0.0001, numpy.pi / 2, 0.0001):
    v = xi - numpy.cos(teta)                       # xi - cos(teta)
    F_B = abs((xi - 1) / numpy.sin(u0) * numpy.sin(A * v) / v)   # (4.3)

    x = numpy.pi * d_ser / lam * numpy.sin(teta)
    L1 = lambda_func1(numpy.array([x]))[0]
    F_1H = L1                                      # (4.5)
    F_1E = L1 * numpy.cos(teta)                    # (4.4)

    F_C = abs(numpy.cos(numpy.pi * h / lam * numpy.sin(teta)))   # (4.18)

    F1H += [F_1H * F_B]
    F1E += [F_1E * F_B]
    F2H += [F_1H * F_B * F_C]                      # (4.16)/(4.25)
    F2E += [F_1E * F_B * F_C]                      # (4.17)/(4.29)
    steps += [math.degrees(teta)]

max1H, max1E = max(F1H), max(F1E)
max2H, max2E = max(F2H), max(F2E)
F1H = [v / max1H for v in F1H]
F1E = [v / max1E for v in F1E]
F2H = [v / max2H for v in F2H]
F2E = [v / max2E for v in F2E]


# ---------------- Ширина головної пелюстки (рівень 0.707) ----------------
def hpbw(steps_deg, F):
    for i in range(1, len(F)):
        if F[i] < 0.707 <= F[i - 1]:
            t1, t2 = steps_deg[i - 1], steps_deg[i]
            f1, f2 = F[i - 1], F[i]
            th = t1 + (0.707 - f1) * (t2 - t1) / (f2 - f1)
            return 2 * th, th
    return float('nan'), float('nan')


SGP1H, th1H = hpbw(steps, F1H)
SGP1E, th1E = hpbw(steps, F1E)
SGP2H, th2H = hpbw(steps, F2H)
SGP2E, th2E = hpbw(steps, F2E)

print("\nШирина головної пелюстки на рівні 0.707 (розрахунок за ДС):")
print(f"  Однострижнева ДСА: площина H = {SGP1H:.2f} deg, площина E = {SGP1E:.2f} deg")
print(f"  Двострижнева ДСА:  площина H = {SGP2H:.2f} deg, площина E = {SGP2E:.2f} deg")
print(f"\nОцінка за наближеними формулами методички:")
print(f"  по нулях (4.22): 2*teta0 = 115*sqrt(lambda/l) = {115 * numpy.sqrt(lam / l):.1f} deg")
print(f"  по 0.5   (4.23): 2*teta0.5 = 61*sqrt(lambda/l) = {61 * numpy.sqrt(lam / l):.1f} deg")


print("\nНулі множника F_Б (4.19): teta0 = arccos(xi - m*lambda/l) у передній напівплощині:")
for m in range(1, 10):
    c = xi - m * lam / l
    if 0 <= c <= 1:
        print(f"  m = {m}: teta0 = {math.degrees(math.acos(c)):.2f} deg")

print("\nМаксимуми бічних пелюсток F_Б (4.20) та їх рівні (4.21) у передній напівплощині:")
for m in range(1, 10):
    c = xi - (2 * m + 1) / 2 * lam / l
    if 0 <= c <= 1:
        lvl = abs(2 * l * (xi - 1) / ((2 * m + 1) * lam * numpy.sin(u0)))
        print(f"  m = {m}: teta_m = {math.degrees(math.acos(c)):.2f} deg, "
              f"|F| = {lvl:.3f} ({20 * numpy.log10(lvl):.1f} dB)")

print("\nМножник ґрат F_C (4.26)-(4.27):")
for m in range(0, 8):
    s = (2 * m + 1) * lam / (2 * h)
    if s <= 1:
        print(f"  нуль: teta_0C = {math.degrees(math.asin(s)):.2f} deg (m = {m})")
for m in range(1, 8):
    s = m * lam / h
    if s <= 1:
        print(f"  максимум: teta_mC = {math.degrees(math.asin(s)):.2f} deg (m = {m})")


def extrema(steps_deg, F):
    mins_x, mins_y, maxs_x, maxs_y = [], [], [], []
    for i in range(1, len(F) - 1):
        if F[i] < F[i - 1] and F[i] < F[i + 1]:
            mins_x.append(steps_deg[i]); mins_y.append(F[i])
        # дрібні локальні максимуми (рівень < 0.001) біля збіжних нулів не враховуємо
        if F[i] > F[i - 1] and F[i] > F[i + 1] and F[i] > 0.001:
            maxs_x.append(steps_deg[i]); maxs_y.append(F[i])
    return mins_x, mins_y, maxs_x, maxs_y


def print_combined(title, steps_deg, FH, FE):
    mnH_x, mnH_y, mxH_x, mxH_y = extrema(steps_deg, FH)
    mnE_x, mnE_y, mxE_x, mxE_y = extrema(steps_deg, FE)
    n = max(len(mnH_x), len(mnE_x), len(mxH_x), len(mxE_x))
    print("\n" + title)
    print("-------------------------------------------------------------------------")
    print("| № | θminH,° | θminE,° | θmaxH,° | FH(θ) | θmaxE,° | FE(θ) |")
    print("-------------------------------------------------------------------------")
    for i in range(n):
        c1 = f"{mnH_x[i]:.2f}" if i < len(mnH_x) else "  -  "
        c2 = f"{mnE_x[i]:.2f}" if i < len(mnE_x) else "  -  "
        c3 = f"{mxH_x[i]:.2f}" if i < len(mxH_x) else "  -  "
        c4 = f"{mxH_y[i]:.3f}" if i < len(mxH_y) else "  -  "
        c5 = f"{mxE_x[i]:.2f}" if i < len(mxE_x) else "  -  "
        c6 = f"{mxE_y[i]:.3f}" if i < len(mxE_y) else "  -  "
        print(f"| {i + 1} | {c1} | {c2} | {c3} | {c4} | {c5} | {c6} |")
    print("-------------------------------------------------------------------------")
    if mxH_y:
        print(f"Рівень першої бічної пелюстки, площина H: {mxH_y[0]:.3f} "
              f"({20 * math.log10(mxH_y[0]):.1f} dB) при θ = {mxH_x[0]:.2f} deg")
    if mxE_y:
        print(f"Рівень першої бічної пелюстки, площина E: {mxE_y[0]:.3f} "
              f"({20 * math.log10(mxE_y[0]):.1f} dB) при θ = {mxE_x[0]:.2f} deg")


print_combined("Табл. 1 - Результати аналізу форми ДС однострижневої ДСА", steps, F1H, F1E)
print_combined("Табл. 2 - Результати аналізу форми ДС двострижневої ДСА", steps, F2H, F2E)

# ---------------- Лінійні ДС (передня напівплощина, 0..90 deg) ----------------
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, F1H, linewidth=0.9, color="blue", label="$F_{1H}(\\theta)$ - однострижнева, H")
ax.plot(steps, F1E, linewidth=0.9, color="red", label="$F_{1E}(\\theta)$ - однострижнева, E")
ax.plot(steps, F2H, linewidth=0.9, color="c", linestyle="--", label="$F_{2H}(\\theta)$ - двострижнева, H")
ax.plot(steps, F2E, linewidth=0.9, color="orange", linestyle="--", label="$F_{2E}(\\theta)$ - двострижнева, E")

ax.plot([0, max(SGP1H, SGP2H) / 2], [0.707, 0.707], 'k--', linewidth=0.6)
for sgp, th, col in [(SGP1H, th1H, 'blue'), (SGP1E, th1E, 'red'),
                     (SGP2H, th2H, 'c'), (SGP2E, th2E, 'orange')]:
    ax.plot(th, 0.707, 'o', markersize=4, color=col)
    ax.plot([th, th], [0, 0.707], linestyle=':', linewidth=0.5, color=col)
    ax.annotate(f'{sgp:.1f}$^\\circ$', xy=(th, 0.707),
                xytext=(th + 1, 0.76), fontsize=7, color=col)

mins_x, mins_y, _, _ = extrema(steps, F1H)
ax.plot(mins_x, mins_y, "o", markersize=4, color="black", label="$\\theta_{min}$")
for mx, my in zip(mins_x, mins_y):
    ax.vlines(x=mx, ymin=0, ymax=my, colors='green', linestyles='dashed', linewidth=0.5)
    ax.annotate(f'({mx:.1f}$^\\circ$)', xy=(mx, my), xytext=(mx - 3, my + 0.05), fontsize=6)

ax.set_xlabel('$\\theta$, grad', fontsize=10)
ax.set_ylabel('|F(teta)|', fontsize=10)
plt.xticks(numpy.arange(0, 95, 5), fontsize=8)
plt.yticks(numpy.arange(0, 1.1, 0.1), fontsize=8)
plt.ylim(-0.01, 1.05)
plt.xlim(0, 90)
plt.title(f"ДС ДСА: lambda = {lam} cm, h = {h} cm, l = {l} cm, xi = {xi:.3f}", fontsize=11)
plt.legend(loc="upper right", fontsize=8)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')
fig.savefig("ДС ДСА у площинах E та H (лінійні).jpg", dpi=600)
plt.show()

# ---------------- Радіальні (полярні) ДС ----------------
fig2 = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
ax = fig2.add_subplot(111, projection='polar')
theta_rad = numpy.radians(steps)

ax.plot(theta_rad, F1H, linewidth=0.9, color="blue", label="$F_{1H}(\\theta)$")
ax.plot(theta_rad, F1E, linewidth=0.9, color="red", label="$F_{1E}(\\theta)$")
ax.plot(theta_rad, F2H, linewidth=0.9, color="c", linestyle="--", label="$F_{2H}(\\theta)$")
ax.plot(theta_rad, F2E, linewidth=0.9, color="orange", linestyle="--", label="$F_{2E}(\\theta)$")

ax.plot(-theta_rad, F1H, linewidth=0.9, color="blue", alpha=0.6)
ax.plot(-theta_rad, F1E, linewidth=0.9, color="red", alpha=0.6)
ax.plot(-theta_rad, F2H, linewidth=0.9, color="c", linestyle="--", alpha=0.6)
ax.plot(-theta_rad, F2E, linewidth=0.9, color="orange", linestyle="--", alpha=0.6)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 181, 10))
ax.set_title("Радіальні ДС ДСА в площинах E та H", fontsize=12)
ax.set_rlabel_position(90)
ax.grid(True, linestyle='--', linewidth=0.3)
for label in ax.get_xticklabels():
    label.set_fontsize(7)
ax.legend(loc="lower left", fontsize=8)
fig2.savefig("ДС ДСА радіальні (E та H).jpg", dpi=600)
plt.show()