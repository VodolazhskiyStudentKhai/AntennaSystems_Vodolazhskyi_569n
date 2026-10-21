import numpy as np
import matplotlib.pyplot as plt

# ---------------- Початкові дані (варіант) ----------------
lambd = 2.9 * 10**-2         # довжина хвилі, м
a_p = 14.0 * 10**-2           # розмір розкриву рупора в площині H, м
b_p = 14.0 * 10**-2           # розмір розкриву рупора в площині E, м (квадратний розкрив)


print(f"Довжина хвилі λ = {lambd * 100} см")
print(f"Розмір розкриву рупора a_p x b_p = {a_p * 100} x {b_p * 100} см")
print(f"a_p/λ = {a_p / lambd:.2f};  b_p/λ = {b_p / lambd:.2f}")

# Приблизна ширина головної пелюстки за формулами (3.43), (3.44)
print(f"2θ_0.5E ≈ 53°·λ/b_p = {53 * lambd / b_p:.2f}°")
print(f"2θ_0.5H ≈ 80°·λ/a_p = {80 * lambd / a_p:.2f}°")

# КСД оптимального пірамідального рупора (3.45), (3.46): v_A = 0.52
vA = 0.52
Dm = (4 * np.pi / lambd ** 2) * a_p * b_p * vA
print(f"Апертурний КВП оптимального рупора v_A = {vA}")
print(f"КСД D_m = 4π/λ² · S_p · v_A = {Dm:.1f}  ({10 * np.log10(Dm):.2f} дБ)")


F1E = [1]
FCH = [1]
FCE = [1]
FH = [1]
FE = [1]
steps = [0]

SGPH = 0
SGPE = 0
fSH = 0
fSE = 0

max_x_FH = []
max_y_FH = []
max_x_FE = []
max_y_FE = []
min_x_FH = []
min_y_FH = []
min_x_FE = []
min_y_FE = []

def getF1(theta):
    return (1 + np.cos(theta))/2

def getFCH(theta):
    return ((np.cos(np.pi * a_p/lambd * np.sin(theta)))/(1-(2*a_p/lambd*np.sin(theta))**2))

def getFCE(theta):
    return (np.sin(np.pi * b_p/lambd * np.sin(theta)))/(np.pi * b_p/lambd * np.sin(theta))

def getFE(theta):
    return getFCE(theta) * getF1(theta)

for teta in np.arange(0.01, np.pi / 2, 0.00001):
    s = np.sin(teta)

    mn1 = getF1(teta)    # F1(θ)
    mn2 = getFCH(teta)   # F_CH(θ)
    mn3 = getFCE(teta)   # F_CE(θ)

    mH = abs(mn2 * mn1)   # F_H(θ)
    mE = abs(mn3 * mn1)   # F_E(θ)

    F1E += [mn1]
    FCH += [abs(mn2)]
    FCE += [abs(mn3)]
    FH += [mH]
    FE += [mE]

    if 0.707 < mH < 0.708:
        SGPH = 2 * np.degrees(teta)
        fSH = mH

    if 0.707 < mE < 0.708:
        SGPE = 2 * np.degrees(teta)
        fSE = mE

    steps += [np.degrees(teta)]

print()
print("Ширина головної пелюстки в площині H (за розрахованою ДС) = " + str(round(SGPH, 2)) + '\u00b0')
print("Ширина головної пелюстки в площині E (за розрахованою ДС) = " + str(round(SGPE, 2)) + '\u00b0')

# ---------------- Пошук максимумів і мінімумів ДС ----------------
for i in range(1, len(FH) - 1):
    if FH[i] > FH[i - 1] and FH[i] > FH[i + 1]:
        max_x_FH.append(steps[i])
        max_y_FH.append(FH[i])
    if FE[i] > FE[i - 1] and FE[i] > FE[i + 1]:
        max_x_FE.append(steps[i])
        max_y_FE.append(FE[i])

for i in range(1, len(FH) - 1):
    if FH[i] < FH[i - 1] and FH[i] < FH[i + 1]:
        min_x_FH.append(steps[i])
        min_y_FH.append(FH[i])
    if FE[i] < FE[i - 1] and FE[i] < FE[i + 1]:
        min_x_FE.append(steps[i])
        min_y_FE.append(FE[i])

print()
print("Площина E (для F_CE): нулі θ0n = arcsin(n·λ/b_p), максимуми θmn = arcsin((2n+1)/2·λ/b_p)")
n = 1
while n * lambd / b_p <= 1:
    print(f"  нуль θ0{n} = {np.degrees(np.arcsin(n * lambd / b_p)):.2f}\u00b0")
    n += 1
n = 1
while (2 * n + 1) / 2 * lambd / b_p <= 1:
    thm = np.degrees(np.arcsin((2 * n + 1) / 2 * lambd / b_p))
    print(f"  максимум θm{n} = {thm:.2f}\u00b0,  РБП |F_CE| = 2/((2n+1)π) = {2 / ((2 * n + 1) * np.pi):.3f}")
    n += 1

print()
print("Площина H (для F_CH): нулі θ0n = arcsin((2n+1)/2·λ/a_p), максимуми θmn = arcsin((n+1)·λ/a_p)")
n = 1
while (2 * n + 1) / 2 * lambd / a_p <= 1:
    print(f"  нуль θ0{n} = {np.degrees(np.arcsin((2 * n + 1) / 2 * lambd / a_p)):.2f}\u00b0")
    n += 1
n = 1
while (n + 1) * lambd / a_p <= 1:
    thm = np.degrees(np.arcsin((n + 1) * lambd / a_p))
    print(f"  максимум θm{n} = {thm:.2f}\u00b0,  РБП |F_CH| = 1/|1-[2(n+1)]²| = {1 / abs(1 - (2 * (n + 1)) ** 2):.3f}")
    n += 1

# ---------------- Загальна таблиця ----------------
len_value = [len(max_x_FH), len(max_x_FE), len(min_x_FH), len(min_x_FE)]
print()
print("Табл. 1 - Аналіз ДС пірамідального рупора у площинах Н та Е")
print("----------------------------------------------------")
print("| №  |θminH |θminE |θmaxH | FH(θ)|θmaxE | FE(θ)|")
for i in range(0, max(len_value)):
    v2 = f"{min_x_FH[i]:.2f}" if i < len_value[2] else "  -  "
    v3 = f"{min_x_FE[i]:.2f}" if i < len_value[3] else "  -  "
    v4 = f"{max_x_FH[i]:.2f}" if i < len_value[0] else "  -  "
    h4 = f"{max_y_FH[i]:.3f}" if i < len_value[0] else "  -  "
    v5 = f"{max_x_FE[i]:.2f}" if i < len_value[1] else "  -  "
    h5 = f"{max_y_FE[i]:.3f}" if i < len_value[1] else "  -  "
    print(f"|{i + 1:2}|{v2:>6}|{v3:>6}|{v4:>6}|{h4:>6}|{v5:>6}|{h5:>6}|")
print("----------------------------------------------------")

# ---------------- Графік ДС у площині Н ----------------
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, F1E, linewidth=0.7, label="$ F_{1}(θ) $")
ax.plot(steps, FCH, linewidth=0.7, label="$ F_{CH}(θ) $")
ax.plot(steps, FH, linewidth=0.7, label="$ F_{H}(θ) $")
ax.plot(SGPH / 2, fSH, 'ro', markersize=4, label="Рівень половинної потужності в площині H")

plt.annotate(f'({fSH:.3f}, {SGPH / 2:.2f}\u00b0)',
             xy=(SGPH / 2, fSH),
             xytext=((SGPH / 2) + 4, fSH + 0.08),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

ax.plot([0, SGPH / 2], [fSH, fSH], 'r--', linewidth=0.5)
ax.plot([SGPH / 2, SGPH / 2], [0, fSH], 'r--', linewidth=0.5)

ax.plot(max_x_FH, max_y_FH, "o", markersize=4, color="black", label="$ θ_{max} $ FH")
ax.plot(min_x_FH, min_y_FH, "o", markersize=4, color="blue", label="$ θ_{min} $ FH")

ax.set_xlabel('θ' + '\u00b0', fontsize=10)
ax.set_ylabel('|F1(θ)|, |FCH(θ)|, |FH(θ)|', fontsize=10)
plt.xticks(np.arange(0, 100, 2), fontsize=7)
plt.yticks(np.arange(0, 1.2, 0.1), fontsize=7)
plt.ylim(-0.01, 1.01)
plt.xlim(0, 90.5)

for i in range(len(max_x_FH)):
    plt.vlines(x=max_x_FH[i], ymin=0, ymax=max_y_FH[i], colors='green', linestyles='dashed', linewidth=0.5)
    plt.hlines(y=max_y_FH[i], xmin=0, xmax=max_x_FH[i], colors='green', linestyles='dashed', linewidth=0.5)
    plt.annotate(f'({max_y_FH[i]:.3f}, {max_x_FH[i]:.2f}\u00b0)',
                 xy=(max_x_FH[i], max_y_FH[i]),
                 xytext=(max_x_FH[i] + 4, max_y_FH[i] + 0.08),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

for i in range(len(min_x_FH)):
    plt.annotate(f'({min_x_FH[i]:.2f}\u00b0)',
                 xy=(min_x_FH[i], min_y_FH[i]),
                 xytext=(min_x_FH[i] - 13, min_y_FH[i] + 0.06),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

plt.legend(loc="upper right", fontsize=7)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')
fig.savefig("ДС рупора у площині Н, лр3.jpg", dpi=600)
plt.show()

# ---------------- Графік ДС у площині Е ----------------
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, F1E, linewidth=0.7, label="$ F_{1}(θ) $")
ax.plot(steps, FCE, linewidth=0.7, label="$ F_{CE}(θ) $")
ax.plot(steps, FE, linewidth=0.7, label="$ F_{E}(θ) $")
ax.plot(SGPE / 2, fSE, 'go', markersize=4, label="Рівень половинної потужності в площині E")

plt.annotate(f'({fSE:.3f}, {SGPE / 2:.2f}\u00b0)',
             xy=(SGPE / 2, fSE),
             xytext=((SGPE / 2) + 4, fSE + 0.08),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

ax.plot([0, SGPE / 2], [fSE, fSE], 'g--', linewidth=0.5)
ax.plot([SGPE / 2, SGPE / 2], [0, fSE], 'g--', linewidth=0.5)

ax.plot(max_x_FE, max_y_FE, "o", markersize=4, color="grey", label="$ θ_{max} $ FE")
ax.plot(min_x_FE, min_y_FE, "o", markersize=4, color="blue", label="$ θ_{min} $ FE")

ax.set_xlabel('θ' + '\u00b0', fontsize=10)
ax.set_ylabel('|F1(θ)|, |FCE(θ)|, |FE(θ)|', fontsize=10)
plt.xticks(np.arange(0, 100, 2), fontsize=7)
plt.yticks(np.arange(0, 1.2, 0.1), fontsize=7)
plt.ylim(-0.01, 1.01)
plt.xlim(0, 90.5)

for i in range(len(max_x_FE)):
    plt.vlines(x=max_x_FE[i], ymin=0, ymax=max_y_FE[i], colors='green', linestyles='dashed', linewidth=0.5)
    plt.hlines(y=max_y_FE[i], xmin=0, xmax=max_x_FE[i], colors='green', linestyles='dashed', linewidth=0.5)
    plt.annotate(f'({max_y_FE[i]:.3f}, {max_x_FE[i]:.2f}\u00b0)',
                 xy=(max_x_FE[i], max_y_FE[i]),
                 xytext=(max_x_FE[i] + 4, max_y_FE[i] + 0.08),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

for i in range(len(min_x_FE)):
    plt.annotate(f'({min_x_FE[i]:.2f}\u00b0)',
                 xy=(min_x_FE[i], min_y_FE[i]),
                 xytext=(min_x_FE[i] - 13, min_y_FE[i] + 0.06),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

plt.legend(loc="upper right", fontsize=7)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')
fig.savefig("ДС рупора у площині Е, лр3.jpg", dpi=600)
plt.show()

# ---------------- Радіальна (полярна) ДС ----------------
fig2 = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
ax = fig2.add_subplot(111, projection='polar')
theta_rad = np.radians(steps)
ax.plot(theta_rad, FH, linewidth=0.8, color="blue", label="$F_H(θ)$")
ax.plot(theta_rad, FE, linewidth=0.8, color="red", label="$F_E(θ)$")
theta_rad_mirror = -theta_rad
ax.plot(theta_rad_mirror, FH, linewidth=0.8, color="blue", alpha=0.8)
ax.plot(theta_rad_mirror, FE, linewidth=0.8, color="red", alpha=0.8)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 181, 10))
ax.set_title("Радіальна діаграма спрямованості пірамідального рупора", fontsize=12)
ax.set_rlabel_position(90)
ax.grid(True, linestyle='--', linewidth=0.3)
for label in ax.get_xticklabels():
    label.set_fontsize(7)
ax.legend(loc="lower left", fontsize=8)
fig2.savefig("ДС рупора радіальна, лр3.jpg", dpi=600)
plt.show()