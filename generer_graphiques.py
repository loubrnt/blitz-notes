import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import tikzplotlib  # Nécessaire pour exporter en .tex

print("--- Script de génération des graphiques TCL ---")
print("Assurez-vous d'avoir installé tikzplotlib : pip install tikzplotlib")

# --- Fonctions utilitaires ---

def gaussPDF(x, mu, sigma):
    """Calcule la densité de probabilité d'une loi normale."""
    return stats.norm.pdf(x, mu, sigma)

def binomPDF(n, k, p):
    """Calcule la probabilité de masse d'une loi binomiale."""
    return stats.binom.pmf(k, n, p)

# --- 1. Graphique : Distribution de la moyenne d'un dé ---

print("Génération de 'graphique_clt_de.tex'...")
plt.figure(figsize=(10, 8)) # Tailles ajustées pour correspondre au LaTeX
ax1 = plt.gca()

# n=1 (Uniforme)
n1_x = [1, 2, 3, 4, 5, 6]
n1_y = [1/6] * 6
# La barre a une largeur de 1.0 pour correspondre à 'ybar interval'
ax1.bar(n1_x, n1_y, width=1.0, align='center', edgecolor='black', 
        alpha=0.5, label='$n=1$ (Uniforme)')

# n=2 (Triangulaire)
n2_means = np.array([
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
    4.0, 4.5, 5.0, 5.5, 6.0
])
# Probabilités (sur 36) pour la moyenne
n2_probs = np.array([
    1, 2, 3, 4, 5, 6,
    5, 4, 3, 2, 1
]) / 36.0
# Densité = Probabilité / largeur_bin (largeur de 0.5)
n2_density = n2_probs / 0.5
ax1.bar(n2_means, n2_density, width=0.5, alpha=0.7, 
        label='$n=2$ (Triangulaire)', color='red')

# n=30 (Normale)
mu_die = 3.5
var_die = 35/12 # Variance d'un dé à 6 faces
n_30 = 30
mu_n30 = mu_die
se_n30 = np.sqrt(var_die / n_30) # sqrt( (35/12) / 30 ) = sqrt(7/72)

x_n30 = np.linspace(2, 5, 200)
y_n30 = gaussPDF(x_n30, mu_n30, se_n30)
ax1.plot(x_n30, y_n30, 'g-', lw=2.5, label='$n=30$ (Approximation TCL)')

ax1.set_title("Distribution de la moyenne $\\bar{X}_n$ d'un dé (Uniforme)")
ax1.set_xlabel("Moyenne $\\bar{X}_n$")
ax1.set_ylabel("Densité de probabilité")
ax1.set_xlim(0.5, 6.5)
ax1.grid(True, linestyle='--')
ax1.legend(loc='upper left')

# Sauvegarde en .tex en utilisant les tailles d'origine
tikzplotlib.save("graphique_clt_de.tex",
                 axis_height='10cm',
                 axis_width='12cm',
                 strict=True) # strict=True pour une meilleure compatibilité
plt.close()
print("... Terminé.")

# --- 2. Graphique : LLN vs TCL (Zoom) ---

print("Génération de 'graphique_clt_zoom.tex'...")
# Note : tikzplotlib gère les subplots
fig, (ax_lln, ax_tcl) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: LLN
mu = 3.5
var = 35/12
n_vals = [5, 20, 80]
colors = ['red', 'blue', 'green']
x_lln = np.linspace(2, 5, 200)

for n, color in zip(n_vals, colors):
    se = np.sqrt(var / n)
    y = gaussPDF(x_lln, mu, se)
    ax_lln.plot(x_lln, y, color=color, lw=2, label=f'$n={n}$')

ax_lln.set_title("Loi des Grands Nombres (LLN)")
ax_lln.set_xlabel("Valeur de la moyenne $\\bar{X}_n$")
ax_lln.set_ylabel("Densité")
ax_lln.set_xlim(2, 5)
ax_lln.set_ylim(bottom=0)
ax_lln.grid(True, linestyle='--')
ax_lln.legend(loc='upper left')
# Ajout du texte comme dans l'original PGFPlots
ax_lln.text(3.5, 1.5, # Coordonnées ajustées manuellement pour un bon placement
            "Distribution de $\\bar{X}_n$\n(converge vers un pic à $\\mu=3.5$)", 
            ha='center', va='center', fontsize=10)

# Panel 2: TCL
x_tcl = np.linspace(-3.5, 3.5, 200)
y_tcl = gaussPDF(x_tcl, 0, 1)
ax_tcl.plot(x_tcl, y_tcl, 'b-', lw=2)

ax_tcl.set_title("Théorème Central Limite (TCL)")
ax_tcl.set_xlabel("Valeur standardisée $Z_n$")
ax_tcl.set_ylabel("Densité")
ax_tcl.set_xlim(-3.5, 3.5)
ax_tcl.set_ylim(bottom=0, top=0.45) # Ajusté pour correspondre à l'original
ax_tcl.grid(True, linestyle='--')
# Ajout du texte
ax_tcl.text(0, 0.25, # Coordonnées ajustées manuellement
            "Distribution de $Z_n = \\frac{\\bar{X}_n - \\mu}{\\sigma / \\sqrt{n}}$\n(converge vers $N(0,1)$)", 
            ha='center', va='center', fontsize=10)

plt.tight_layout()

# Sauvegarde en .tex
tikzplotlib.save("graphique_clt_zoom.tex",
                 figure=fig,
                 axis_height='7cm',
                 axis_width='7.5cm',
                 strict=True)
plt.close()
print("... Terminé.")

# --- 3. Graphique : Approximation Binomiale ---

print("Génération de 'graphique_clt_binomial.tex'...")
plt.figure(figsize=(11, 7)) # Ajusté
ax3 = plt.gca()

n_bin = 100
p_bin = 0.5
mu_bin = n_bin * p_bin
std_bin = np.sqrt(n_bin * p_bin * (1 - p_bin))

# 1. Binomial PMF (barres)
k_values = np.arange(35, 71)
pmf_values = binomPDF(n_bin, k_values, p_bin)
ax3.bar(k_values, pmf_values, width=1.0, align='center', 
        color='gray', alpha=0.6, label='Loi Binomiale $B(100, 0.5)$')

# 2. Normal PDF (ligne)
x_norm = np.linspace(35, 70, 200)
pdf_values = gaussPDF(x_norm, mu_bin, std_bin)
ax3.plot(x_norm, pdf_values, 'b-', lw=2, label='Approximation $N(50, 5)$')

# 3. Zone hachurée (P > 60.5)
k_fill = 60.5
x_fill = np.linspace(k_fill, 70, 100)
y_fill = gaussPDF(x_fill, mu_bin, std_bin)
prob_shaded = 1 - stats.norm.cdf(k_fill, mu_bin, std_bin)
ax3.fill_between(x_fill, y_fill, color='red', alpha=0.5, 
                 label=f'$P(S > {k_fill}) \\approx {prob_shaded:.4f}$') # Légende mise à jour

# 4. Ligne pointillée
y_line_at_k = gaussPDF(k_fill, mu_bin, std_bin)
ax3.vlines(k_fill, 0, y_line_at_k, color='red', linestyle='--', lw=2)

# 5. Annotation (légèrement différente de PGF 'pin')
z_score = (k_fill - mu_bin) / std_bin
ax3.annotate(f'$k={k_fill}$\n$Z = \\frac{{{k_fill} - {mu_bin}}}{{{std_bin:.0f}}} = {z_score:.1f}$',
             xy=(k_fill, y_line_at_k),
             xytext=(k_fill + 2, y_line_at_k + 0.015), # Placement du texte
             arrowprops=dict(facecolor='black', arrowstyle='-|>'),
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.5))

ax3.set_title("Approximation de $B(100, 0.5)$ par $N(50, 5)$")
ax3.set_xlabel("Nombre de \"Face\" $S_n$")
ax3.set_ylabel("Probabilité / Densité")
ax3.set_xlim(35, 70)
ax3.set_ylim(bottom=0)
ax3.grid(True, linestyle='--')
ax3.legend(loc='upper left')

# Sauvegarde en .tex
tikzplotlib.save("graphique_clt_binomial.tex",
                 axis_height='8cm',
                 axis_width='13cm',
                 strict=True)
plt.close()
print("... Terminé.")
print("\n--- Tous les fichiers .tex ont été générés. ---")