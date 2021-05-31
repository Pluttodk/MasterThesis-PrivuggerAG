import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def temp(guess):
    return guess == 124

plt.style.use('seaborn-darkgrid')

def secret_entropy(lower, spread):
    secret = st.randint(lower,lower+spread).rvs(10_000)
    o = np.array([temp(si) for si in secret])
    s, counts = np.unique(o, return_counts=True)
    counts = counts/len(o)
    return st.entropy(counts, base=2)

fig = plt.figure()
ax = fig.gca(projection='3d')

lower = np.arange(0,199)
spread = np.arange(1,200)
X, Y = np.meshgrid(lower,spread)

# for l in lower:
#     upper = np.arange(l, 200)
#     y = [secret_entropy(l, u) for u in upper]
Z = np.array([[secret_entropy(X[i][j],Y[i][j]) for j in range(199)] for i in range(199)])
print(Z.shape)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# print(Z.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()