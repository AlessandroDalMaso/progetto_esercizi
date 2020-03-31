import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc as sp #scipy won't import misc :\
import math as ma

ascisse = np.linspace(2,12,1000)
Xdisturbate = ascisse
def funza (x):
    return ma.sin(0.9*x)/(ma.sin(x)+1.1)
    
vunza = np.vectorize(funza)

ordinate = vunza(ascisse)

fig, ax = plt.subplots()
ax.plot(ascisse, ordinate, "k")
