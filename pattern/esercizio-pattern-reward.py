import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc as sp #scipy won't import misc :\
import math

def f(x): #la funzione di cui vogliamo trovare il minimo
    return x*x+x
def f_primo(x): #la sua fedele derivata
    return sp.derivative(f, x)

space_start = -1.
space_end = 1.
cicli = 10
x_min_at_start = 0.
eta = 0.1 #un parametro

def migliora():
    x_min=x_min_at_start
    Xdata = np.linspace(space_start, space_end, 10000) #lo spazio su cui cercheremo il minimo
    Ydata = f(Xdata)
    serie = np.zeros(cicli) #ci servirà per tenere in memoria i valori del nostro candidato minimo
    for i in range(cicli):
        print("iterazione ", i)
        print("x_min = ", x_min, '\n')
        serie[i]=x_min
        x_min = x_min - eta * f_primo(x_min) #alla ricerca del minimo
    fig, ax = plt.subplots()
    ax.plot( Xdata , Ydata, ",b" )
    ax.plot(serie, f(serie), ".r")
        
while True:
    comando = input()
    if comando=="space start":
        space_start=float(input("entert space start:\n"))
    elif comando == "space end":
        space_end = float( input("enter space end:\n") )
    elif comando == "eta":
        eta = float( input("enter the reward parameter:\n") )
    elif comando == "cicli":
        cicli = int( input("enter number of iterations:\n") )
    elif comando == "starting point":
        x_min_at_start = int( input("enter the starting point\n") )
    elif comando == "start":
        migliora()
    elif comando == "quit":
        break
    else:
        continue

    
