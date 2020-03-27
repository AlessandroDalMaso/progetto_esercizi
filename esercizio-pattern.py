import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def f(x): #la funzione di cui vogliamo trovare il minimo
    return x*x+x
def f1(x): #la sua fedele derivata
    return 2*x+1

Xdata = np.linspace(-1, 1, 1000)
Ydata = f(Xdata)

x_min = 0.
x_min = float(input("il valore iniziale del minimo: "))

eta = 0. #un parametro
eta = float(input("il valore di eta: "))

cicli = 1
cicli = int(input("quante interazioni faremo: "))
serie = np.zeros(cicli) #ci servirà per tenere in memoria i valori del nostro candidato minimo

for i in range(cicli):
    print("iterazione ", i)
    print("x_min = ", x_min, '\n')
    serie[i]=x_min
    x_min = x_min-eta*f1(x_min) #alla ricerca del minimo
    
fig, ax = plt.subplots()
ax.plot( Xdata , Ydata, ",b" )
ax.plot(serie, f(serie), "or")
    
