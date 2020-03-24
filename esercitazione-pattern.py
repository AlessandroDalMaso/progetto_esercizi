#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:00:12 2019

@author: silvia
"""

import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import pandas as pd
#%%
def mytable(mypolyfit):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(18,4))
    data = list(mypolyfit.values.astype(str))
    columns = tuple(mypolyfit.columns.astype(str))
    rows = np.arange(1,len(mypolyfit)+1).astype(str)
    
    #values = np.arange(len(mypolyfit))
    #value_increment = 1
    
    # Get some pastel shades for the colors
    ccolors = plt.cm.BuPu(np.linspace(0, 0.5, len(columns)))
    rcolors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    #n_rows = len(rows)
    
   # index = np.arange(len(rows)) + 1
    #bar_width = 0.4
    
    
    # Plot bars and create text labels for the table
    cell_text = data
    # Reverse colors and text labels to display the last value at the top.
    rcolors = rcolors[::-1]
    ccolors = ccolors[::-1]
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          #colWidths=[1/15] * 15,
                          rowLabels=rows,
                          colColours=ccolors,
                          rowColours=rcolors,
                          colLabels=columns,
                          loc='center')
    the_table.scale(1, 1)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.title('Polynomial degree',fontsize=20)
    plt.axis('off')
    plt.tight_layout()    
    plt.show()
    

def datagen(x,eps):
    return np.sin(2*np.pi*x) + st.norm.rvs(size=len(x),loc=0,scale=eps)

   
def RMS(y,yexp):
    s = np.sum((yexp-y)**2)
    n=len(y)
    return np.sqrt(s)/n
       
#%%
#FITTING DATA WITH POLYNOMIALS
#POLY MINIMIZE THE SQUARED ERROR


x=np.linspace(0,1,1000)
plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='true func')


X=np.linspace(0,1,10)
eps=0.2
Y=datagen(X,eps) 
plt.scatter(X,Y,s=140,label='mydata',facecolors='w',edgecolor='b')


m=8
p=np.polyfit(X, Y, m)
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'r--',label='poly '+str(m))
plt.legend()
plt.show()
#%%

for i,m in enumerate([0,1,3,9]):
    plt.subplot(221+i)
    p=np.polyfit(X, Y, m)
    yexp=np.poly1d(p)
    plt.scatter(X,Y,s=40,label='mydata',facecolors='w',edgecolor='b')
    plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='true func')
    plt.plot(x,yexp(x),'r--',label='poly '+str(m))
    plt.title('M = '+str(m))

plt.tight_layout()
plt.show()

#%%
#STESSA COSA CON DIVERSI VALORI DI M

x=np.linspace(0,1,1000)
plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='$\sin(2\pi x)$')

X1=np.linspace(0,1,10)
train=datagen(X1,eps)
plt.scatter(X1,train,s=140,label='train',facecolors='w',edgecolor='b')
m=8
p=np.polyfit(X1, train, m)
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'r--',label='poly M='+str(m))


X=np.linspace(0,1,7)
eps=0.25
test=datagen(X,eps)
plt.scatter(X,test,s=140,label='test',facecolors='w',edgecolor='r')

plt.legend()
plt.show()

testRMS=np.empty(15)
trainRMS=np.empty(15)
pp=np.zeros((15,15))
M=np.arange(15)
for m in M:
    p=np.polyfit(X1, train, m)
    #print(m,p)
    for i in np.arange(m+1):
            pp[i,m]=p[i]
    #print(p,pp[:,m])
        
    fexp=np.poly1d(p)
    
    trainexp=fexp(X1)
    trainRMS[m]=RMS(train,trainexp)
    
    testexp=fexp(X)
    testRMS[m]=RMS(test,testexp)

#OVERFITTING: GOOD PERFORMANCE ON TRAIN BAD ON TEST     
plt.scatter(M,trainRMS,s=140,facecolors='w',edgecolor='b')  
plt.scatter(M,testRMS,s=140,facecolors='w',edgecolor='r')
plt.plot(M,trainRMS,'ob-',label='train')  
plt.plot(M,testRMS,'or-',label='test')
plt.legend()
plt.show()

# EFFECT OF OVER FITTING ON FIT COEFFICIENTS
mypolyfit=pd.DataFrame(pp).apply(lambda x: np.round(x,3))
mypolyfit.to_csv('./mypolyfit.csv')
mytable(mypolyfit)

#%% 
#FIRST WAY TO AVOID CURVE FITTING
# LARGER SAMPLE SIZE ALLOW FOR MORE PARAMETERS
for i,size in enumerate([10,100]):
    plt.subplot(121+i)    
    x=np.linspace(0,1,1000)
    plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='true func')
    X1=np.linspace(0,1,size)
    train=datagen(X1,eps)
    plt.scatter(X1,train,s=100,label='train',facecolors='w',edgecolor='b')
    m=8
    p=np.polyfit(X1, train, m)
    yexp=np.poly1d(p)
    plt.plot(x,yexp(x),'r--',label='poly '+str(m))
    plt.legend()
    
plt.show()

#%%
#SECOND WAY TO AVOID CURVE FITTING
#FEW DATA POINTS? -> PENALIZED REGRESSION
#KEEP THE COEFFICIENTS CLOSE TO ZERO


def my_poly(x,args=()):
    f=np.poly1d(args)
    return f(x)

x=np.arange(10)
args=tuple(np.array([1,1,1]))
y=my_poly(x,args)

#1)standard minimization
def sq_err(x,y,f,p=()):
    from scipy.optimize import minimize
    def cost(args):
        return np.sum((y-f(x,args))**2)*0.5
    return minimize(cost,p)
    

res = sq_err(x,y,my_poly,args)

#2)penalized minimization
#1)standard minimization
def sq_err_ridge(x,y,f,lamb=0, p=()):
    from scipy.optimize import minimize
    def cost(args):
        return np.sum((y-f(x,args))**2)*0.5 + lamb*np.sum(args**2)*0.5
    return minimize(cost,p)
    

res = sq_err_ridge(x,y,my_poly,lamb=1,p=args).x


#%%
#LET'S APPLY TO THECASE BEFORE
#genero i dati
X=np.linspace(0,1,10)
eps=0.2
Y=datagen(X,eps) 
#genero x per plot dei fit
x=np.linspace(0,1,1000)

for i,m in enumerate([0,3,9,15]):
    plt.subplot(221+i)
    plt.scatter(X,Y,s=40,label='mydata',facecolors='w',edgecolor='b')
    plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='true func')
    
    p=sq_err(X,Y,my_poly,np.zeros(m+1)).x
    yexp=np.poly1d(p)
    plt.plot(x,yexp(x),'r--',label='sqerr '+str(m))
    
    
    p=sq_err_ridge(X,Y,my_poly,lamb=0.001,p=np.zeros(m+1)).x
    yexp=np.poly1d(p)
    plt.plot(x,yexp(x),'y--',label='ridge '+str(m))
    
    plt.title('M = '+str(m))
    
plt.legend()

plt.tight_layout()
plt.show()

#%%
#COME VARIA LA PERFOMANCE SUI TEST CON LA PENALIZZAZIONE?
lamb=1e-3

x=np.linspace(0,1,1000)
plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='$\sin(2\pi x)$')

#TRAIN
X1=np.linspace(0,1,10)
train=datagen(X1,eps)
plt.scatter(X1,train,s=140,label='train',facecolors='w',edgecolor='b')
m=10
p=np.polyfit(X1, train, m)
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'r--',label='poly M='+str(m))

    
p=sq_err_ridge(X1,train,my_poly,lamb=0.001,p=np.zeros(m+1)).x
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'y--',label='ridge '+str(m))

#TEST
X=np.linspace(0,1,7)
eps=0.25
test=datagen(X,eps)
plt.scatter(X,test,s=140,label='test',facecolors='w',edgecolor='r')

plt.legend()
plt.show()

testRMS=np.empty(15)
trainRMS=np.empty(15)
pp=np.zeros((15,15))
M=np.arange(15)
for m in M:
    p=sq_err_ridge(X1,train,my_poly,lamb=lamb,p=np.zeros(m+1)).x
    
    for i in np.arange(m+1):
            pp[i,m]=p[i]
    #print(p,pp[:,m])
    fexp=np.poly1d(p)
    
    trainexp=fexp(X1)
    trainRMS[m]=RMS(train,trainexp)
    
    testexp=fexp(X)
    testRMS[m]=RMS(test,testexp)

#OVERFITTING: GOOD PERFORMANCE ON TRAIN BAD ON TEST     
plt.scatter(M,trainRMS,s=140,facecolors='w',edgecolor='b')  
plt.scatter(M,testRMS,s=140,facecolors='w',edgecolor='r')
plt.plot(M,trainRMS,'ob-',label='train')  
plt.plot(M,testRMS,'or-',label='test')
plt.legend()
plt.show()

# EFFECT OF OVER FITTING ON FIT COEFFICIENTS
mypolyfit=pd.DataFrame(pp).apply(lambda x: np.round(x,3))
mypolyfit.to_csv('./mypolyfit.csv')
mytable(mypolyfit)

#%%
#WHAT IS THE IMPACT OF LAMBDA

x=np.linspace(0,1,1000)
plt.plot(x,np.sin(2*np.pi*x),'g',linewidth=2,label='$\sin(2\pi x)$')

#TRAIN
X1=np.linspace(0,1,10)
train=datagen(X1,eps)
plt.scatter(X1,train,s=140,label='train',facecolors='w',edgecolor='b')
m=10
p=np.polyfit(X1, train, m)
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'r--',label='poly M='+str(m))    
lamb=0.001
p=sq_err_ridge(X1,train,my_poly,lamb=lamb,p=np.zeros(m+1)).x
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'y--',label='ridge $\lambda=$'+str(lamb))
    

#TEST
X=np.linspace(0,1,9)
eps=0.25
test=datagen(X,eps)
plt.scatter(X,test,s=140,label='test',facecolors='w',edgecolor='r')

plt.legend()
plt.show()

lambda_=np.arange(5,25)
testRMS=np.empty(len(lambda_))
trainRMS=np.empty(len(lambda_))
pp=np.zeros((m+1,len(lambda_)))

for j,l in enumerate(lambda_):
    lamb = np.exp(-l)
    p=sq_err_ridge(X1,train,my_poly,lamb=lamb,p=np.zeros(m+1)).x
    #print(p)
    
    for i in np.arange(m+1):
            pp[i,j]=p[i]
    #print(p,pp[:,m])
    fexp=np.poly1d(p)
    
    trainexp=fexp(X1)
    trainRMS[j]=RMS(train,trainexp)
    
    testexp=fexp(X)
    testRMS[j]=RMS(test,testexp)

#OVERFITTING: GOOD PERFORMANCE ON TRAIN BAD ON TEST     
plt.scatter(-lambda_[::-1],trainRMS[::-1],s=140,facecolors='w',edgecolor='b')  
plt.scatter(-lambda_[::-1],testRMS[::-1],s=140,facecolors='w',edgecolor='r')
plt.plot(-lambda_[::-1],trainRMS[::-1],'ob-',label='train')  
plt.plot(-lambda_[::-1],testRMS[::-1],'or-',label='test')
plt.legend()
plt.show()

# EFFECT OF OVER FITTING ON FIT COEFFICIENTS
mypolyfit=pd.DataFrame(pp).apply(lambda x: np.round(x,3))
mypolyfit.columns=lambda_
mypolyfit=mypolyfit[mypolyfit.columns[::-1]]
mypolyfit.columns=-mypolyfit.columns
mypolyfit.to_csv('./mypolyfit.csv')
mytable(mypolyfit)

#%%
#CALCULATE LIKELIHOOD OF DATA

#1) GENERATE DATA:

#I) GENERATE GAUSSIAN FROM CLT
n=10000
def gauss_gen(n,mu,sigma):
    sample=np.array([np.mean(10*np.random.random_sample(size=10)) for i in np.arange(n)])
    return (sample -np.mean(sample))*sigma/np.std(sample)+mu

sample= gauss_gen(n,0,1)
plt.hist(sample,bins=100,density=True, alpha=0.3,label='mygen')
popt=st.norm.fit(sample)
x=np.linspace(-4,4,100)
y=st.norm.pdf(x,*popt)
plt.plot(x,y,'k-',linewidth=2)
print(st.shapiro(sample))

#II) GENERATE GAUSS WITH PACKAGE STATS
sample=st.norm.rvs(size=n)
plt.hist(sample,bins=100,density=True,color='y', alpha=0.3,label='stat')
popt=st.norm.fit(sample)
x=np.linspace(-4,4,100)
y=st.norm.pdf(x,*popt)
plt.plot(x,y,'r--',linewidth=2)
print(st.shapiro(sample))

#plt.yscale('log')
#%%
#USIAMO METODO 2
n=5
sample=st.norm.rvs(size=n)
def LH(sample,dist,args=()):
    return np.prod(dist(sample,*args))
    

f=st.norm.pdf
x=np.linspace(-4,4,100)
y=f(x)
plt.plot(x,y,'r',linewidth=3)
plt.fill_between(x,y,color='g',alpha=0.2)
for s in sample:
    plt.plot([s,s],[0,f(s)],'m-',linewidth=3)
plt.plot(sample,f(sample),'og',markersize=8)

like = LH(sample,f)


y=f(x,1,2)
plt.plot(x,y,'orange',linewidth=1)
plt.fill_between(x,y,color='y',alpha=0.2)
for s in sample:
    plt.plot([s,s],[0,f(s,1,2)],'y--',linewidth=2)
plt.plot(sample,f(sample,1,2),'ob',markersize=6)


plt.plot(sample,[0]*len(sample),'ok',markersize=8)
plt.show()


like2 = LH(sample,f,[1,2])

#%%
#SINCE WE ARE DEALING WITH SMALLER AND SMALLER NUMBERS 
#IT CAN BE USEFUL DEAL WITH LOG LIKELIHOOD

def log_LH(sample,dist,args=()):
    return np.sum(np.log(dist(sample,*args)))

like = log_LH(sample,f)

like2 = log_LH(sample,f,[1,2])

#%%
#ASSUMIAMO CHE LE OSSERVABILI SIANO DISTRIBUITE COME UNA GAUSSIANA INTORNO AL VALORE ATTESO
#COME LO VEDIAMO?
#RIPRENDIAMO IL POLINOMIO

lamb=1e-3

x=np.linspace(0,1,1000)
plt.plot(x,np.sin(2*np.pi*x),'g:',linewidth=2,label='$\sin(2\pi x)$')

#TRAIN
X1=np.linspace(0,1,20)
Y1=datagen(X1,eps)
plt.scatter(X1,Y1,s=140,label='train',facecolors='w',edgecolor='b')
m=9
    
p=sq_err_ridge(X1,Y1,my_poly,lamb=0.001,p=np.zeros(m+1)).x
yexp=np.poly1d(p)
plt.plot(x,yexp(x),'r-',label='ridge '+str(m))
plt.show()
#eq1.63 
distance = Y1 -yexp(X1)
plt.hist(distance,bins=5,density=True,color='lightblue',label='residues')
popt=st.norm.fit(distance)
x=np.linspace(-1,1,100)
plt.plot(x, st.norm.pdf(x,*popt),'b-',linewidth=2,label='fit dist')

var=np.var(distance)
plt.plot(x, st.norm.pdf(x,0,np.sqrt(var)),'r--',linewidth=2,label='expected dist')
plt.title('$\mu,\sigma = $' +str(np.round(popt,3)))
plt.legend()
plt.show()

#%%

#PRECISION PARAMETER IS THE INVERSE OF THE VARIANCE DISTRIBUTION
plt.scatter(X1,Y1,s=140,label='train',facecolors='w',edgecolor='lightblue')
m=9

x=np.linspace(0,1,1000)
plt.plot(x,yexp(x),'r-',label='ridge '+str(m))

x0=0.3
xm=yexp(x0)
plt.plot([x0,x0],[-1.5,1.5],'k--')
plt.plot([0,1],[xm,xm],'k--')
plt.scatter(x0,xm,color='g')
x=np.linspace(xm-0.5,xm+0.5,100)
y=st.norm.pdf(x,xm,popt[1])
plt.plot(y/10+x0*0.9,x,'b')
plt.show()

#%%
#COME FACCIAMO A INCLUDERE QUESTA INFORMAZIONE NEL FIT 
#EQ 1.62 VOGLIAMO MASSIMIZZARE LA LOG LIKELIHOOD

def log_LH_GaussNoise(x,y,f,beta=1,args=()):
    yexp=f(x,args)
    return -beta*0.5*np.sum((yexp-y)**2)+len(x)*np.log(beta)*0.5-N*0.5*np.log(2*np.pi)

#Pcondizionata delle osservazioni dato il fato 
#che si assumono distribuite in modo gaussiano intorno al valore atteso
#Le p di ogni osservazione indipendenti quindi la totale è il prodotto delle singole

def min_neglog_LH_GaussNoise(x,y,f,p=[1,()]):
    from scipy.optimize import minimize
    def neg_log_LH_GaussNoise(args):
        beta=args[0]
        w=args[1:]
        yexp=f(x,w)
        N=len(x)
        ll= -beta*0.5*np.sum((yexp-y)**2)+N*np.log(beta)*0.5-N*0.5*np.log(2*np.pi)
        return -ll
    bnds = np.array([(0, None)]+[(None,None)]*len(p[1:])) 
    return minimize(neg_log_LH_GaussNoise,p,method='SLSQP',bounds=bnds)

def f(x,args=[1,2*np.pi]):
    a,b=args[0],args[1]
    return a*np.sin(b*x)

f=my_poly
x=np.linspace(0,1,20)
eps=0.5
y=datagen(x,eps)

#res= min_neglog_LH_GaussNoise(x,y,f,p=[0.4,1,5]).x
res= min_neglog_LH_GaussNoise(x,y,f,p=[0.4,0.1,0.1,0.1,0.01,0.001,0,0,0.1,0.1]).x
print('eps=sqrt(1/beta)=',np.round(np.sqrt(1/res[0]),2), ', p=',np.round(res[1:],3))


plt.scatter(x,y,s=140,label='train',facecolors='w',edgecolor='lightblue')

x=np.linspace(0,1,100)
plt.plot(x,f(x,res[1:]),'r-')

x0=0.3
xm=f(x0,res[1:])
plt.plot([x0,x0],[-1.5,1.5],'k--')
plt.plot([0,1],[xm,xm],'k--')
plt.scatter(x0,xm,color='g')
x=np.linspace(xm-0.5,xm+0.5,100)
y=st.norm.pdf(x,xm,np.sqrt(1/res[0]))
plt.plot(y/10+x0*0.9,x,'b')
plt.show()
#%%
#STEP TOWARD BAYESIAN APPROACH
#FOR EACH OF THEPARAMETER WE HAVE A PRIOR DISTRIBUTION
#considering gaussian prior it is possible to determine the posterior
#maximize posterior to obtein opt parameter --> MAP
#the same is minimize negative log of the posterior
#equivalente a una ridge regression con lambda=alpha/beta
def MAP(x,y,f,alpha=1,p=[1,()]):
    from scipy.optimize import minimize
    def neg_log_Posterior(args):
        beta=args[0]+1e-4       
        w=args[1:]
        yexp=f(x,w)
        return 0.5*np.sum((yexp-y)**2)+alpha*0.5*np.sum(w**2)/beta
    bnds = np.array([ (0, None)]+[(None,None)]*len(p[1:]))     
    return minimize(neg_log_Posterior,p,method='SLSQP',bounds=bnds)



x=np.linspace(0,1,10)
eps=0.2
y=datagen(x,eps)
plt.scatter(x,y,s=140,label='train',facecolors='w',edgecolor='lightblue')

def f(x,args=[1,2*np.pi]):
    a,b=args[0],args[1]
    return a*np.sin(b*x)

f=my_poly
alpha=0.001
#res= MAP(x,y,f,p=[16,0.001,1,7]).x
res= MAP(x,y,f,alpha,p=[1]+[0.1]*20).x
print('eps=sqrt(1/beta)=',np.round(np.sqrt(1/res[0]),2), ', lambda=',alpha/res[0],', p=',np.round(res[2:],3))

print(np.std(y-f(x,res[1:])))

x=np.linspace(0,1,100)
plt.plot(x,f(x,res[1:]),'r-')


x0=0.3
xm=f(x0,res[1:])
plt.plot([x0,x0],[-1.5,1.5],'k--')
plt.plot([0,1],[xm,xm],'k--')
plt.scatter(x0,xm,color='g')
x=np.linspace(xm-0.5,xm+0.5,100)
y=st.norm.pdf(x,xm,np.sqrt(1/res[0]))
plt.plot(y/10+x0*0.9,x,'b')
plt.show()

#point estimate of w and so this does not yet amount to a Bayesian treatment