#Code for using lagrange multipliers to minimize Rsum. This has been done by hand using only a solver for solving systems of equations. All derivatives have been double-checked. 

#In every set, for deg 3 and deg 4, this method fits Rsum exactly to its constraint at x=1, even for those that do not require the constraint


from scipy.optimize import curve_fit
import numpy as np
import glob
import pandas as pd
from sympy import solve, symbols, diff


#FLAG -> j=3, degree 3; j=4, degree 4
j=3

#pull files from folder
filenames=np.sort(glob.glob('*.txt'))

#empty list to create dataframe
df_list=[]

#column names for datafiles
columns = ['Q2','W','cos','eps','o','do','pz1','dpz1','pz','dpz','c','Rs','dRs']

#add files to dataframe and name columns
for f in filenames:
    df_list.append(pd.read_csv(f,names=columns))

#chunk dataframe for iterations through datasets
data_set = [df_list[0],df_list[1],df_list[2],df_list[3],df_list[4],df_list[5],df_list[6],df_list[7]]

#datafile for energies, sigma/dsigma at 1, c0 at 1 for each dataset
s = pd.read_csv('all_sigma_odo1.csv', header=None, names=['dataset','o1','do1','c1'])


#iterate through datasets
for i in range(0,8):
    #variables from files
    cos = data_set[i]['cos']
    pz = data_set[i]['pz']
    pz1 = data_set[i]['pz1']
    dpz = data_set[i]['dpz']
    dpz1 = data_set[i]['dpz1']
    o = data_set[i]['o']
    do = data_set[i]['do']
    w = data_set[i]['W']
    q2 = data_set[i]['Q2']
    e = data_set[i]['eps']    
    c = data_set[i]['c']  
    Rs = data_set[i]['Rs']
    dRs = data_set[i]['dRs']        
    o1 = s['o1'][i]
    do1 = s['do1'][i]

    

    #energies for each set
    k=s['dataset'][i]
    
    #x values for fit plots
    xx = np.arange(-1., 1.5,0.0001)

    #define function to be minimized
    def func(x,y,a0,a1,a2,a3):
        return (a0*(1+x)+a1*(x+x**2)+a2*(x**3-x)+a3*(x+x**4))-y
    
    #curve fit for unminimized fit
    popt,pcov = curve_fit(func,cos,Rs)
    
    
#Lagrange multiplier method, solve for x,y,λ in ∇L=∇f(x,y)-λ∇g(x,y).
#The steps to do this are as follows:
    #Find the Karush-Kuhn-Tucker conditions:
        #dL/dx, dL/dy, dL/dλ = 0
    #Use this system of equations to solve for x,y,λ, where the maximum or minimum of the function is at (x,y) (critial point)

    
    #symbols to solve for (using sympy)
    x,y,λ = symbols("x,y,λ")
  
    #function to be minimized
    func = (popt[0]*(1+x) + popt[1]*(x+x**2) + popt[2]*(x**3-x) + popt[3]*(x**4+x))
    
    #constraint function
    gunc = y-2*o1
        
    #Lagrangian function
    L = func - λ*gunc
    
    #Karush-Kuhn-Tucker (KKT) conditions. all derivatives = 0
    dLx = diff(L,x)    
    dLy = diff(L,y)
    dLλ = diff(L,λ)
        
    #solve system of equations 
    solutions = solve((dLx,dLy,dLλ),[x,y,λ])
    
    print('\n','solutions for x, y, and λ','\n',solutions)
    
    print('Maximum Rsum(1) =',2*o1)
    
    
    
    
