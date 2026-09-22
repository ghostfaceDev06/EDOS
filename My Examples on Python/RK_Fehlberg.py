import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#Fehlberg Runge-Kutta
def fehlberg(f,t0,tf,w0,tol,hmax,hmin):

    #First pass
    t = t0
    w = w0
    h = hmax
    FLAG = 1
    
    t_hist = [t0]
    w_hist = [w0]
    h_hist = [h]
    mission = True

    #Second pass
    while FLAG == 1:

        #Third pass
        K1 = h*f(t,w)
        K2 = h*f(t + h/4,w + K1/4)
        K3 = h*f(t + 3*h/8,w + 3*K1/32 + 9*K2/32)
        K4 = h*f(t + 12*h/13,w + 1932*K1/2197 - 7200*K2/2197 + 7296*K3/2197)
        K5 = h*f(t + h,w + 439*K1/216 - 8*K2 + 3680*K3/513 - 845*K4/4104 )
        K6 = h*f(t + h/2,w - 8*K1/27 + 2*K2 - 3544*K3/2565 + 1859*K4/4104 - 11*K5/40) 

        #Fourth pass
        R = 1/h * abs(K1/360 -128*K3/4275 - 2197*K4/75240 + K5/50 + 2*K6/55)

        #Fifth pass
        if R <= tol:
            #Sixth pass
            t += h
            w += 25*K1/216 + 1408*K3/2565 + 2197*K4/4104 - K5/5

            #Seventh pass
            h_hist.append(h)
            t_hist.append(t)
            w_hist.append(w)
        #End if

        #Eighth pass
        delta = 0.84*(tol/R)**(1/4)

        #Nineth pass
        if delta < 0.1:
            h = 0.1*h
        #End if
        
        elif delta >= 4.0:
            h = 4.0*h
        #End elif

        else:
            h = delta*h
        #End else

        #Tenth pass
        if h > hmax:
            h = hmax
        #End if

        #Eleventh pass
        if t >= tf:
            FLAG = 0
            mission = True
        #End if

        elif t + h > tf:
            h = tf - t
        #End elif

        elif h < hmin:
            FLAG = 0
            mission = False
            print('h in excess')
        #End elif
    #End while        

    return h_hist,w_hist,t_hist,mission
#End function
        
#ODE
f = lambda t,y: y - t**2 + 1

#Analytic ODE
exact_y = lambda t: (t+1)**2 - 0.5*np.exp(t)

#Defining values
tf, t0 , w0 , hmax , hmin , tol = 2, 0, 0.5 , 0.25, 0.01, 1e-5

#Calling function
h_hist,w_hist,t_hist,mission = fehlberg(f,t0,tf,w0,tol,hmax,hmin) 

#To transform in array
t_arr = np.array(t_hist)
w_arr = np.array(w_hist)
h_arr = np.array(h_hist)

#Calculing y this ODE
original_y = exact_y(t_arr)
n = len(t_arr)

#Calculing Absolute and relative error
abs_error = np.abs(w_arr - original_y) 
rel_error = np.abs(abs_error/abs(w_arr))

#Create dictionary
dc = {
    'h': h_arr,
    't': t_arr,
    'Exact': original_y,
    'Fehlberg': w_arr,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(t_arr,original_y,'k-',linewidth=2,label='Exact')
plt.plot(t_arr,w_arr,'ro--',linewidth=1.5,label='Fehlberg')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Fehlberg Runge-Kutta x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()