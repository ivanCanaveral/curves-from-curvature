from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def k(s):
    ''' Curvature function'''
    #return 1-np.tanh(s)
    #return (s**2+1.5)
    #return np.cos(s)
    return 1./5 + np.sin(s/2.) - np.cos(s/3.)
    
def rhs_eqs(Y, s):
    ''' Eq. system'''
    x,y,dx,dy = Y
    return [dx, dy, -k(s)*dy, k(s)*dx]

delta = 0.01

x0, y0, dx0, dy0 = 0,0,1,0
init_cond = [x0, y0, dx0, dy0]

#interval = np.arange(-2*np.pi, 2*np.pi+delta, delta)
interval = np.arange(-30*np.pi, 30*np.pi+delta, delta)  # interval for f

solu = odeint(rhs_eqs, init_cond, interval)

curve_x = [x for [x,y,dx,dy] in solu]
curve_y = [y for [x,y,dx,dy] in solu]

speed = [np.sqrt(dx**2+dy**2) for [x,y,dx,dy] in solu]

fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])


ax0.plot(curve_x,curve_y)
ax0.axis('equal')
ax0.set_title('Curve')
ax1.plot(interval,speed)
ax1.axis('equal')
ax1.set_title('Speed')

plt.tight_layout()
#plt.savefig('grid_figure.pdf')
plt.show()
