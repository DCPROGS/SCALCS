import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        draw()

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0,4,50)
ydata = func(xdata, 2.5, 1.3, 0.5)
yn = ydata + 0.01*np.random.normal(size=len(xdata))

plt.figure()
plt.plot(xdata, yn, 'ko', label="Original Noised Data")
vl1 = plt.axvline(x=0, visible=True)
vl2 = plt.axvline(x=xdata[-1], visible=True)

s = 'Define the range of data to fit. First click left limit then click right limit...'
plt.title(s,fontsize=16)
plt.waitforbuttonpress()


happy = False
while not happy:
    pts = []
    while len(pts) < 2:
        pts = np.asarray(plt.ginput(2)) #,timeout=-1) )
        if len(pts) < 2:
            tellme('Too few points, starting over')
            time.sleep(1) # Wait a second


    vl1.remove()
    vl2.remove()
    vl1 = plt.axvline(x=pts[0,0], visible=True)
    vl2 = plt.axvline(x=pts[1,0], visible=True)
    plt.draw()
    print 'points=', pts
    s = 'Happy? Key click for yes, mouse click for no'
    plt.title(s,fontsize=16)
    happy = plt.waitforbuttonpress()

x1 = xdata[np.where( xdata > pts[0, 0] )]
y1 = yn[len(xdata)-len(x1): ]
x2 = x1[np.where( x1 < pts[1, 0] )]
y2 = y1[:len(x2)]
#plt.plot(x2, y2, 'ro', label="Selected points")

popt, pcov = curve_fit(func, x2, y2)
print 'pcov=', pcov

plt.plot(x2, func(x2, *popt), 'r-', label="Fitted Curve")
plt.legend()
s = 'Data fitting finished'
plt.title(s,fontsize=16)
plt.show()