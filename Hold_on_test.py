# -*- coding: utf-8 -*-
from pylab import *

figure()
y = array([1,2,0,-1,1])
t = array([10,11,12,13,14])

plot(t, y, '+-.', t, 2*y,'s-')
savefig("example7.png")
show()

hold(0);
plot(t, y,'bx--')
plot(t, y + 1,'gx--')
plot(t, y - 1,'kx--')
savefig("example8.png")
show()