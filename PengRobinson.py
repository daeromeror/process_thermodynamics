__author__ = 'Daniel Romero Romero'

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy.optimize import fsolve

class PengRobinson:
    
    R = constants.R * 10 #bar cm3/gmol K
    
    def __init__(self, compounds:dict, z:list, T = 318.5, P = None):
        self.compounds = compounds
        
        self.T = T
        self.P = P
        self.init_z = np.array(z)
        
        self.w = self._array('w')
        self.Pc = self._array('P_c')
        self.Tc = self._array('T_c')
    
    def _array(self, label:str):
        return np.array([self.compounds.get(x).get(label) 
                        for x in self.compounds.keys()])
    
    def __A(self, z):
        return (np.sum(z**2*self.__a()) + 
                2*np.prod(z*self.__a()**0.5))
    
    def __a(self, Pc=None, Tc=None, w=None):
        w = self.w if w is None else np.array(w)
        Pc = self.Pc if Pc is None else np.array(Pc)
        Tc = self.Tc if Tc is None else np.array(Tc)
        beta = 0.37464 + 1.54226*w - 0.26992*w**2
        alpha = (1 + beta*(1 - (self.T/Tc)**0.5))**2
        return (0.45724*(type(self).R*Tc)**2*alpha/Pc)
    
    def __ak(self, z):
        a = self.__a()
        a0k = np.array([a[0],(np.prod(a))**0.5])
        a1k = np.array([a0k[1], a[1]])
        return np.array([np.dot(z,a0k), np.dot(z,a1k)])
    
    def __B(self, z):
        return np.sum(z*self.__b())
    
    def __b(self, Pc=None, Tc=None):
        Pc = self.Pc if Pc is None else np.array(Pc)
        Tc = self.Tc if Tc is None else np.array(Tc)
        return 0.07780*type(self).R*Tc/Pc
    
    def Z(self, z, P=None):
        P = self.P if P is None else P
        A = self.__A(z)*P/(type(self).R*self.T)**2
        B = self.__B(z)*P/(type(self).R*self.T)
        coeff = [1, B-1, A-3*B**2-2*B, B**3+B**2-A*B]
        return np.roots(coeff)
    
    def _phi(self, z, Z, P=None):
        P = self.P if P is None else P
        A = self.__A(z)*P/(type(self).R*self.T)**2
        B = self.__B(z)*P/(type(self).R*self.T)
        mean_ak = self.__ak(z)
        ln_phi = (
            self.__b()/self.__B(z)*(Z-1)-np.log(Z-B)-
            (A/(8**0.5*B))*(2*self.__ak(z)/self.__A(z)-
                            self.__b()/self.__B(z))*
            np.log((Z+B*(1+2**0.5))/(Z+B*(1-2**0.5))))
        return np.exp(ln_phi)
    
    def _y0(self, x, P:float):
        return self.Pc/P*np.exp(5.37*(1+self.w)*(1-self.Tc/self.T))*x
    
    def _bpp0(self, x):
        return np.sum(x*self.Pc*np.exp(5.37*(1+self.w)*(1-self.Tc/self.T)))
    
    def bpp(self, z=None):
        z = self.init_z if z is None else np.array(z)
        P0 = self._bpp0(z)
        sol = fsolve(self._bpp, P0, args=(z))
        return sol
    
    def _bpp(self, P, z):
        n = 0
        y = self._y0(z, P)
        y0 = np.full_like(y, 0.01)
        Z_l = np.min(self.Z(z, P))
        phi_l = self._phi(z, Z_l, P)
        while np.all(np.abs(y-y0)>1e-3) and n <= 30:
            n += 1
            y0 = y
            Z_v = np.max(self.Z(y, P))
            phi_v = self._phi(z, Z_v, P)
            y = (phi_l*z/phi_v)
        return 1-np.sum(y)
    
    def _x0(self, y, P:float):
        return y/(np.exp(np.log(self.Pc/P)+np.log(10)*7/3*(1+self.w)*(1-self.Tc/self.T)))
    
    def _dpp0(self, y):
        return 1/np.sum(y/np.exp(np.log(self.Pc)+np.log(10)*7/3*(1+self.w)*(1-self.Tc/self.T)))
    
    def dpp(self, z=None):
        z = self.init_z if z is None else np.array(z)
        P0 = self._dpp0(z)
        sol = fsolve(self._dpp, P0, args=(z))
        return sol
    
    def _dpp(self, P, z):
        n = 0
        x = self._x0(z, P)
        x0 = np.full_like(x, 0.01)
        Z_v = np.max(self.Z(z, P))
        phi_v = self._phi(z, Z_v, P)
        while np.all(np.abs(x-x0)>1e-3) and n <= 30:
            n += 1
            x0 = x
            Z_l = np.min(self.Z(x, P))
            phi_l = self._phi(z, Z_l, P)
            x = (phi_v*z/phi_l)
        return 1-np.sum(x)

def main():
    z = []
    BPP = []
    DPP = []
    
    compounds = {
        'isobutane':{'P_c':36.48,'T_c':408.1,'w':0.181},
        'isopentane':{'P_c':33.80,'T_c':460.4,'w':0.200}
    }
    
    x_exp = [0.9495, 0.9137, 0.8546, 0.5032, 0.499, 0.4903]
    BPP_exp = [5.869, 5.74, 5.464, 4.029, 3.965, 3.894]
    
    fluid = PengRobinson(compounds, [0, 1])
    for i in np.linspace(0,1,21):
        z.append(i)
        BPP.append(fluid.bpp([i, 1-i]).item())
        DPP.append(fluid.dpp([i, 1-i]).item())
    plt.title('Pxy diagram. System: i-butane/i-pentane @318.5K')
    plt.xlabel('x, y')
    plt.ylabel('Pressure, (bar)')
    plt.plot(z, BPP, '-k', label='Bubble P.')
    plt.plot(z, DPP, '--k', label='Dew P.')
    plt.plot(x_exp, BPP_exp, 'ok', ls='None', label='Exp. Bubble P.')
    plt.legend()

if __name__ == '__main__':
    main()
