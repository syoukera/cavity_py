import numpy as np
import matplotlib.pyplot as plt


class variables():
    '''Class for strage global variables'''
    
    def __init__(self):
        self.nx = 12
        self.ny = 12
        
        self.ni = self.nx
        self.nj = self.ny
        self.nim1 = self.nx - 1 # index of -1
        self.njm1 = self.ny - 1 # index of -1 
                
        self.xmax = 1.0
        self.ymax = 1.0
        self.uwall = 1.0e-4
        
        self.dx = self.xmax/float(self.nx-2)
        self.dy = self.ymax/float(self.ny-2)
        self.X = np.linspace(-self.dx/2, self.xmax+self.dx/2, self.nx)
        self.Y = np.linspace(-self.dy/2, self.xmax+self.dy/2, self.ny)
        
        self.X_mat = np.zeros((self.nx, self.ny))
        self.X_mat += self.X.reshape((self.nx, 1))
        self.Y_mat = np.zeros((self.nx, self.ny))
        self.Y_mat += self.Y.reshape((1, self.ny))
            
        self.DXEP = np.zeros(self.nx)
        self.DXPW = np.zeros(self.nx)
        self.DXEP[:-1] = self.X[1:].copy() - self.X[:-1].copy()
        self.DXPW[1:] = self.X[1:].copy() - self.X[:-1].copy()
        
        self.DYNP = np.zeros(self.ny)
        self.DYPS = np.zeros(self.ny)
        self.DYNP[:-1] = self.Y[1:].copy() - self.Y[:-1].copy()
        self.DYPS[1:] = self.Y[1:].copy() - self.Y[:-1].copy()
        
        self.SEW = np.zeros(self.nx)
        self.SEW[1:-1] = 0.5*(self.DXEP[1:-1].copy() + self.DXPW[1:-1].copy())
        self.SNS = np.zeros(self.ny)
        self.SNS[1:-1] = 0.5*(self.DYNP[1:-1].copy() + self.DYPS[1:-1].copy())
        
        self.XU = np.zeros(self.nx)
        self.XU[1:] = (self.X[1:].copy() + self.X[:-1].copy())/2
        self.YV = np.zeros(self.ny)
        self.YV[1:] = (self.Y[1:].copy() + self.Y[:-1].copy())/2
        
        self.nswpu = 1
        self.urfu = 0.5
        self.resoru = 0.0
        self.DXEPU = np.zeros(self.nx)
        self.DXEPU[1:-1] = self.XU[2:].copy() - self.XU[1:-1].copy()
        self.DXPWU = np.zeros(self.nx)
        self.DXPWU[2:] = self.XU[2:].copy() - self.XU[1:-1].copy()
        self.SEWU = np.zeros(self.nx)
        self.SEWU[1:] = (self.X[1:].copy() - self.X[:-1].copy())
        
        self.nswpv = 1
        self.urfv = 0.5
        self.resorv = 0.0
        self.DYNPV = np.zeros(self.ny)
        self.DYNPV[1:-1] = self.YV[2:].copy() - self.YV[1:-1].copy()
        self.DYPSV = np.zeros(self.ny)
        self.DYPSV[2:] = self.YV[2:].copy() - self.YV[1:-1].copy()
        self.SNSV = np.zeros(self.ny)
        self.SNSV[1:] = self.Y[1:].copy() - self.Y[:-1].copy()
        
        self.nswpp = 1
        self.ipref = 2
        self.jpref = 2
        self.urfp = 0.8
        self.resorm = 0.0
        self.DU = np.zeros((self.nx, self.ny))
        self.DV = np.zeros((self.nx, self.ny))
        
        self.U = np.zeros((self.nx, self.ny))
        self.U[:, -1] = self.uwall
        self.V = np.zeros((self.nx, self.ny))
        self.P = np.zeros((self.nx, self.ny))
        self.PP = np.zeros((self.nx, self.ny))
        
        self.viscos = 1.0e-3
        self.densit = 1000.0
        
        self.AP = np.zeros((self.nx, self.ny))
        self.AN = np.zeros((self.nx, self.ny))
        self.AS = np.zeros((self.nx, self.ny))
        self.AE = np.zeros((self.nx, self.ny))
        self.AW = np.zeros((self.nx, self.ny))
        self.SU = np.zeros((self.nx, self.ny))
        self.SP = np.zeros((self.nx, self.ny))
        
        self.flowin = (self.SNS*self.densit*self.uwall).sum()
        self.xmonin = (self.SNS*self.densit*self.uwall*self.uwall).sum()


def lisolv(istart, jstart, PHI, var):
    
    A = np.zeros(var.ny)
    B = np.zeros(var.ny)
    C = np.zeros(var.ny)
    D = np.zeros(var.ny)
    
    jstm1 = jstart - 1
    A[jstm1-1] = 0.0
    
    for i in range(istart-1, var.nim1):
        C[jstm1-1] = PHI[i, jstm1-1]
        
        for j in range(jstart-1, var.njm1):
            A[j] = var.AN[i, j]
            B[j] = var.AS[i, j]
            C[j] = var.AE[i, j]*PHI[i+1, j] + var.AW[i, j]*PHI[i-1, j] + var.SU[i, j]
            D[j] = var.AP[i, j]
            
            term = 1.0/(D[j] - B[j]*A[j-1])
            A[j] = A[j]*term
            C[j] = (C[j] + B[j]*C[j-1])*term
            
        for j in range(var.njm1-1, jstart-2, -1):
                        
            PHI[i, j] = A[j]*PHI[i, j+1] + C[j]


def calcu(var):
    
    for i in range(2, var.nim1):
        for j in range(1, var.njm1):
            cn = 0.5*var.densit*(var.V[i  , j+1] + var.V[i-1, j+1])*var.SEWU[i]
            cs = 0.5*var.densit*(var.V[i  , j  ] + var.V[i-1, j  ])*var.SEWU[i]
            ce = 0.5*var.densit*(var.U[i+1, j  ] + var.U[i  , j  ])*var.SNS[j]
            cw = 0.5*var.densit*(var.U[i  , j  ] + var.U[i-1, j  ])*var.SNS[j]
            dn = var.viscos*var.SEWU[i]/var.DYNP[j]
            ds = var.viscos*var.SEWU[i]/var.DYPS[j]
            de = var.viscos*var.SNS[j]/var.DXEPU[i]
            dw = var.viscos*var.SNS[j]/var.DXPWU[i]
            var.AN[i, j] = max(abs(0.5*cn), dn) - 0.5*cn
            var.AS[i, j] = max(abs(0.5*cs), ds) + 0.5*cs
            var.AE[i, j] = max(abs(0.5*ce), de) - 0.5*ce
            var.AW[i, j] = max(abs(0.5*cw), dw) + 0.5*cw
            var.DU[i, j] = var.SNS[j]
            var.SU[i, j] = var.DU[i, j]*(var.P[i-1, j] - var.P[i, j])
            var.SP[i, j] = 0.0
            
    yp = var.YV[-1] - var.Y[-2]
    j = var.njm1-1
    for i in range(2, var.nim1):
        tmult = var.viscos/yp
        var.SP[i, j] = var.SP[i, j] - tmult*var.SEWU[i]
        var.SU[i, j] = var.SU[i, j] + tmult*var.SEWU[i]*var.U[i, j+1]
        var.AN[i, j] = 0.0
        
    yp = var.Y[1] - var.YV[1]
    j = 1
    for i in range(2, var.nim1):
        tmult = var.viscos/yp
        var.SP[i, j] = var.SP[i, j] - tmult*var.SEWU[i]
        var.SU[i, j] = var.SU[i, j] + tmult*var.SEWU[i]*var.U[i, j-1]
        var.AS[i, j] = 0.0
        
    for j in range(1, var.njm1):
        var.AE[-2, j] = 0.0
        var.AW[2, j] = 0.0
        
    var.resoru = 0.0
    for i in range(2, var.nim1):
        for j in range(1, var.njm1):
            var.AP[i, j] = var.AN[i, j] + var.AS[i, j] + var.AE[i, j] + var.AW[i, j]                          - var.SP[i, j]
            var.DU[i, j] = var.DU[i, j]/var.AP[i, j]
            resor = var.AN[i, j]*var.U[i  , j+1] + var.AS[i, j]*var.U[i  , j-1]                   + var.AE[i, j]*var.U[i+1, j  ] + var.AW[i, j]*var.U[i-1, j  ]                   - var.AP[i, j]*var.U[i  , j  ] + var.SU[i, j]
            var.resoru += abs(resor)
            
            # under relaxation
            var.AP[i, j] = var.AP[i, j]/var.urfu
            var.SU[i, j] = var.SU[i, j] + (1.0 - var.urfu)*var.AP[i, j]*var.U[i, j]
            var.DU[i, j] = var.DU[i, j]*var.urfu
            
    for i in range(var.nswpu):
        lisolv(3, 2, var.U, var)      


def calcv(var):
    
    for i in range(1, var.nim1):
        for j in range(2, var.njm1):
            cn = 0.5*var.densit*(var.V[i  , j+1] + var.V[i  , j  ])*var.SEW[i]
            cs = 0.5*var.densit*(var.V[i  , j  ] + var.V[i  , j-1])*var.SEW[i]
            ce = 0.5*var.densit*(var.U[i+1, j  ] + var.U[i+1, j-1])*var.SNSV[j]
            cw = 0.5*var.densit*(var.U[i  , j  ] + var.U[i  , j-1])*var.SNSV[j]
            dn = var.viscos*var.SEW[i]/var.DYNPV[j]
            ds = var.viscos*var.SEW[i]/var.DYPSV[j]
            de = var.viscos*var.SNSV[j]/var.DXEP[i]
            dw = var.viscos*var.SNSV[j]/var.DXPW[i]
            var.AN[i, j] = max(abs(0.5*cn), dn) - 0.5*cn
            var.AS[i, j] = max(abs(0.5*cs), ds) + 0.5*cs
            var.AE[i, j] = max(abs(0.5*ce), de) - 0.5*ce
            var.AW[i, j] = max(abs(0.5*cw), dw) + 0.5*cw
            var.DV[i, j] = var.SEW[i]
            var.SU[i, j] = var.DV[i, j]*(var.P[i, j-1] - var.P[i, j])
            var.SP[i, j] = 0.0
            
    # West wall
    xp = var.X[1] - var.XU[1]
    i = 1
    for j in range(2, var.njm1):
        tmult = var.viscos/xp
        var.SP[i, j] = var.SP[i, j] - tmult*var.SNSV[j]
        var.SU[i, j] = var.SU[i, j] + tmult*var.SNSV[j]*var.V[i-1, j]
        var.AW[i, j] = 0.0
        
    # East wall
    xp = var.XU[-1] - var.X[-2]
    i = var.nim1 - 1
    for j in range(2, var.njm1):
        tmult = var.viscos/xp
        var.SP[i, j] = var.SP[i, j] - tmult*var.SNSV[j]
        var.SU[i, j] = var.SU[i, j] + tmult*var.SNSV[j]*var.V[i+1, j]
        var.AE[i, j] = 0.0
        
    # Top and bottom wall
    for i in range(1, var.nim1):
        var.AS[i, 2] = 0.0
        var.AN[i, -2] = 0.0
        
    var.resorv = 0.0
    for i in range(1, var.nim1):
        for j in range(2, var.njm1):
            var.AP[i, j] = var.AN[i, j] + var.AS[i, j] + var.AE[i, j] + var.AW[i, j]                          - var.SP[i, j]
            var.DV[i, j] = var.DV[i, j]/var.AP[i, j]
            resor = var.AN[i, j]*var.V[i  , j+1] + var.AS[i, j]*var.V[i  , j-1]                   + var.AE[i, j]*var.V[i+1, j  ] + var.AW[i, j]*var.V[i-1, j  ]                   - var.AP[i, j]*var.V[i  , j  ] + var.SU[i, j]
            var.resorv += abs(resor)
            
            # under relaxation
            var.AP[i, j] = var.AP[i, j]/var.urfv
            var.SU[i, j] = var.SU[i, j] + (1.0 - var.urfv)*var.AP[i, j]*var.V[i, j]
            var.DV[i, j] = var.DV[i, j]*var.urfv
            
    for i in range(var.nswpv):
        lisolv(2, 3, var.V, var)


def calcp(var):
    
    var.resorm = 0.0
    
    for i in range(1, var.nim1):
        for j in range(1, var.njm1):
            var.AN[i, j] = var.densit*var.SEW[i]*var.DV[i  , j+1]
            var.AS[i, j] = var.densit*var.SEW[i]*var.DV[i  , j  ]
            var.AE[i, j] = var.densit*var.SNS[j]*var.DU[i+1, j  ]
            var.AW[i, j] = var.densit*var.SNS[j]*var.DU[i  , j  ]
            
            cn = var.densit*var.V[i  , j+1]*var.SEW[i]
            cs = var.densit*var.V[i  , j  ]*var.SEW[i]
            ce = var.densit*var.U[i+1, j  ]*var.SNS[j]
            cw = var.densit*var.U[i  , j  ]*var.SNS[j]
            smp = cn - cs + ce - cw
            
            var.SP[i, j] = 0.0
            var.SU[i, j] = - smp
            
            var.resorm += abs(smp)
        
    for i in range(1, var.nim1):
        for j in range(1, var.njm1):
            var.AP[i, j] = var.AN[i, j] + var.AS[i, j] + var.AE[i, j] + var.AW[i, j]                          - var.SP[i, j]
            
    var.PP *= 0.0
            
    for i in range(var.nswpp):
        lisolv(2, 2, var.PP, var)      

    for i in range(1, var.nim1):
        for j in range(1, var.njm1):
            if i != 1: 
                var.U[i, j] = var.U[i, j] + var.DU[i, j]*(var.PP[i-1, j  ] - var.PP[i, j])
            if j != 1:
                var.V[i, j] = var.V[i, j] + var.DV[i, j]*(var.PP[i  , j-1] - var.PP[i, j])
            
    ppref = var.PP[var.ipref-1, var.jpref-1]
    
    for i in range(1, var.nim1):
        for j in range(1, var.njm1):
            var.P[i, j] = var.P[i, j] + var.urfp*(var.PP[i, j] - ppref)
#             var.PP[i, j] = 0.0

def main():
    var = variables()

    maxit = 100
    imon = 5
    jmon = 5
    sormax = 1.0e-3
    source = 1.0e10

    for niter in range(maxit):

        calcu(var)
        calcv(var)
        calcp(var)

        resorm = var.resorm/var.flowin
        resoru = var.resoru/var.xmonin
        resorv = var.resorv/var.xmonin
        
        print(f'{niter} {resoru:.2e} {resorv:.2e} {resorm:.2e} ', end='')
        print(f'{var.U[imon, jmon]:.2e} {var.V[imon, jmon]:.2e} {var.P[imon, jmon]:.2e}')
        
        source = max(resorm, resoru, resorv)
        
        if source < sormax:
            print('Converged!')
            break
            
        if niter == maxit-1:
            print('Not converged...')

if __name__ is '__main__':
    main()