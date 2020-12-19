import numpy as np
import matplotlib.pyplot as plt

class cavity():
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


    def lisolv(self, istart, jstart, PHI):
        
        A = np.zeros(self.ny)
        B = np.zeros(self.ny)
        C = np.zeros(self.ny)
        D = np.zeros(self.ny)
        
        jstm1 = jstart - 1
        A[jstm1-1] = 0.0
        
        for i in range(istart-1, self.nim1):
            C[jstm1-1] = PHI[i, jstm1-1]
            
            for j in range(jstart-1, self.njm1):
                A[j] = self.AN[i, j]
                B[j] = self.AS[i, j]
                C[j] = self.AE[i, j]*PHI[i+1, j] + self.AW[i, j]*PHI[i-1, j] + self.SU[i, j]
                D[j] = self.AP[i, j]
                
                term = 1.0/(D[j] - B[j]*A[j-1])
                A[j] = A[j]*term
                C[j] = (C[j] + B[j]*C[j-1])*term
                
            for j in range(self.njm1-1, jstart-2, -1):
                            
                PHI[i, j] = A[j]*PHI[i, j+1] + C[j]


    def calcu(self):
        
        for i in range(2, self.nim1):
            for j in range(1, self.njm1):
                cn = 0.5*self.densit*(self.V[i  , j+1] + self.V[i-1, j+1])*self.SEWU[i]
                cs = 0.5*self.densit*(self.V[i  , j  ] + self.V[i-1, j  ])*self.SEWU[i]
                ce = 0.5*self.densit*(self.U[i+1, j  ] + self.U[i  , j  ])*self.SNS[j]
                cw = 0.5*self.densit*(self.U[i  , j  ] + self.U[i-1, j  ])*self.SNS[j]
                dn = self.viscos*self.SEWU[i]/self.DYNP[j]
                ds = self.viscos*self.SEWU[i]/self.DYPS[j]
                de = self.viscos*self.SNS[j]/self.DXEPU[i]
                dw = self.viscos*self.SNS[j]/self.DXPWU[i]
                self.AN[i, j] = max(abs(0.5*cn), dn) - 0.5*cn
                self.AS[i, j] = max(abs(0.5*cs), ds) + 0.5*cs
                self.AE[i, j] = max(abs(0.5*ce), de) - 0.5*ce
                self.AW[i, j] = max(abs(0.5*cw), dw) + 0.5*cw
                self.DU[i, j] = self.SNS[j]
                self.SU[i, j] = self.DU[i, j]*(self.P[i-1, j] - self.P[i, j])
                self.SP[i, j] = 0.0
                
        yp = self.YV[-1] - self.Y[-2]
        j = self.njm1-1
        for i in range(2, self.nim1):
            tmult = self.viscos/yp
            self.SP[i, j] = self.SP[i, j] - tmult*self.SEWU[i]
            self.SU[i, j] = self.SU[i, j] + tmult*self.SEWU[i]*self.U[i, j+1]
            self.AN[i, j] = 0.0
            
        yp = self.Y[1] - self.YV[1]
        j = 1
        for i in range(2, self.nim1):
            tmult = self.viscos/yp
            self.SP[i, j] = self.SP[i, j] - tmult*self.SEWU[i]
            self.SU[i, j] = self.SU[i, j] + tmult*self.SEWU[i]*self.U[i, j-1]
            self.AS[i, j] = 0.0
            
        for j in range(1, self.njm1):
            self.AE[-2, j] = 0.0
            self.AW[2, j] = 0.0
            
        self.resoru = 0.0
        for i in range(2, self.nim1):
            for j in range(1, self.njm1):
                self.AP[i, j] = self.AN[i, j] + self.AS[i, j] + self.AE[i, j] + self.AW[i, j] \
                              - self.SP[i, j]
                self.DU[i, j] = self.DU[i, j]/self.AP[i, j]
                resor = self.AN[i, j]*self.U[i  , j+1] + self.AS[i, j]*self.U[i  , j-1] \
                      + self.AE[i, j]*self.U[i+1, j  ] + self.AW[i, j]*self.U[i-1, j  ] \
                      - self.AP[i, j]*self.U[i  , j  ] + self.SU[i, j]
                self.resoru += abs(resor)
                
                # under relaxation
                self.AP[i, j] = self.AP[i, j]/self.urfu
                self.SU[i, j] = self.SU[i, j] + (1.0 - self.urfu)*self.AP[i, j]*self.U[i, j]
                self.DU[i, j] = self.DU[i, j]*self.urfu
                
        for i in range(self.nswpu):
            self.lisolv(3, 2, self.U)      


    def calcv(self):
        
        for i in range(1, self.nim1):
            for j in range(2, self.njm1):
                cn = 0.5*self.densit*(self.V[i  , j+1] + self.V[i  , j  ])*self.SEW[i]
                cs = 0.5*self.densit*(self.V[i  , j  ] + self.V[i  , j-1])*self.SEW[i]
                ce = 0.5*self.densit*(self.U[i+1, j  ] + self.U[i+1, j-1])*self.SNSV[j]
                cw = 0.5*self.densit*(self.U[i  , j  ] + self.U[i  , j-1])*self.SNSV[j]
                dn = self.viscos*self.SEW[i]/self.DYNPV[j]
                ds = self.viscos*self.SEW[i]/self.DYPSV[j]
                de = self.viscos*self.SNSV[j]/self.DXEP[i]
                dw = self.viscos*self.SNSV[j]/self.DXPW[i]
                self.AN[i, j] = max(abs(0.5*cn), dn) - 0.5*cn
                self.AS[i, j] = max(abs(0.5*cs), ds) + 0.5*cs
                self.AE[i, j] = max(abs(0.5*ce), de) - 0.5*ce
                self.AW[i, j] = max(abs(0.5*cw), dw) + 0.5*cw
                self.DV[i, j] = self.SEW[i]
                self.SU[i, j] = self.DV[i, j]*(self.P[i, j-1] - self.P[i, j])
                self.SP[i, j] = 0.0
                
        # West wall
        xp = self.X[1] - self.XU[1]
        i = 1
        for j in range(2, self.njm1):
            tmult = self.viscos/xp
            self.SP[i, j] = self.SP[i, j] - tmult*self.SNSV[j]
            self.SU[i, j] = self.SU[i, j] + tmult*self.SNSV[j]*self.V[i-1, j]
            self.AW[i, j] = 0.0
            
        # East wall
        xp = self.XU[-1] - self.X[-2]
        i = self.nim1 - 1
        for j in range(2, self.njm1):
            tmult = self.viscos/xp
            self.SP[i, j] = self.SP[i, j] - tmult*self.SNSV[j]
            self.SU[i, j] = self.SU[i, j] + tmult*self.SNSV[j]*self.V[i+1, j]
            self.AE[i, j] = 0.0
            
        # Top and bottom wall
        for i in range(1, self.nim1):
            self.AS[i, 2] = 0.0
            self.AN[i, -2] = 0.0
            
        self.resorv = 0.0
        for i in range(1, self.nim1):
            for j in range(2, self.njm1):
                self.AP[i, j] = self.AN[i, j] + self.AS[i, j] + self.AE[i, j] + self.AW[i, j] \
                              - self.SP[i, j]
                self.DV[i, j] = self.DV[i, j]/self.AP[i, j]
                resor = self.AN[i, j]*self.V[i  , j+1] + self.AS[i, j]*self.V[i  , j-1] \
                      + self.AE[i, j]*self.V[i+1, j  ] + self.AW[i, j]*self.V[i-1, j  ] \
                      - self.AP[i, j]*self.V[i  , j  ] + self.SU[i, j]
                self.resorv += abs(resor)
                
                # under relaxation
                self.AP[i, j] = self.AP[i, j]/self.urfv
                self.SU[i, j] = self.SU[i, j] + (1.0 - self.urfv)*self.AP[i, j]*self.V[i, j]
                self.DV[i, j] = self.DV[i, j]*self.urfv
                
        for i in range(self.nswpv):
            self.lisolv(2, 3, self.V)


    def calcp(self):
        
        self.resorm = 0.0
        
        for i in range(1, self.nim1):
            for j in range(1, self.njm1):
                self.AN[i, j] = self.densit*self.SEW[i]*self.DV[i  , j+1]
                self.AS[i, j] = self.densit*self.SEW[i]*self.DV[i  , j  ]
                self.AE[i, j] = self.densit*self.SNS[j]*self.DU[i+1, j  ]
                self.AW[i, j] = self.densit*self.SNS[j]*self.DU[i  , j  ]
                
                cn = self.densit*self.V[i  , j+1]*self.SEW[i]
                cs = self.densit*self.V[i  , j  ]*self.SEW[i]
                ce = self.densit*self.U[i+1, j  ]*self.SNS[j]
                cw = self.densit*self.U[i  , j  ]*self.SNS[j]
                smp = cn - cs + ce - cw
                
                self.SP[i, j] = 0.0
                self.SU[i, j] = - smp
                
                self.resorm += abs(smp)
            
        for i in range(1, self.nim1):
            for j in range(1, self.njm1):
                self.AP[i, j] = self.AN[i, j] + self.AS[i, j] + self.AE[i, j] + self.AW[i, j] \
                              - self.SP[i, j]
                
        self.PP *= 0.0
                
        for i in range(self.nswpp):
            self.lisolv(2, 2, self.PP)      

        for i in range(1, self.nim1):
            for j in range(1, self.njm1):
                if i != 1: 
                    self.U[i, j] = self.U[i, j] + self.DU[i, j]*(self.PP[i-1, j  ] - self.PP[i, j])
                if j != 1:
                    self.V[i, j] = self.V[i, j] + self.DV[i, j]*(self.PP[i  , j-1] - self.PP[i, j])
                
        ppref = self.PP[self.ipref-1, self.jpref-1]
        
        for i in range(1, self.nim1):
            for j in range(1, self.njm1):
                self.P[i, j] = self.P[i, j] + self.urfp*(self.PP[i, j] - ppref)
    #             self.PP[i, j] = 0.0