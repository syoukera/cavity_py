from cavity import cavity

def main():
    var = cavity()

    maxit = 100
    imon = 5
    jmon = 5
    sormax = 1.0e-3
    source = 1.0e10

    for niter in range(maxit):

        var.calcu()
        var.calcv()
        var.calcp()

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

if __name__ == '__main__':
    main()