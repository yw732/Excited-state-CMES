import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from scipy.constants import hbar, u, e
from scipy.sparse.linalg import eigsh
from scipy.sparse import spdiags, csc_matrix
from scipy.integrate import romb
from scipy.optimize import root
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S',filename='surfaces.log', encoding='utf-8', level=logging.INFO)

def grid_generator(bound):
    
    left = bound[0]
    right = bound[1]
    eps = bound[2]
        
    N = int((right-left)/eps+1)
    N = int(2**int(np.ceil(np.log2(N)))+1)  # odd number of grids: 2**k+1
    n = int((N-1)/2)
    
    dx = (right-left)/(N-1)
    dk = 2*np.pi/(N*dx)
    x_grid=np.linspace(left, right, N)
    
    return N, n, dx, dk, x_grid

# Fourier grid of 1D kinetic operator
def T_1D_generator(N, n, dx, dk, mass=1.0, debug=False):
    l=np.arange(1,n+0.1)
    
    z = smp.symbols('z')
    cos_sum = smp.sin(1/2*n*z)*smp.cos((n+1)/2*z)/smp.sin(z/2)   # sum_l^n(cos(lz))
    second_df = smp.diff(cos_sum, z, 2)
    T_z_zero = hbar**2*1e20/e * dk**2 / (N*mass*u) * sum(l**2)
    
    diagonals=np.zeros((N-1,N))
    T_upper=np.zeros((N,N))
    
    for i in range(1,N):
        z_val=2*np.pi*i/N
        T_z = -hbar**2*1e20/e * dk**2 / (N*mass*u) * second_df.subs({z:z_val}).evalf()
        diagonals[i-1,:]=T_z*np.ones(N)
        
    
        
    T_upper = spdiags(diagonals, np.arange(1,N), N, N).toarray()
    T_mat = T_upper + T_upper.T
    if debug:
        print(T_mat)
        
    T_mat+=np.diag(np.ones(N)*T_z_zero)
    
    if debug:
        print(T_mat)
        
    return T_mat

# double integral
def double_romb(function, Nx, Ny, dx, dy):
    function_xy = function.reshape([Nx, Ny], order='F')
    
    int_x = np.zeros(Ny)  # integrate along x direction first
    for i in range(Ny):
        int_x[i] = romb(function_xy[:,i], dx=dx)
        
    return romb(int_x, dx=dy)

# potential
def V(ax, ay, bx, by, cxy, cx2y, x, y, fx, fy):
    return ax*x**2+ay*y**2+bx*x**4+by*y**4+cxy*x*y+cx2y*x**2*y+fx*x+fy*y


def fdm_2D_constrained_solver(x_exp, y_exp, state, ax, ay, bx, cxy,cx2y=0, gs_coef = None, ex_coef = None, ex2_coef = None, init_f = None,
 by=0, mass=1.0, x_bound=(-0.7,0.7,0.03), y_bound=(-0.7,0.7,0.03), ntrial=300, debug = False,energy_solver=False):
    
    Nx, nx, dx, dkx, x_grid = grid_generator(x_bound)
    #logging.info(f'{Nx=}')
    Tx_mat = T_1D_generator(Nx, nx, dx, dkx)

    Ny, ny, dy, dky, y_grid = grid_generator(y_bound)
    Ty_mat = T_1D_generator(Ny, ny, dy, dky)

    # 4D indices: x,y,x,y
    H_4D = np.zeros((Nx, Ny, Nx, Ny))
    T_4D = np.zeros((Nx, Ny, Nx, Ny))
    V_4D = np.zeros((Nx, Ny, Nx, Ny))
    
    x_2D_grid = np.resize(x_grid, Nx*Ny)
    y_2D_grid = np.repeat(y_grid, Nx)
    
    for i in range(Nx):
        T_4D[:,i,:,i] += Tx_mat
    for i in range(Ny):
        T_4D[i,:,i,:] += Ty_mat
        
    if energy_solver:        #numerially solve the Schrodinger equation
        for i in range(Nx):
            for j in range(Ny):
                xi = x_grid[i]
                yj = y_grid[j]
                V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, 0, 0)
                
        H_4D = T_4D + V_4D
        H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')    
        ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
        
        return ene, coef    


    if state==0:
        if init_f == None:
            init_f = [0.0, 0.0]
            
        def xy_diff(f):
            for i in range(Nx):
                for j in range(Ny):
                    xi = x_grid[i]
                    yj = y_grid[j]
                    V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, f[0], f[1])
                    
            H_4D = T_4D + V_4D
            H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')
            
            ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
            xExp = []
            yExp = []
            true_e = []
            for i in range(ntrial):
                norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
                xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                true_e.append(ene[i]-f.item(0)*xExp[-1]-f.item(1)*yExp[-1])
                
            xExp = np.array(xExp)
            yExp = np.array(yExp)
            true_e = np.array(true_e)
            idx = np.argpartition(true_e, range(state+3))
            if debug:
                logging.info(f'{f=}, {xExp[idx[0]]-x_exp}, {yExp[idx[0]]-y_exp}')
                      
            return [xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp]
        
        sol = root(xy_diff, init_f, method = 'df-sane', options={'fatol': 1e-11})
        sol_f = sol.x
        for i in range(Nx):
            for j in range(Ny):
                xi = x_grid[i]
                yj = y_grid[j]
                V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, sol_f[0], sol_f[1])
                
        H_4D = T_4D + V_4D
        H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')

        ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
        xExp = []
        yExp = []
        true_e = []
        coef_norm = np.copy(coef)
        for i in range(ntrial):
            norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
            xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            true_e.append(ene[i]-sol_f[0]*xExp[-1]-sol_f[1]*yExp[-1])
            coef_norm[:,i] = coef_norm[:,i]/ np.sqrt(norm)

        xExp = np.array(xExp)
        yExp = np.array(yExp)
        true_e = np.array(true_e)
        idx = np.argpartition(true_e, range(state+3))
        return true_e[idx[0]], coef_norm[:,idx[0]], sol_f, xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp
    
    elif state == 1:
        if init_f == None:
            init_f = [0.0,0.0,0.0]   # fx, fy, g
            
        overlap_mat = np.outer(gs_coef, gs_coef)
        def xy_diff(f):
            for i in range(Nx):
                for j in range(Ny):
                    xi = x_grid[i]
                    yj = y_grid[j]
                    V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, f[0], f[1])
                    
            H_4D = T_4D + V_4D
            H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')+f[2]*overlap_mat
            
            ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
            xExp = []
            yExp = []
            true_e = []
            overlap = []
            for i in range(ntrial):
                norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
                xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
                true_e.append(ene[i]-f.item(0)*xExp[-1]-f.item(1)*yExp[-1]-f.item(2)*overlap[-1])
                
            xExp = np.array(xExp)
            yExp = np.array(yExp)
            true_e = np.array(true_e)
            overlap = np.array(overlap)
            idx = np.argpartition(true_e, range(state+3))
     
            if debug:
                logging.info(idx[0:3])
                logging.info(f'{f[0]=}, {xExp[idx[0]]-x_exp=}')
                logging.info(f'{f[1]=}, {yExp[idx[0]]-y_exp=}')
                logging.info(f'{f[2]=}, {overlap[idx[0]]=}')
                logging.info(f'{true_e[idx[0]]=}')      
            return [xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]]]
        
        sol = root(xy_diff, init_f, method = 'df-sane', options={'fatol': 1e-8})
        sol_f = sol.x
        for i in range(Nx):
            for j in range(Ny):
                xi = x_grid[i]
                yj = y_grid[j]
                V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, sol_f[0], sol_f[1])
                
        H_4D = T_4D + V_4D
        H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')+sol_f[2]*overlap_mat

        ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
        xExp = []
        yExp = []
        true_e = []
        overlap = []
        coef_norm = np.copy(coef)
        for i in range(ntrial):
            norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
            xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
            true_e.append(ene[i]-sol_f[0]*xExp[-1]-sol_f[1]*yExp[-1]-sol_f[2]*overlap[-1])

            coef_norm[:,i] = coef_norm[:,i]/ np.sqrt(norm)

        xExp = np.array(xExp)
        yExp = np.array(yExp)
        true_e = np.array(true_e)
        overlap = np.array(overlap)
        idx = np.argpartition(true_e, range(state+3))
        return true_e[idx[0]],coef_norm[:,idx[0]], sol_f, xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]]
    elif state==2:
        if init_f == None:
            init_f=[0.0,0.0,0.0,0.0]  #fx, fy, g00, g01
            
        overlap_mat = np.outer(gs_coef, gs_coef)
        overlap_mat_ex = np.outer(ex_coef, ex_coef)
        def xy_diff(f):
            for i in range(Nx):
                for j in range(Ny):
                    xi = x_grid[i]
                    yj = y_grid[j]
                    V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, f[0], f[1])
                    
            H_4D = T_4D + V_4D
            H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')+f[2]*overlap_mat + f[3]*overlap_mat_ex
            
            ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
            xExp = []
            yExp = []
            true_e = []
            overlap = []
            overlap_ex = []
            for i in range(ntrial):
                norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
                xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
                overlap_ex.append(double_romb(ex_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
                true_e.append(ene[i]-f.item(0)*xExp[-1]-f.item(1)*yExp[-1]-f.item(2)*overlap[-1]-f.item(3)*overlap_ex[-1])
                
            xExp = np.array(xExp)
            yExp = np.array(yExp)
            true_e = np.array(true_e)
            overlap = np.array(overlap)
            overlap_ex = np.array(overlap_ex)
            idx = np.argpartition(true_e, range(state+3))

            if debug:
                logging.info(idx[0:3])
                logging.info(f'{f[0]=}, {xExp[idx[0]]-x_exp=}')
                logging.info(f'{f[1]=}, {yExp[idx[0]]-y_exp=}')
                logging.info(f'{f[2]=}, {overlap[idx[0]]=}')
                logging.info(f'{f[3]=}, {overlap_ex[idx[0]]=}')
                      
            return [xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]], overlap_ex[idx[0]]]
        
        sol = root(xy_diff, init_f, method = 'df-sane', options={'fatol': 1e-8})
        sol_f = sol.x
        for i in range(Nx):
            for j in range(Ny):
                xi = x_grid[i]
                yj = y_grid[j]
                V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, sol_f[0], sol_f[1])
                
        H_4D = T_4D + V_4D
        H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')+sol_f[2]*overlap_mat+sol_f[3]*overlap_mat_ex

        ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
        xExp = []
        yExp = []
        true_e = []
        overlap = []
        overlap_ex = []
        coef_norm = np.copy(coef)
        for i in range(ntrial):
            norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
            xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
            overlap_ex.append(double_romb(ex_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
            true_e.append(ene[i]-sol_f[0]*xExp[-1]-sol_f[1]*yExp[-1]-sol_f[2]*overlap[-1]-sol_f[3]*overlap_ex[-1])

            coef_norm[:,i] = coef_norm[:,i]/ np.sqrt(norm)

        xExp = np.array(xExp)
        yExp = np.array(yExp)
        true_e = np.array(true_e)
        overlap = np.array(overlap)
        overlap_ex = np.array(overlap_ex)
        idx = np.argpartition(true_e, range(state+3))
        return true_e[idx[0]], coef_norm[:,idx[0]], sol_f, xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]], overlap_ex[idx[0]]
    elif state==3:
        if init_f == None:
            init_f=[0.0,0.0,0.0,0.0,0.0]  #fx, fy, g00, g01
            
        overlap_mat = np.outer(gs_coef, gs_coef)
        overlap_mat_ex = np.outer(ex_coef, ex_coef)
        overlap_mat_ex2 = np.outer(ex2_coef, ex2_coef)
        def xy_diff(f):
            for i in range(Nx):
                for j in range(Ny):
                    xi = x_grid[i]
                    yj = y_grid[j]
                    V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, f[0], f[1])
                    
            H_4D = T_4D + V_4D
            H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F') + f[2]*overlap_mat +f[3]*overlap_mat_ex+f[4]*overlap_mat_ex2
            
            ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
            xExp = []
            yExp = []
            true_e = []
            overlap = []
            overlap_ex = []
            overlap_ex2 = []
            for i in range(ntrial):
                norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
                xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
                overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
                overlap_ex.append(double_romb(ex_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
                overlap_ex2.append(double_romb(ex2_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
                true_e.append(ene[i]-f.item(0)*xExp[-1]-f.item(1)*yExp[-1]-f.item(2)*overlap[-1]-f.item(3)*overlap_ex[-1]-f.item(4)*overlap_ex2[-1])
                
            xExp = np.array(xExp)
            yExp = np.array(yExp)
            true_e = np.array(true_e)
            overlap = np.array(overlap)
            overlap_ex = np.array(overlap_ex)
            overlap_ex2 = np.array(overlap_ex2)
            idx = np.argpartition(true_e, range(state+3))

            if debug:
                logging.info(idx[0:3])
                logging.info(f'{f[0]=}, {xExp[idx[0]]-x_exp=}')
                logging.info(f'{f[1]=}, {yExp[idx[0]]-y_exp=}')
                logging.info(f'{f[2]=}, {overlap[idx[0]]=}')
                logging.info(f'{f[3]=}, {overlap_ex[idx[0]]=}')
                logging.info(f'{f[4]=}, {overlap_ex2[idx[0]]=}')
                logging.info(f'{true_e[idx[0]]=}')
                      
            return [xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]], overlap_ex[idx[0]], overlap_ex2[idx[0]]]
        
        sol = root(xy_diff, init_f, method = 'df-sane', options={'fatol': 5e-8})
        sol_f = sol.x
        for i in range(Nx):
            for j in range(Ny):
                xi = x_grid[i]
                yj = y_grid[j]
                V_4D[i,j,i,j] = V(ax, ay, bx, by, cxy,cx2y, xi, yj, sol_f[0], sol_f[1])
                
        H_4D = T_4D + V_4D
        H_mat = H_4D.reshape((Nx*Ny, Nx*Ny), order='F')+sol_f[2]*overlap_mat +sol_f[3]*overlap_mat_ex+sol_f[4]*overlap_mat_ex2
        
        ene, coef = eigsh(H_mat, k=ntrial,sigma=0, which='LM')
        xExp = []
        yExp = []
        true_e = []
        overlap = []
        overlap_ex = []
        overlap_ex2 = []
        coef_norm = np.copy(coef)
        for i in range(ntrial):
            norm = double_romb(coef[:,i]**2,Nx, Ny, dx, dy)
            xExp.append(double_romb(x_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            yExp.append(double_romb(y_2D_grid * coef[:,i] ** 2, Nx, Ny, dx, dy) / norm)
            overlap.append(double_romb(gs_coef * coef[:,i], Nx, Ny, dx, dy) ** 2/ norm)
            overlap_ex.append(double_romb(ex_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
            overlap_ex2.append(double_romb(ex2_coef * coef[:,i], Nx, Ny, dx, dy) **2/ norm)
            true_e.append(ene[i]-sol_f[0]*xExp[-1]-sol_f[1]*yExp[-1]-sol_f[2]*overlap[-1]-sol_f[3]*overlap_ex[-1]-sol_f[4]*overlap_ex2[-1])
            coef_norm[:,i] = coef_norm[:,i]/ np.sqrt(norm)

        xExp = np.array(xExp)
        yExp = np.array(yExp)
        true_e = np.array(true_e)
        overlap = np.array(overlap)
        overlap_ex = np.array(overlap_ex)
        overlap_ex2 = np.array(overlap_ex2)
        idx = np.argpartition(true_e, range(state+3))
        return true_e[idx[0]], coef_norm[:,idx[0]], sol_f, xExp[idx[0]]-x_exp, yExp[idx[0]]-y_exp, overlap[idx[0]], overlap_ex[idx[0]],overlap_ex2[idx[0]]


ax=15
ay=15
bx=30
by=20
cxy=10
cx2y=0

ene_qm, coef_qm = fdm_2D_constrained_solver(0, 0, 0, ax, ay, bx, cxy,cx2y=cx2y, by=by, energy_solver = True)
np.save('energy_ax%d_ay%d_bx%d_by%d_cxy_%d_cx2y%d'%(ax, ay, bx,by, cxy, cx2y), ene_qm)
np.save('wavefunction_ax%d_ay%d_bx%d_by%d_cxy_%d_cx2y%d'%(ax, ay, bx, by, cxy, cx2y), coef_qm)

xExp_range = np.arange(-0.02,0.021,0.02)
yExp_range = np.arange(-0.02,0.021,0.02)
lines=[]

for x_exp in xExp_range:
    for y_exp in yExp_range:
        E_gs, c_gs, f_gs, error_x, error_y = fdm_2D_constrained_solver(x_exp, y_exp, 0, ax, ay, bx, cxy,cx2y=cx2y, by=by, debug=False)
        E_ex, c_ex, f_ex, error_x, error_y, error_orth = fdm_2D_constrained_solver(x_exp, y_exp, 1,  ax, ay, bx, cxy,cx2y=cx2y, by=by, gs_coef = c_gs, init_f = [f_gs[0],f_gs[1],-10], debug = True)
        E_ex2, c_ex2, f_ex2, error_x, error_y, error_1, error_2 = fdm_2D_constrained_solver(x_exp, y_exp, 2,  ax, ay, bx, cxy,cx2y=cx2y, by=by, gs_coef = c_gs, ex_coef = c_ex, init_f = [f_ex[0],f_ex[1],-10,-10], debug = True)
        E_ex3, c_ex3, f_ex3, error_x, error_y, error_1, error_2, error_3 = fdm_2D_constrained_solver(x_exp, y_exp, 3,  ax, ay, bx, cxy, gs_coef = c_gs, ex_coef = c_ex, ex2_coef = c_ex2, init_f = [f_ex2[0],f_ex2[1],-20,-20,-20], debug = True)
        lines.append([x_exp, y_exp, E_gs, E_ex, E_ex2, E_ex3])
        print(lines[-1])
        
lines_ = np.array(lines)
np.savetxt('surfaces_ax%d_ay%d_bx%d_by%d_cxy%d_cx2y%d.txt'%(ax, ay, bx, by, cxy, cx2y),lines_)
        

