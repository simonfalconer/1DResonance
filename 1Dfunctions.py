# ------------------------------------
# Imports
# ------------------------------------
import numpy as np
import matplotlib.pyplot as plt



# ------------------------------------
# Functions
# ------------------------------------
def gaussian(x,alpha,beta):
    return beta*np.exp(-alpha*x**2)

def cos_E(x,E):
    return np.cos(np.sqrt(2*E)*x)



# ------------------------------------
# Basic 1D Operations
# ------------------------------------
def projection(x,psi,basis):
    c = []
    for phi_j in basis:
        c.append(np.trapz(psi*phi_j,x)/np.trapz(phi_j*phi_j,x))
    return c

def lin_comb(coeffs,vectors):
    psi = np.zeros(len(vectors[0]), dtype=complex)
    j = 0
    while j < len(vectors):
        psi += coeffs[j]*vectors[j]
        j+=1
    return psi

def normalize(x,psi):
    norm = np.sqrt(np.trapz(np.conj(psi)*psi, x))
    return psi/norm

def density(psi):
    return np.conj(psi)*psi

def normalized_density(x,psi):
    return np.conj(psi)*psi*np.trapz(np.conj(psi)*psi,x)**(-1)

def local_density(x,low,high,density):
    low_idx = np.argsort((x-low)**2)[0]
    high_idx = np.argsort((x-high)**2)[0]
    return np.trapz(density[low_idx:high_idx],x[low_idx:high_idx])

def local_density_from_psi(x,low,high,psi):
    low_idx = np.argsort((x-low)**2)[0]
    high_idx = np.argsort((x-high)**2)[0]
    return np.trapz(np.conj(psi[low_idx:high_idx])*psi[low_idx:high_idx],x[low_idx:high_idx])

def ovlp(x,left_psi,right_psi):
    return np.trapz(np.conj(left_psi)*right_psi,x)

def T_ij(x,dx,psii,psij):
    # <psi_i|T|psi_j>
    scnd_deriv_psij = (-1/(2*dx**2))*(np.roll(psij, -1) + np.roll(psij, 1) - 2*psij)
    return np.trapz(np.conj(psii)*scnd_deriv_psij,x)

def V_ij(x,potential,psii,psij):
    # <psi_i|V|psi_j>
    return np.trapz(np.conj(psii)*potential*psij,x)

def H_ij(x,dx,potential,psii,psij):
    # <psi_i|H|psi_j>
    return T_ij(x,dx,psii,psij) + V_ij(x,potential,psii,psij)

def timestep(x,t,c_initial,basis_E):
    return c_initial*np.exp(-1.0j*basis_E*t)



# ------------------------------------
# Procedures
# ------------------------------------
def orthogonalize(x,basis):
    # Canonical orthogonalization algorithm
    N = len(basis)
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            S[i,j] = ovlp(x,basis[i],basis[j])     
    s,U = np.linalg.eig(S)
    X = U@np.diag(s**(-0.5))@np.conj(U).T
    ortho_basis = []
    for i in range(N):
        ortho_basis.append(lin_comb(X.T[i],basis))
    return ortho_basis

def gram_schmidt(x,v):
    # Gram-Schmidt orthogonalization algorithm
    u = [v[0]]
    i = 1
    while i < len(v):
        projections = np.zeros_like(v[0])
        for uj in u:
            projections += np.trapz(v[i]*uj,x)/np.trapz(uj*uj,x)*uj
        ui = v[i] - projections
        u.append(ui)
        i+=1
    e = []
    for ui in u:
        e.append(normalize(x,ui))
    return e

def eta_traj(eta_list,H0,W_mat):
    N = len(H0[0])
    traj = []
    for i in H0[0]:
        traj.append([])
    for eta in eta_list:
        if len(traj[0]) == 0:
            Hcap = H0 - 1.0j*eta*W_mat
            Rw,Rv = np.linalg.eig(Hcap)
            Rw = Rw[np.argsort(Rw)]
            i = 0
            while i < N:
                traj[i].append(Rw[i])
                i+=1
        else:
            Hcap = H0 - 1.0j*eta*W_mat
            Rw,Rv = np.linalg.eig(Hcap)
            diff = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    diff[i,j] = np.abs(np.real(Rw[j]) - np.real(traj[i][-1])) + np.abs(np.imag(Rw[j]) - np.imag(traj[i][-1]))
                traj[i].append(Rw[np.argsort(diff[i])[0]])
    return traj

def first_deriv(x,f):
    dx = x[1] - x[0]
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2])/2
    df[0] = f[1] - f[0]
    df[-1] = f[-1] - f[-2]
    return df/dx

def log_vel(eta_list,state_traj):
    return eta_list*first_deriv(eta_list,state_traj)

def corr_log_vel(eta_list,state_traj):
    return 0.5*eta_list**2*first_deriv(eta_list,first_deriv(eta_list,state_traj))

def find_opt(low_eta,high_eta,eta_list,state_traj,corrected=False):
    if corrected == False:
        log_velocity = log_vel(eta_list,state_traj)
        abs_log_velocity = np.abs(log_vel(eta_list,state_traj))
        low_eta_idx = np.argsort(np.abs(eta_list - low_eta))[0]
        high_eta_idx = np.argsort(np.abs(eta_list - high_eta))[0]
        min_log_vel_idx = np.argsort(abs_log_velocity[low_eta_idx:high_eta_idx])[0] + low_eta_idx
        
        opt_eta = eta_list[min_log_vel_idx]
        opt_log_vel = log_velocity[min_log_vel_idx]

        real_E = np.real(state_traj[min_log_vel_idx])
        imag_E = np.imag(state_traj[min_log_vel_idx])
    elif corrected == True:
        corr_log_velocity = corr_log_vel(eta_list,state_traj)
        abs_corr_log_velocity = np.abs(corr_log_velocity)
        low_eta_idx = np.argsort(np.abs(eta_list - low_eta))[0]
        high_eta_idx = np.argsort(np.abs(eta_list - high_eta))[0]
        min_log_vel_idx = np.argsort(abs_corr_log_velocity[low_eta_idx:high_eta_idx])[0] + low_eta_idx

        opt_eta = eta_list[min_log_vel_idx]
        opt_log_vel = corr_log_velocity[min_log_vel_idx]
        
        corr_E = state_traj - log_vel(eta_list,state_traj)
        real_E = np.real(corr_E[min_log_vel_idx])
        imag_E = np.imag(corr_E[min_log_vel_idx])
    else:
        print('"corrected" parameter must be True or False')
    return opt_eta,opt_log_vel,real_E,imag_E

def linear_fitting(x,y):
    x = np.array(x)
    y = np.array(y)

    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y))/(n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x))/n

    y_pred = slope * x + intercept

    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual/ss_total)

    return slope, intercept, r_squared



# ------------------------------------
# Generators
# ------------------------------------
def even_tempered(start,number):
    list = []
    i = 0
    while i < number:
        list.append(start*(1/2)**i)
        i+=1
    return list

def custom_even_tempered(start,number,contraction):
    list = []
    i = 0
    while i < number:
        list.append(start*(1/2)**(i/contraction))
        i+=1
    return list

def kaufmann_continuum(start,number,l):
    al = [0.584342,0.452615,0.382362,0.337027,0.304679]
    bl = [0.424483,0.309805,0.251333,0.215013,0.189944]
    a = al[l]
    b = bl[l]
    c = start*4*(a+b)**2
    list = []
    i = 0
    while i < number:
        list.append(c*(4*(a*(i+1)+b)**2)**(-1))
        i+=1
    return list



# ------------------------------------
# Miscellaneous
# ------------------------------------
def color_spectrum(colors):
    list = []
    delta = 1/colors
    i = 0
    j = 0
    while j < colors:
        list.append(i)
        i+=delta
        j+=1
    spectrum = plt.cm.plasma(list)
    return spectrum


