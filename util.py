import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def make_random_tr_ham(N):
    sy = np.kron(np.eye(N // 2), np.array([[0, -1j], [1j, 0]]))
    h = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    h += h.T.conj()
    Th = sy @ h.conj() @ sy
    return (h + Th) / 4

def make_random_tr_unitary(N):
    return expm(1j*make_random_tr_ham(N))

def create_multiport_grid_unitary(U, m, n):
    G = np.zeros((m*n*4, m*n*4), dtype=complex)
    for i1 in range(m):
        for j1 in range(n):
            idx1 = 4*(i1*n + j1)
            for w, (i2, j2) in enumerate([(i1, j1+1), (i1+1, j1), (i1, j1-1), (i1-1, j1)]):
                idx2 = (4 * (i2*n + j2)) % (m*n*4)
                G[idx2:idx2+4, idx1+w] = U[:, w]
    return G

def get_floquet_eig(U, kx, ky):
    M = np.diag([np.exp(1j*ky), np.exp(1j*kx), np.exp(-1j*ky), np.exp(-1j*kx)])
    return np.linalg.eig(U @ M)

def get_momentum_eigenvals(U, kx, ky):
    return np.real(-1j*np.log(get_floquet_eig(U, kx, ky)[0]))

def plot_dispersion_relation(U, n, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    vfunc = np.vectorize(lambda k_x, k_y: get_momentum_eigenvals(U, k_x, k_y)[n])
    E = vfunc(KX, KY)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(KX, KY, E, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E')
    plt.show()

def get_scatter_dispersion_relation(U, res=50):
    E = np.array([])
    KX = np.array([])
    KY = np.array([])
    for kx in np.linspace(-np.pi, np.pi, res):
        for ky in np.linspace(-np.pi, np.pi, res):
            KX = np.append(KX, [kx, kx, kx, kx])
            KY = np.append(KY, [ky, ky, ky, ky])
            E = np.append(E, get_momentum_eigenvals(U, kx, ky))
    return KX, KY, E

def scatter_plot_dispersion_relation(U, res=50):
    KX, KY, E = get_scatter_dispersion_relation(U)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(KX, KY, E)
    plt.show()

class TransitionPlot:
    def __init__(self, U1, U2, res=50):
        self._U1 = U1
        self._U2 = U2
        self._res = res

        KX, KY, E = get_scatter_dispersion_relation(U1, res)

        plt.subplots_adjust(left=0.25, bottom=0.25)
        fig = plt.figure()
        ax = Axes3D(fig)
        self._scatter = ax.scatter(KX, KY, E, s=2)

        axtime = plt.axes([0.25, 0.15, 0.65, 0.03])
        time = Slider(axtime, 'Time', 0, 1, valinit=0)
        time.on_changed(lambda t: self._update(t))
        
        plt.show()

    def _update(self, t):
        KX, KY, E = get_scatter_dispersion_relation(self._U1*(1-t)+self._U2*t, self._res)
        self._scatter._offsets3d = KX, KY, E
        plt.draw()

def plot_spectrum(U, res=200):
    E = np.array([])
    for kx in np.linspace(-np.pi, np.pi, res):
        for ky in np.linspace(-np.pi, np.pi, res):
            E = np.append(E, get_momentum_eigenvals(U, kx, ky))

    fig, ax = plt.subplots()
    ax.hist(E, density=True, bins=100)

    plt.show()

def get_gradient(U, eigvecs, kx, ky):
    Mx = np.diag([0, 1j*np.exp(1j*kx), 0, -1j*np.exp(-1j*kx)])
    My = np.diag([1j*np.exp(1j*ky), 0, -1j*np.exp(-1j*ky), 0])

    dx = np.diagonal(eigvecs.T.conj() @ U @ Mx @ eigvecs)
    dy = np.diagonal(eigvecs.T.conj() @ U @ My @ eigvecs)

    return dx, dy

"""
#https://mathoverflow.net/questions/229425/derivative-of-eigenvectors-of-a-matrix-with-respect-to-its-components
def get_eig_gradient(U, eigvals, eigvecs, kx, ky):
    Mx = np.diag([0, 1j*np.exp(1j*kx), 0, -1j*np.exp(-1j*kx)])
    My = np.diag([1j*np.exp(1j*ky), 0, -1j*np.exp(-1j*ky), 0])

    result = np.zeros((4, 4, 2), dtype=complex)

    for i in range(4):
        for j in range(4):
            if i==j:
                continue
            k = 1/(eigvals[i]-eigvals[j])
            result[i, :, 0] += np.vdot(U @ Mx @ eigvecs[i], eigvecs[j])*k*eigvecs[j]
            result[i, :, 1] += np.vdot(U @ My @ eigvecs[i], eigvecs[j])*k*eigvecs[j]
    return result
"""

class DispersionRelation():
    def __init__(self, U, res):
        self._U = U
        self._res = res
        self._KX, self._KY, self._eigvals, self._eigvecs = get_dispersion_meshgrid(U, res)
        self._kx = []
        self._ky = []
        self._evl = []
        self._evc = []

    def eig(self, kx, ky):
        x_idx = int(round((self._res-1)*(kx+np.pi)/(2*np.pi)))
        y_idx = int(round((self._res-1)*(ky+np.pi)/(2*np.pi)))
        eigvals, eigvecs = get_floquet_eig(self._U, kx, ky)
        eigvals, eigvecs = eig_adj(self._eigvals[x_idx, y_idx], self._eigvecs[x_idx, y_idx], eigvals, eigvecs)
        #M = np.diag([np.exp(1j*ky), np.exp(1j*kx), np.exp(-1j*ky), np.exp(-1j*kx)])
        #for i in range(4):
        #    if not np.allclose((((self._U@M)@eigvecs)/eigvecs)[i], eigvals):
        #        print(eigvals)
        #        print(eigvecs)
        #        raise Exception("nope")
        self._kx.append(kx)
        self._ky.append(ky)
        self._evl.append(np.real(-1j*np.log(eigvals))[0])
        self._evc.append(eigvecs)
        return eigvals, eigvecs

def get_floquet_eig_sorted(U, kx, ky):
    eigvals, eigvecs = get_floquet_eig(U, kx, ky)
    idx = np.real(-1j*np.log(eigvals)).argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    return eigvals, eigvecs

def get_eig_gradient(disp, kx, ky, dk=0.0000001):
    result = np.zeros((4, 4, 2), dtype=complex)
    dn_dkx = (disp.eig(kx+dk, ky)[1]-disp.eig(kx-dk, ky)[1])/(2*dk)
    dn_dky = (disp.eig(kx, ky+dk)[1]-disp.eig(kx, ky-dk)[1])/(2*dk)
    result[:, :, 0] = dn_dkx
    result[:, :, 1] = dn_dky
    return result

def get_berry_connection(disp, kx, ky):
    result = np.zeros((4, 2))
    eigvecs = disp.eig(kx, ky)[1]
    for i in range(4):
        result[i, :] = np.real(1j * eigvecs[:, i].conj() @ get_eig_gradient(disp, kx, ky)[:, i, :])
    return result

def get_berry_curvature(disp, kx, ky, dk=0.000001):
    dA_dkx = (get_berry_connection(disp, kx+dk, ky)-get_berry_connection(disp, kx-dk, ky))/(2*dk)
    dA_dky = (get_berry_connection(disp, kx, ky+dk)-get_berry_connection(disp, kx, ky-dk))/(2*dk)
    return dA_dkx[:, 1]-dA_dky[:, 0]

def get_berry_curvature2(disp, kx, ky):
    grad = get_eig_gradient(disp, kx, ky)
    result = np.zeros(4)
    for i in range(4):
        for j in range(4):
            if i==j:
                continue
            v1 = grad[:, i].T.conj() @ disp.eig(kx, ky)[1][:, j]
            v2 = disp.eig(kx, ky)[1][:, j].conj() @ grad[:, i]
            result[i] += -np.imag((v1[0]*v2[1])-(v1[1]*v2[0]))
    return result

def plot_berry_connection(U, n, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    disp = DispersionRelation(U, 50)

    vfunc = np.vectorize(lambda k_x, k_y: get_berry_connection(disp, k_x, k_y), signature='(),()->(4,2)')
    berry_connection = vfunc(KX, KY)

    fig, ax = plt.subplots()
    ax.quiver(KX, KY, berry_connection[:, :, n, 0], berry_connection[:, :, n, 1])
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    plt.show()

def plot_berry_curvature(U, n, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    disp = DispersionRelation(U, 50)

    vfunc = np.vectorize(lambda k_x, k_y: get_berry_curvature2(disp, k_x, k_y), signature='(),()->(4)')
    berry_curvature = vfunc(KX, KY)

    fig, ax = plt.subplots()
    ax = Axes3D(fig)
    ax.plot_surface(KX, KY, berry_curvature[:, :, n], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Omega')
    plt.show()

def get_chern_number(U, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    disp = DispersionRelation(U, res)

    vfunc = np.vectorize(lambda k_x, k_y: get_berry_curvature2(disp, k_x, k_y), signature='(),()->(4)')
    berry_curvature = vfunc(KX, KY)

    return (-2*np.pi*np.sum(berry_curvature, axis=(0,1)))/(res*res)

"""
def eig_adj(eigvals1, dx1, dy1, eigvals2, dx2, dy2):
    eig2_sorted = np.zeros(np.shape(eigvals2), dtype=complex)
    dx2_sorted = np.zeros(np.shape(dx2), dtype=complex)
    dy2_sorted = np.zeros(np.shape(dy2), dtype=complex)
    for i, (E, dx, dy) in enumerate(zip(eigvals1, dx1, dy1)):
        dists = np.arcsin(np.sin(np.abs(0.5*(np.angle(E)-np.angle(eigvals2)))))
        threshold = dists/max(1e-10, min(dists)) < 2

        closest_E, closest_dx, closest_dy = eigvals2[threshold], dx2[threshold], dy2[threshold]
        gradient_dists = np.sqrt(((dx-closest_dx)**2)+((dy-closest_dy)**2))

        E2 = closest_E[np.argmin(gradient_dists)]
        dx2_sorted = closest_dx[np.argmin(gradient_dists)]
        dy2_sorted = closest_dy[np.argmin(gradient_dists)]

        eig2_sorted[i] = E2
        dx2 = dx2[eigvals2 != E2]
        dy2 = dy2[eigvals2 != E2]
        eigvals2 = eigvals2[eigvals2 != E2]
    
    return eig2_sorted, dx2_sorted, dy2_sorted
"""

def eig_adj(eigvals1, eigvecs1, eigvals2, eigvecs2):
    min_dist = 1e20
    indices = [0, 1, 2, 3]
    for p in itertools.permutations(indices):
        eigval = eigvals2[list(p)]
        eigvec = eigvecs2[:, list(p)]
        dist = np.linalg.norm(eigval-eigvals1)
        if dist < min_dist:
            min_dist = dist
            eigvals2_sorted = eigval
            eigvecs2_sorted = eigvec

    return eigvals2_sorted, eigvecs2_sorted

def plot_gradient(U, n, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    vfunc = np.vectorize(lambda k_x, k_y: get_floquet_eig(U, k_x, k_y), signature='(),()->(4),(4,4)')
    eigvals, eigvecs = vfunc(KX, KY)

    vfunc = np.vectorize(lambda eigv, k_x, k_y: get_gradient(U, eigv, k_x, k_y), signature='(4,4),(),()->(4),(4)')
    DX, DY = vfunc(eigvecs, KX, KY)

    fig = plt.figure()
    ax = Axes3D(fig)
    W = -np.ones((res, res))
    ax.quiver(KX, KY, np.real(-1j*np.log(eigvals[:, :, 0])), np.imag(eigvals[:, :, 0].conj()*DX[:, :, 0]), np.imag(eigvals[:, :, 0].conj()*DY[:, :, 0]), W, length=0.5)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E')
    plt.show()

"""
def plot_dispersion_matched(U, n, res=50):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    vfunc = np.vectorize(lambda k_x, k_y: get_floquet_eig(U, k_x, k_y), signature='(),()->(4),(4,4)')
    eigvals, eigvecs = vfunc(KX, KY)

    vfunc = np.vectorize(lambda eigv, k_x, k_y: get_gradient(U, eigv, k_x, k_y), signature='(4,4),(),()->(4),(4)')
    dx, dy = vfunc(eigvecs, KX, KY)

    eigvals_out = np.zeros((res, res, 4), dtype=complex)
    dx_out = np.zeros((res, res, 4), dtype=complex)
    dy_out = np.zeros((res, res, 4), dtype=complex)

    eigvals_out[0][0] = eigvals[0][0]
    dx_out[0][0] = dx[0][0]
    dy_out[0][0] = dy[0][0]

    for i in range(1, res):
        eigvals_out[i][0], dx_out[i][0], dy_out[i][0] = eig_adj(eigvals_out[i-1][0], dx_out[i-1][0], dy_out[i-1][0], eigvals[i][0], dx[i][0], dy[i][0])

    for j in range(1, res):
        for i in range(0, res):
            eigvals_out[i][j], dx_out[i][j], dy_out[i][j] = eig_adj(eigvals_out[i][j-1], dx_out[i][j-1], dy_out[i][j-1], eigvals[i][j], dx[i][j], dy[i][j])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(KX, KY, np.real(-1j*np.log(eigvals_out[:, :, n])), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E')
    plt.show()
"""

def get_dispersion_meshgrid(U, res):
    kx = np.linspace(-np.pi, np.pi, res)
    ky = np.linspace(-np.pi, np.pi, res)
    KX, KY = np.meshgrid(kx, ky)

    vfunc = np.vectorize(lambda k_x, k_y: get_floquet_eig(U, k_x, k_y), signature='(),()->(4),(4,4)')
    eigvals, eigvecs = vfunc(KX, KY)

    eigvals_out = np.zeros((res, res, 4), dtype=complex)
    eigvecs_out = np.zeros((res, res, 4,  4), dtype=complex)
    eigvals_out[0][0] = eigvals[0][0]
    eigvecs_out[0][0] = eigvecs[0][0]

    for i in range(1, res):
        eigvals_out[i][0], eigvecs_out[i][0] = eig_adj(eigvals_out[i-1][0], eigvecs_out[i-1][0], eigvals[i][0], eigvecs[i][0])

    for j in range(1, res):
        for i in range(0, res):
            eigvals_out[i][j], eigvecs_out[i][j] = eig_adj(eigvals_out[i][j-1], eigvecs_out[i][j-1], eigvals[i][j], eigvecs[i][j])

    return KX, KY, eigvals_out, eigvecs_out

def plot_dispersion_matched(U, n_array=[0, 1, 2, 3], res=50):
    KX, KY, eigvals, _ = get_dispersion_meshgrid(U, res)

    fig = plt.figure()
    ax = Axes3D(fig)
    for n in n_array:
        ax.plot_surface(KX, KY, np.real(-1j*np.log(eigvals[:, :, n])), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E')
    plt.show()

"""
def plot_berry_connection(U, n, res=50):
    KX, KY, eigvals, eigvecs = get_dispersion_meshgrid(U, res)

    vfunc = np.vectorize(lambda evl, evc, k_x, k_y: get_berry_connection(U, evl, evc, k_x, k_y), signature='(4),(4,4),(),()->(4,2)')
    berry_connection = vfunc(eigvals, eigvecs, KX, KY)

    fig, ax = plt.subplots()
    ax.quiver(KX, KY, berry_connection[:, :, n, 0], berry_connection[:, :, n, 1])
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    plt.show()

def get_chern_number(U, n, res=50):
    KX, KY, eigvals, eigvecs = get_dispersion_meshgrid(U, res)

    vfunc = np.vectorize(lambda evl, evc, k_x, k_y: get_berry_connection(U, evl, evc, k_x, k_y), signature='(4),(4,4),(),()->(4,2)')
    berry_connection = vfunc(eigvals, eigvecs, KX, KY)

    bottom = sum(berry_connection[:, 0, n, 0])
    right = sum(berry_connection[res-1, :, n, 1])
    top = -sum(berry_connection[:, res-1, n, 0])
    left = -sum(berry_connection[0, :, n, 1])

    return (bottom)/res
"""
