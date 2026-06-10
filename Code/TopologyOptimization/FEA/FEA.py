import numpy as np
from numpy.ma.extras import setdiff1d
from scipy.sparse import coo_matrix
import cvxopt
import cvxopt.cholmod
from torch.distributions import VonMises

from .Mesh import Mesh
from .Km2Compliance import *


class FE:
    def __init__(self, mesh, matProp, bc, device='cuda'):
        self.mesh = Mesh(mesh, matProp, bc)
        if self.mesh.material.get('E') is not None:
            self.init_Matrix_idx(device)

    def init_Matrix_idx(self, device):
        self.Ksize = self.mesh.ndof - self.mesh.fixed.flatten().shape[0]
        row_indices_np = self.mesh.iK  # NumPy array of row indices
        col_indices_np = self.mesh.jK  # NumPy array of column indices'

        keep_index = self.mesh.free
        mask = ~(np.isin(row_indices_np, self.mesh.fixed) | np.isin(col_indices_np, self.mesh.fixed))

        self.valid_mask = torch.from_numpy(mask).bool().to(device)

        ## ** Need to make sure Array not out of bounds ** ##
        self.indexMap = -np.ones(self.mesh.ndof, dtype=np.int32)
        self.indexMap[keep_index] = np.arange(0, self.Ksize, dtype=np.int32)

        self.new_row_indices = self.indexMap[row_indices_np.astype(np.int32)][mask]
        self.new_col_indices = self.indexMap[col_indices_np.astype(np.int32)][mask]

        self.f = torch.tensor(self.mesh.f[keep_index, 0], dtype=torch.float32, device=device)

    def solve_c(self, density, eps=1e-3):
        # self.u = torch.zeros((self.mesh.ndof, 1), device=density.device)

        E = self.mesh.material['E'] * (eps + density) ** self.mesh.material['penal']
        KE = torch.tensor(self.mesh.KE, dtype=torch.float32, device=density.device)
        sK = torch.einsum('i,jk->ijk', E, KE).flatten()
        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        c = sk2c(d, (self.new_row_indices, self.new_col_indices), f, self.f, self.Ksize)
        return c


    def solve_stress(self, density, eps=1e-3):
        ## isotropic

        self.u = torch.zeros((self.mesh.ndof, 1), dtype=torch.float32, device=density.device)
        E = self.mesh.material['E']*(eps + density) ** self.mesh.material['penal']
        KE = torch.tensor(self.mesh.KE, dtype=torch.float32, device=density.device)
        sK = torch.einsum('i,jk->ijk', E, KE).flatten()
        B = torch.tensor(self.mesh.B, dtype=torch.float32, device=density.device)
        D = torch.tensor(self.mesh.D, dtype=torch.float32, device=density.device)
        E_q = self.mesh.material['E'] * (eps + density) ** (self.mesh.material['penal']-0.5)
        DE = torch.einsum('i,jk->ijk', E_q, D)  #torch.tensor(self.mesh.DE, dtype=torch.float32, device=density.device)  #.expand(self.mesh.numElems,-1,-1)
        # H = torch.tensor(self.mesh.H, dtype=torch.float32, device=density.device).expand(self.mesh.numElems,-1,-1)
        #i = self.sparseKIdx
        d = sK[self.valid_mask]

        f = self.mesh.f[self.mesh.free, 0]
        u = sk2u(d, (self.new_row_indices, self.new_col_indices), f, self.f, self.Ksize)
        self.u[self.mesh.free, 0] = u
        c = (self.f * u).sum()
        uElem = self.u[self.mesh.edofMat].reshape(self.mesh.numElems, self.mesh.numDOFPerElem)
        sigmaElem = torch.einsum('bij,gjk,bk -> bgi', DE, B, uElem) #[x,y,z,xy,yz,zx]

        VM = 0.5 * ((sigmaElem[..., 0] - sigmaElem[..., 1]) ** 2 + (sigmaElem[..., 1] - sigmaElem[..., 2]) ** 2
                                     + (sigmaElem[..., 2] - sigmaElem[..., 0]) ** 2 +
                                     6 * torch.sum(sigmaElem[..., 3:] ** 2, dim=-1))
        VonMisesElem = torch.sqrt(torch.clamp(VM, min=0.0)).mean(-1)

        return VonMisesElem, c


    def deleterowcol(self, A, delrow, delcol):
        m = A.shape[0]
        keep = np.delete(np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete(np.arange(0, m), delcol)
        A = A[:, keep]
        return A


def hex8_face_N_integral(face, hx=1, hy=1, hz=1):
    """
    return w
    w = hex8_face_N_integral('z+')
    f_e_face = w * q_n      # (8,)
    """
    A = {'x-': hy * hz, 'x+': hy * hz, 'y-': hx * hz, 'y+': hx * hz, 'z-': hx * hy, 'z+': hx * hy}[face]
    w = np.zeros(8)
    if face == 'x-':   ids = [0, 3, 4, 7]  # 1,4,5,8
    if face == 'x+':   ids = [1, 2, 5, 6]  # 2,3,6,7
    if face == 'y-':   ids = [0, 1, 4, 5]  # 1,2,5,6
    if face == 'y+':   ids = [2, 3, 6, 7]  # 3,4,7,8
    if face == 'z-':   ids = [0, 1, 2, 3]  # 1,2,3,4
    if face == 'z+':   ids = [4, 5, 6, 7]  # 5,6,7,8
    w[ids] = A / 4.0
    return w  #shape (8,)

def hex8_face_dNdn_integral(B, face, hx=1, hy=1, hz=1):
    """
    return g
    face ∈ {'x-','x+','y-','y+','z-','z+'}
    """
    if face == 'x-':
        n = -1.0
        d = B[0]
        A = hy * hz
    elif face == 'x+':
        n = +1.0
        d = B[0]
        A = hy * hz
    elif face == 'y-':
        n = -1.0
        d = B[1]
        A = hx * hz
    elif face == 'y+':
        n = +1.0
        d = B[1]
        A = hx * hz
    elif face == 'z-':
        n = -1.0
        d = B[2]
        A = hx * hy
    elif face == 'z+':
        n = +1.0
        d = B[2]
        A = hx * hy
    else:
        raise ValueError("face must be one of {'x-','x+','y-','y+','z-','z+'}")

    return (n * d) * A  # shape (8,)






