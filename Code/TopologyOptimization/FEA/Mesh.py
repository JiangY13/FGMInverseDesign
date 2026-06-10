import numpy as np
import torch
class Mesh:
    def __init__(self, mesh, material = None, bc = None):
        self.mesh = mesh
        self.initMesh()
        if(bc != None):
            self.bc = bc
            if self.bc.get('fixed') is not None:
                self.initBC()
            # if self.bc.get('T_out') is not None:
            #     self.initHeatBC()
            # if self.bc.get('fixed') is None and self.bc.get('T_out') is None:
            #     raise Exception('No efficient boundary conditions for fea!')

        if(material != None):
            self.material = material
            if self.material.get('penal') is None:
                self.material['penal'] = 3.

            # if self.material.get('lambda') is not None:
            #     self.init_H()

            if self.material.get('E') is not None:
                self.init_B_C()
                self.initK()

            # if self.material.get('lambda') is None and self.material.get('E') is None:
            #     self.material['lambda'] = 1
            #     self.init_H()
            #     print('Heat transfer test with Young\'s modulus = 1! \n')
            # self.init_LargeScale()
    #-----------------------#
    def initMesh(self):
        self.nelx = self.mesh['nelx']
        self.nely = self.mesh['nely']
        self.nelz = self.mesh['nelz']
        self.elemSize = self.mesh['elemSize']
        self.solid_elems = self.mesh['solid_elems']
        self.void_elems = self.mesh['void_elems']
        self.numElems = self.nelx*self.nely*self.nelz
        self.numNodes = (self.nelx+1)*(self.nely+1)*(self.nelz+1)
        self.elemNodes = np.zeros((self.numElems, 8))
        self.elemArea = self.elemSize[0]*self.elemSize[1]*self.elemSize[2]*torch.ones((self.numElems))
        self.netArea = torch.sum(self.elemArea)
        self.solidArea = self.elemSize[0]*self.elemSize[1]*self.elemSize[2]*len(self.solid_elems)
        self.voidArea = self.elemSize[0]*self.elemSize[1]*self.elemSize[2]*len(self.void_elems)
        elemStructure = np.array(range(self.numElems))
        elemStructure = np.setdiff1d(elemStructure, self.solid_elems)
        self.elemStructure = np.setdiff1d(elemStructure, self.void_elems)
        nxy = (self.nely+1)*(self.nelx+1)
        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely + elx * self.nely + elz*self.nelx*self.nely
                    n1 = (self.nely + 1) * elx + ely
                    n2 = (self.nely + 1) * (elx + 1) + ely
                    self.elemNodes[el, :] = np.array([n1 + 1 + nxy*elz, n2 + 1 + nxy*elz, n2 + nxy*elz, n1 + nxy*elz,
                                                      n1 + 1 + nxy*(elz+1), n2 + 1 + nxy*(elz+1), n2 + nxy*(elz+1), n1 + nxy*(elz+1)])
        self.elemNodes = self.elemNodes.astype(np.int32)
        self.elemCenters = self.generatePoints()

    #-----------------------#
    def initBC(self):
        self.ndof = 3*(self.nelx+1)*(self.nely+1)*(self.nelz+1)
        self.fixed = self.bc['fixed'].astype(np.int32)
        # self.force_dof = self.bc['force_dof']
        self.free = np.setdiff1d(np.arange(self.ndof),self.fixed)
        self.f = self.bc['force'].astype(np.float32)
        self.numDOFPerElem = 24
        self.edofMat=np.zeros((self.numElems, self.numDOFPerElem), dtype=np.int32)
        dofxy = 3 * (self.nely + 1) * (self.nelx + 1)
        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely + elx * self.nely + elz*self.nelx*self.nely
                    n1 = (self.nely + 1) * elx + ely + (self.nely + 1) * (self.nelx + 1) * elz
                    n2 = (self.nely + 1) * (elx + 1) + ely + (self.nely + 1) * (self.nelx + 1) * elz
                    self.edofMat[el, :] = np.array(
                        [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5, 3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5,
                         3 * n2, 3 * n2 + 1, 3 * n2 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                         3 * n1 + 3 + dofxy, 3 * n1 + 4 + dofxy, 3 * n1 + 5 + dofxy, 3 * n2 + 3 + dofxy, 3 * n2 + 4 + dofxy, 3 * n2 + 5 + dofxy,
                         3 * n2 + dofxy, 3 * n2 + 1 + dofxy, 3 * n2 + 2 + dofxy, 3 * n1 + dofxy, 3 * n1 + 1 + dofxy, 3 * n1 + 2 + dofxy])

        self.edofMat = self.edofMat.astype(np.int32)  #1(-,-,-), 2(+,-,-), 3(+,+,-), 4(-,+,-), 5(-,-,+), 6(+,-,+), 7(+,+,+), 8(-,+,+)
        
        self.iK = np.kron(self.edofMat, np.ones((self.numDOFPerElem, 1))).astype(np.int32).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, self.numDOFPerElem))).astype(np.int32).flatten()
        bK = tuple(np.zeros((len(self.iK))).astype(np.int32)) #batch values
        self.nodeIdx = [bK, self.iK, self.jK]

    # def initHeatBC(self):
    #     if self.bc.get('T_out') is not None:
    #         self.T_out, self.elem_out = self.bc['T_out']
    #         # self.T_out = np.unique(self.elemNodes[self.elem_out].ravel())
    #     else:
    #         self.T_out = []
    #     if self.bc.get('T_in') is not None:
    #         T_in, T_in_value = self.bc['T_in']
    #         self.T_in = T_in #  np.unique(self.elemNodes[T_in].ravel())
    #         self.T_inVec = T_in_value * np.ones_like(self.T_in).reshape(-1,1)
    #     else:
    #         self.T_in = []
    #         self.T_inVec = np.array([0,])
    #     if self.bc.get('qn') is not None:
    #         self.q = self.bc['qn']  # Source
    #     else:
    #         self.q = np.zeros((self.numNodes,1))
    #
    #     self.heat_free = np.setdiff1d(np.arange(self.numNodes), self.T_out)
    #     self.heat_free_total = np.setdiff1d(self.heat_free, self.T_in)
    #     self.heat_numDOFPerElem = 8
    #     self.heat_iK = np.kron(self.elemNodes, np.ones((self.heat_numDOFPerElem, 1))).flatten()
    #     self.heat_jK = np.kron(self.elemNodes, np.ones((1, self.heat_numDOFPerElem))).flatten()
    #     bK = tuple(np.zeros((len(self.heat_iK))).astype(np.int32))  # batch values
    #     self.heat_nodeIdx = [bK, self.heat_iK, self.heat_jK]
    #-----------------------#
    # def init_H(self, hx=1, hy=1, hz=1):
    #     h_lambda = self.material['lambda']
    #     Ve = hx*hy*hz
    #     def hex8_Bth(hx=1, hy=1, hz=1):
    #         Bx = np.array([-0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25]) / hx  #x
    #         By = np.array([-0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25]) / hy  #y
    #         Bz = np.array([-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]) / hz  #z
    #         return np.vstack([Bx, By, Bz])  # (3,8)  1(-,-,-), 2(+,-,-), 3(+,+,-), 4(-,+,-), 5(-,-,+), 6(+,-,+), 7(+,+,+), 8(-,+,+)
    #     def lk_H8(k):
    #         A1 = 4*np.eye(2)
    #         A2 = -np.eye(2)
    #         A3 = np.fliplr(A2)
    #         A4 = -np.ones((2,2))
    #         KE1 = np.concatenate((A1, A2, A3, A4), axis = 1)
    #         KE2 = np.concatenate((A2, A1, A4, A3), axis = 1)
    #         KE3 = np.concatenate((A3, A4, A1, A2), axis = 1)
    #         KE4 = np.concatenate((A4, A3, A2, A1), axis = 1)
    #         KE = 1/12 * k * np.concatenate((KE1, KE2, KE3, KE4), axis=0)
    #         return KE
    #     self.H_B = hex8_Bth()  # (3,8)
    #     # Kmat = np.eye(3) * float(h_lambda) if np.isscalar(h_lambda) else np.asarray(h_lambda, float).reshape(3, 3)
    #     H_KE = lk_H8(h_lambda)  # (self.H_B.T @ Kmat @ self.H_B) * Ve  # (8,8)
    #     self.H_KE = np.broadcast_to(H_KE, (self.numElems,)+H_KE.shape)  #np.tile(H_KE[np.newaxis, :, :], (self.numElems, 1, 1))

    def initK(self):
        # def getDMatrix(materialProperty):
        #     E = materialProperty['E']
        #     nu = materialProperty['nu']
        #     A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        #                   [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
        #     A = A.T
        #     B = np.array([[1], [nu]])
        #     k = 1 / 144 * np.dot(A, B)
        #     k = k.flatten()
        #     K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
        #                    [k[1], k[0], k[1], k[3], k[5], k[6]],
        #                    [k[1], k[1], k[0], k[3], k[6], k[5]],
        #                    [k[2], k[3], k[3], k[0], k[7], k[7]],
        #                    [k[4], k[5], k[6], k[7], k[0], k[1]],
        #                    [k[4], k[6], k[5], k[7], k[1], k[0]]])
        #     K2 = np.array([[k[8], k[7], k[11], k[5], k[3], k[6]],
        #                    [k[7], k[8], k[11], k[4], k[2], k[4]],
        #                    [k[9], k[9], k[12], k[6], k[3], k[5]],
        #                    [k[5], k[4], k[10], k[8], k[1], k[9]],
        #                    [k[3], k[2], k[4], k[1], k[8], k[11]],
        #                    [k[10], k[3], k[5], k[11], k[9], k[12]]])
        #     K3 = np.array([[k[5], k[6], k[3], k[8], k[11], k[7]],
        #                    [k[6], k[5], k[3], k[9], k[12], k[9]],
        #                    [k[4], k[4], k[2], k[7], k[11], k[8]],
        #                    [k[8], k[9], k[1], k[5], k[10], k[4]],
        #                    [k[11], k[12], k[9], k[10], k[5], k[3]],
        #                    [k[1], k[11], k[8], k[3], k[4], k[2]]])
        #     K4 = np.array([[k[13], k[10], k[10], k[12], k[9], k[9]],
        #                    [k[10], k[13], k[10], k[11], k[8], k[7]],
        #                    [k[10], k[10], k[13], k[11], k[7], k[8]],
        #                    [k[12], k[11], k[11], k[13], k[6], k[6]],
        #                    [k[9], k[8], k[7], k[6], k[13], k[10]],
        #                    [k[9], k[7], k[8], k[6], k[10], k[13]]])
        #     K5 = np.array([[k[0], k[1], k[7], k[2], k[4], k[3]],
        #                    [k[1], k[0], k[7], k[3], k[5], k[10]],
        #                    [k[7], k[7], k[0], k[4], k[10], k[5]],
        #                    [k[2], k[3], k[4], k[0], k[7], k[1]],
        #                    [k[4], k[5], k[10], k[7], k[0], k[7]],
        #                    [k[3], k[10], k[5], k[1], k[7], k[0]]])
        #     K6 = np.array([[k[13], k[10], k[6], k[12], k[9], k[11]],
        #                    [k[10], k[13], k[6], k[11], k[8], k[1]],
        #                    [k[6], k[6], k[13], k[9], k[1], k[8]],
        #                    [k[12], k[11], k[9], k[13], k[6], k[10]],
        #                    [k[9], k[8], k[1], k[6], k[13], k[6]],
        #                    [k[11], k[1], k[8], k[10], k[6], k[13]]])
        #     A1 = np.concatenate((K1, K2, K3, K4), axis=1)
        #     A2 = np.concatenate((K2.T, K5, K6, K3.T), axis=1)
        #     A3 = np.concatenate((K3.T, K6, K5.T, K2.T), axis=1)
        #     A4 = np.concatenate((K4, K3, K2, K1.T), axis=1)
        #     KE = 1 / ((nu + 1) * (1 - 2 * nu)) * np.concatenate((A1, A2, A3, A4), axis=0)
        #     return (KE)
        # KE_test = getDMatrix(self.material)

        DB = np.einsum("ij,gjk->gik", self.D, self.B)  # (8,6,24)
        # Ke = sum_g B_g[g].T @ D @ B_g[g] * wdet[g]
        # einsum layout: g: gauss, i,j: dofs
        Ke = np.einsum("gki,gkj,g->ij", DB, self.B, self.wdet)
        # Ke = self.B.T @ self.D @ self.B * vol
        self.KE = Ke


    def init_B_C(self):
        nu = self.material['nu']
        E = self.material['E']
        # B = np.array([
        # [-0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0, 0.25,    0,    0, -0.25,     0,     0],
        # [    0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,     0,     0, -0.25,     0,     0, -0.25,     0,    0, 0.25,    0,     0,  0.25,     0],
        # [    0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0, -0.25,     0,     0,  0.25,     0,     0,  0.25,    0,    0, 0.25,     0,     0,  0.25],
        # [-0.25, -0.25,     0, -0.25,  0.25,     0,  0.25,  0.25,     0,  0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, 0.25, 0.25,    0,  0.25, -0.25,     0],
        # [    0, -0.25, -0.25,     0, -0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,    0, 0.25, 0.25,     0,  0.25,  0.25],
        # [-0.25,     0, -0.25, -0.25,     0,  0.25, -0.25,     0,  0.25, -0.25,     0, -0.25,  0.25,     0, -0.25,  0.25,     0,  0.25, 0.25,    0, 0.25,  0.25,     0, -0.25]]
        # ) #[x,y,z,xy,yz,zx], 1(-,+,-) 2(+,+,-) 3(+,-,-) 4(-,-,-) 5(-,+,+) 6(+,+,+) 7(+,-,+) 8(-,-,+)
        self.B, self.wdet = self.hexa8_precompute_B()

        lam = nu / ((1 + nu) * (1 - 2 * nu))
        mu = 1 / (2 * (1 + nu))

        self.D = np.array([
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
            ]) #3D isotropic elasticity matrix D (Voigt: [ex,ey,ez,gxy,gyz,gzx])

        # self.DE = E * self.D
        # self.G = E / (2 * (1 + nu))
        # self.C = np.array(
        #     [[1/E,    -nu/E,  -nu/E,  0,      0,      0],
        #     [-nu/E, 1/E,      -nu/E,  0,      0,      0],
        #     [-nu/E, -nu/E,  1/E,      0,      0,      0],
        #     [0,         0,          0,          1/self.G,  0,      0],
        #     [0,         0,          0,          0,      1/self.G,  0],
        #     [0,         0,          0,          0,      0,      1/self.G]])

        # self.H = np.array([[1,-.5,-.5,0,0,0],
        #                     [-.5,1,-.5,0,0,0],
        #                     [-.5,-.5,1,0,0,0],
        #                     [0,0,0,3,0,0],
        #                     [0,0,0,0,3,0],
        #                     [0,0,0,0,0,3]])  #[xx,yy,zz,yz,xz,xy]

    # def init_LargeScale(self):
    #     res = self.mesh['resolution']
    #     node_idx = np.arange(self.ndof).reshape(self.nelz + 1, self.nelx + 1, self.nely + 1, 3)
    #
    #     mz = (np.arange(self.nelz + 1) % res != 0)[:, None, None, None]
    #     mx = (np.arange(self.nelx + 1) % res != 0)[None, :, None, None]
    #     my = (np.arange(self.nely + 1) % res != 0)[None, None, :, None]
    #
    #     mask_internal = mz & mx & my  # 形状 (Nz, Nx, Ny)
    #     mask_internal_ = np.broadcast_to(mask_internal, (self.nelz + 1, self.nelx + 1, self.nely + 1, 3))
    #
    #     i_idx = node_idx[mask_internal_]  # internal nodes for cell
    #     self.i_idx = np.setdiff1d(i_idx, self.force_dof)  # remove force dof from internal dof to boundary dof
    #     self.b_idx = np.setdiff1d(np.arange(self.ndof), self.i_idx)  # boundary nodes for cell

    def hexa8_precompute_B(self):
        """
        Precompute B matrices and detJ*weight for a regular 8-node brick element.

        Element sizes:
            hx, hy, hz  in x, y, z directions.

        Returns
        -------
        B_g : (8, 6, 24)
            B-matrix at each of the 8 Gauss points.
        wdet : (8,)
            detJ * weight at each Gauss point.
        """
        hx, hy, hz = self.elemSize[0], self.elemSize[1], self.elemSize[2]
        # Natural coordinates of the 8 nodes
        xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
        eta_nodes = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=float) # <== flipped
        zeta_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)

        # 2x2x2 Gauss points in natural coordinates
        gp = 1.0 / np.sqrt(3.0)
        gauss_1d = np.array([-gp, gp], dtype=float)
        XI, ETA, ZETA = np.meshgrid(gauss_1d, gauss_1d, gauss_1d, indexing="ij")

        xi_g = XI.reshape(-1)  # (8,)
        eta_g = ETA.reshape(-1)  # (8,)
        zeta_g = ZETA.reshape(-1)  # (8,)
        n_gp = 8

        # 1) Derivatives in natural coordinates ∂N/∂ξ, ∂N/∂η, ∂N/∂ζ
        #    dN_dxi[g,i]   = 1/8 * xi_i   * (1 + eta_i*eta_g[g]) * (1 + zeta_i*zeta_g[g])
        #    dN_deta[g,i]  = 1/8 * eta_i  * (1 + xi_i*xi_g[g])   * (1 + zeta_i*zeta_g[g])
        #    dN_dzeta[g,i] = 1/8 * zeta_i * (1 + xi_i*xi_g[g])   * (1 + eta_i*eta_g[g])

        dN_dxi = 0.125 * (
                xi_nodes[None, :] *
                (1.0 + eta_nodes[None, :] * eta_g[:, None]) *
                (1.0 + zeta_nodes[None, :] * zeta_g[:, None])
        )  # (ng, 8)

        dN_deta = 0.125 * (
                eta_nodes[None, :] *
                (1.0 + xi_nodes[None, :] * xi_g[:, None]) *
                (1.0 + zeta_nodes[None, :] * zeta_g[:, None])
        )  # (ng, 8)

        dN_dzeta = 0.125 * (
                zeta_nodes[None, :] *
                (1.0 + xi_nodes[None, :] * xi_g[:, None]) *
                (1.0 + eta_nodes[None, :] * eta_g[:, None])
        )  # (ng, 8)

        # 2) Map to physical coordinates using constant diagonal Jacobian:
        dN_dx = (2.0 / hx) * dN_dxi
        dN_dy = (2.0 / hy) * dN_deta
        dN_dz = (2.0 / hz) * dN_dzeta

        # 3) Assemble B matrices for all Gauss points: B_g[g,:,:], shape (8, 6, 24)
        B = np.zeros((n_gp, 6, 24), dtype=float)

        # DOF column indices for each node
        node_cols_x = 3 * np.arange(8)  # ux columns
        node_cols_y = node_cols_x + 1  # uy columns
        node_cols_z = node_cols_x + 2  # uz columns

        # Fill B: Voigt order [ex, ey, ez, gxy, gyz, gzx]
        # ex
        B[:, 0, node_cols_x] = dN_dx
        # ey
        B[:, 1, node_cols_y] = dN_dy
        # ez
        B[:, 2, node_cols_z] = dN_dz
        # gxy
        B[:, 3, node_cols_x] = dN_dy
        B[:, 3, node_cols_y] = dN_dx
        # gyz
        B[:, 4, node_cols_y] = dN_dz
        B[:, 4, node_cols_z] = dN_dy
        # gzx
        B[:, 5, node_cols_x] = dN_dz
        B[:, 5, node_cols_z] = dN_dx

        # 4) detJ * weight for each Gauss point
        # All 8 Gauss points have weight = 1 in 2x2x2 rule, detJ is constant.
        detJ = (hx * hy * hz) / 8.0
        wdet = np.full(n_gp, detJ, dtype=float)

        return B, wdet

    #-----------------------#
    def generatePoints(self,  resolution = 1): # generate points in elements
        ctr = 0
        xy = np.zeros((resolution*self.nelx*resolution*self.nely*resolution*self.nelz,3))

        for k in range(resolution * self.nelz):
            for i in range(resolution * self.nelx):
                for j in range(resolution * self.nely):
                    xy[ctr, 0] = self.elemSize[0] * (i + 0.5) / resolution
                    xy[ctr, 1] = self.elemSize[1] * (resolution * self.nely - j - 0.5) / resolution
                    xy[ctr, 2] = self.elemSize[1] * (k + 0.5) / resolution
                    ctr += 1

        return xy


        
