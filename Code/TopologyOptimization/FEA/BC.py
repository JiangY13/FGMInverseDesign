import numpy as np

def BC_bending(res, len_, elemSize, extra_board):  

    nelx, nely, nelz = int(res * len_[0]) + extra_board[0,0] + extra_board[0,1], int(res * len_[1]) + extra_board[1,0] + extra_board[1,1], int(res * len_[2]) + extra_board[2,0] + extra_board[2,1]
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    force = np.zeros((ndof, 1))

    ### Bending #############
    F_mag = 100 
    force_node_0 = get_nodes('z+x-', [nelx, nely, nelz], extra_board, step=2, type='nodes') 
    offsets = np.array([0, -2, -4, -6])[:, np.newaxis] * (nely + 1) * (nelx + 1) 
    force_node = (offsets + force_node_0).ravel()  
    force[3 * force_node, 0] = F_mag * 1.0 / len(force_node)  

    fixed_nodes_1 = get_nodes('z-x+', [nelx, nely, nelz], extra_board, type='nodes')
    xy_fixed = np.array([0, 1])
    fixed_1 = (3 * fixed_nodes_1[:, None] + xy_fixed[None]).ravel()
    fixed_nodes_2 = get_nodes('z+', [nelx, nely, nelz], extra_board, type='nodes')
    yz_fixed = np.array([1, 2])
    fixed_2 = (3 * fixed_nodes_2[:, None] + yz_fixed[None]).ravel()
    fixed = np.unique(np.concatenate([fixed_1, fixed_2]))

    solid_elems_line = get_nodes('z+x-', [nelx-1, nely-1, nelz-1], extra_board, type='nodes')
    offsets = np.array([0, -1, -2, -3, -4, -5])[:, np.newaxis] * nely * nelx
    solid_elems = (solid_elems_line + offsets).ravel()
    extra_elems = get_nodes('x-', [nelx - 1, nely - 1, nelz - 1], extra_board, type='nodes')
    void_elems = np.setdiff1d(extra_elems, solid_elems)

    mesh = {'type': 'grid', 'nelx': nelx, 'nely': nely, 'nelz': nelz, 'elemSize': elemSize, 'len': len_, 'resolution': res,
            'extra_elems': extra_elems, 'solid_elems': solid_elems, 'void_elems': void_elems}
    bc = {'force': force, 'fixed': fixed}  
    return mesh, bc


def get_grids(nelxyz, len_, res, extra_board=np.array([[0, 0], [0, 0], [0, 0]]).astype(np.int32)):
    nelx, nely, nelz = nelxyz
    step = 1.0 / res  #res_rescale
    v = np.zeros(((nelx+1) * (nely+1) * (nelz+1), 3))
    cen = np.array([-len_[0] / 2 - extra_board[0,0]*step, len_[1] / 2 + extra_board[1,1]*step, -len_[2] / 2 - extra_board[2,0]*step])
    k = 0
    for z in range(nelz+1):
        for x in range(nelx+1):
            for y in range(nely+1):
                v[k, :] = cen + np.array([x, -(y), z]) * step
                k = k + 1
    return v

def get_nodes(boundary_model, nelxyz, extra_board=np.array([[0, 0], [0, 0], [0, 0]]).astype(np.int32), step=1, type='nodes'):
    nelx, nely, nelz = nelxyz

    if boundary_model == 'z-':
        idx_y = np.array(range((nely - extra_board[1, 0] - extra_board[1, 1] + 1))) + extra_board[1, 1]
        idx_x = ((nely + 1)) * (np.array(range(nelx - extra_board[0, 0] - extra_board[0, 1] + 1)) + extra_board[0, 0])
        nodes_idx = (idx_y[0::step][None] + idx_x[0::step][:, None]).flatten()
    elif boundary_model == 'z+':
        idx_y = np.array(range((nely - extra_board[1, 0] - extra_board[1, 1] + 1))) + extra_board[1, 1]
        idx_x = (nely + 1) * (
                    np.array(range(nelx - extra_board[0, 0] - extra_board[0, 1] + 1)) + extra_board[0, 0])
        nodes_idx = (idx_y[0::step][None] + idx_x[0::step][:,None] + (nely + 1) * (nelx + 1) * nelz).flatten()
    elif boundary_model == 'x-':
        idx_y = np.array(range((nely - extra_board[1, 0] - extra_board[1, 1] + 1))) + extra_board[1, 1]
        idx_z = (nelx + 1) * (nely + 1) * (
                    np.array(range(nelz - extra_board[2, 0] - extra_board[2, 1] + 1)) + extra_board[2, 0])
        nodes_idx = (idx_y[0::step][None] + idx_z[0::step][:, None]).flatten()
    elif boundary_model == 'x+':
        idx_y = (np.array(range((nely - extra_board[1, 0] - extra_board[1, 1] + 1))) + extra_board[1, 1]
                 + (nelx * (nely + 1)))
        idx_z = ((nelx + 1) * (nely + 1)) * (
                np.array(range(nelz - extra_board[2, 0] - extra_board[2, 1] + 1)) + extra_board[2, 0])
        nodes_idx = (idx_y[0::step][None] + idx_z[0::step][:, None]).flatten()
    elif boundary_model == 'y-':
        idx_x = (nely + 1) * (np.array(range((nelx - extra_board[0, 0] - extra_board[0, 1] + 1))) + extra_board[0, 0] + 1) - 1
        idx_z = ((nelx + 1) * (nely + 1)) * (
                    np.array(range(nelz - extra_board[2, 0] - extra_board[2, 1] + 1)) + extra_board[2, 0])
        nodes_idx = (idx_x[0::step][None] + idx_z[0::step][:, None]).flatten()
    elif boundary_model == 'y+':
        idx_x = (nely + 1) * (np.array(range((nelx - extra_board[0, 0] - extra_board[0, 1] + 1))) + extra_board[0, 0])
        idx_z = ((nelx + 1) * (nely + 1)) * (
                np.array(range(nelz - extra_board[2, 0] - extra_board[2, 1] + 1)) + extra_board[2, 0])
        nodes_idx = (idx_x[0::step][None] + idx_z[0::step][:, None]).flatten()
    elif boundary_model == 'z-x-' or boundary_model == 'x-z-':
        nodes_idx = np.array(range((nely + 1)))[0::step]
    elif boundary_model == 'z-x+' or boundary_model == 'x+z-':
        nodes_idx = np.array(range((nely + 1)))[0::step] + (nelx * (nely + 1))
    elif boundary_model == 'z+x+' or boundary_model == 'x+z+':
        nodes_idx = np.array(range((nely + 1)))[0::step] + (nelx * (nely + 1)) +  (nelx + 1) * (nely + 1) * nelz
    elif boundary_model == 'z+x-' or boundary_model == 'x-z+':
        nodes_idx = np.array(range((nely + 1)))[0::step] +  (nelx + 1) * (nely + 1) * nelz
    elif boundary_model == 'x-y+' or boundary_model == 'y+x-':
        nodes_idx = (nely + 1) * (nelx + 1) * np.array(range((nelz + 1)))[0::step]
    elif boundary_model == 'x-y-' or boundary_model == 'y-x-':
        nodes_idx = (nely + 1) * (nelx + 1) * np.array(range((nelz + 1)))[0::step] + nely
    elif boundary_model == 'x+y-' or boundary_model == 'y-x+':
        nodes_idx = (nely + 1) * (nelx + 1) * np.array(range((nelz + 1)))[0::step] + nely + (nelx * (nely + 1))
    elif boundary_model == 'x+y+' or boundary_model == 'y+x+':
        nodes_idx = (nely + 1) * (nelx + 1) * np.array(range((nelz + 1)))[0::step] + (nelx * (nely + 1))
    elif boundary_model == 'y-z-' or boundary_model == 'z-y-':
        nodes_idx = (nely + 1) * np.array(range((nelx + 1)))[0::step] + nely
    elif boundary_model == 'y-z+' or boundary_model == 'z+y-':
        nodes_idx = (nely + 1) * np.array(range((nelx + 1)))[0::step] + nely + (nelx + 1) * (nely + 1) * nelz
    elif boundary_model == 'y+z+' or boundary_model == 'z+y+':
        nodes_idx = (nely + 1) * np.array(range((nelx + 1)))[0::step] + (nelx + 1) * (nely + 1) * nelz
    elif boundary_model == 'y+z-' or boundary_model == 'z-y+':
        nodes_idx = (nely + 1) * np.array(range((nelx + 1)))[0::step]
    else:
        raise Exception('No such boundary model!')
    if type == 'nodes':
        return nodes_idx.astype(np.int32)
    elif type == 'dofs':
        dofs = np.array(range(3))
        dofs_idx = (3 * nodes_idx[:, None] + dofs[None]).flatten()
        return dofs_idx.astype(np.int32)
    else:
        dofs = np.array(range(3))
        dofs_idx = (3 * nodes_idx[:, None] + dofs[None]).flatten()
        return nodes_idx.astype(np.int32), dofs_idx.astype(np.int32)
