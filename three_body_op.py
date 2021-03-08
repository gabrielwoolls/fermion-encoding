import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_core import bonds, tensor_contract

import qubit_networks as beeky


def triangle_gate_absorb(
    gate,
    reindex_map,
    vertex_tensors, 
    face_tensor,
    phys_inds,
    gate_tags=('GATE',),
    **compress_opts
    ):
    '''
    Absorbs 3-body operator ``gate`` into a triangle of tensors, 
    like we have in a ``QubitEncodeVector`` TN (i.e. a non-rectangular
    lattice geometry).

    ``gate`` is assumed to act on a 3-tuple (vertex_a, vertex_b, face_c) of
    qubits on the lattice. 

    #      \ a   b ╱ 
    #      ─●─────●─      
    #       │ \│╱ │       
    #       │  ● c│  
    #      \│ ╱ \ │╱      
    #      ─●─────●─      
    #       :     : 

    gate: Tensor, shape [2]*6 
        Operator to be applied. Should be factorizable
        into shape (2,2, 2,2, 2,2)

    reindex_map: dict[physical_inds: gate_bond_inds]
        Maps physical inds (e.g. 'q3') to new index ids
        corresponding to site-to-gate bonds.
    
    vertex_tensors: sequence of 2 Tensors
        The vertex-site tensors getting acted on,
        shape [2] + [D]*6 (in the lattice bulk).

    face_tensor: Tensor, shape [2] + [D]*4
        The face-site tensor.
    
    phys_inds: sequence of str
        The physical "qubit" indices ('qA', 'qB', 'qC')
        at each of the 3 sites.

    gate_tags: sequence of str, optional
        All 3 site tensors will be tagged with these
        after being acted on with `gate`.

    **compress_opts: will be passed to `tensor_split()`
        for the main `blob` tensor. Some keywords are
        
        -method: {'svd', 'lq', 'qr', etc}
            If rank-revealing methods like SVD are 
            used, the site-to-site bonds will not
            be equal size.
        
        -cutoff: float
            The threshold below which to discard singular values. 
            Only applies to rank-revealing methods (not QR or LQ).
        
        -max_bond: int
            Max number of singular values to keep, regardless
            of ``cutoff``.
    '''
    compress_opts.setdefault('method', 'svd')    


    t_a, t_b = vertex_tensors
    t_c = face_tensor

    triangle_tensors = {'A': t_a,
                        'B': t_b,
                        'C': t_c}

    triangle_bonds = {'AC': tuple(bonds(t_a, t_c))[0],
                      'CB': tuple(bonds(t_b, t_c))[0],
                      'AB': tuple(bonds(t_a, t_b))[0]}

    # map labels A,B,C to physical indices
    physical_bonds = {k: tuple(ix for ix in t.inds 
        if ix in phys_inds)[0] 
        for k, t in triangle_tensors.items()}

    inner_tensors = []

    # split 'A' inward
    rix = (triangle_bonds['AC'], physical_bonds['A'])
    Q_a, R_a = t_a.split(left_inds=None, right_inds=rix,
                         method='qr', get='tensors')
    
    # outer_tensors.append(Q_a)
    inner_tensors.append(R_a.reindex_(reindex_map)) 

    # split 'C' inward
    lix = (triangle_bonds['AC'], physical_bonds['C'])
    L_c, Q_c = t_c.split(left_inds=lix, method='lq',
                        get='tensors')

    inner_tensors.append(L_c.reindex_(reindex_map))

    # merge gate, R_a and L_c tensors into `blob`
    blob = tensor_contract(*inner_tensors, gate)

    lix = bonds(blob, Q_a)
    lix.add(physical_bonds['A'])


    U, *maybe_svals, V = blob.split(left_inds=lix,
            get='tensors', bond_ind=triangle_bonds['AC'], 
            **compress_opts)
    
    # Absorb U into Q_a; this is the new tensor at 'A'
    # Absorb V into Q_c (this 'C'-site tensor will be changed) 
    new_tensors = {'A': tensor_contract(Q_a, U, output_inds=t_a.inds),
                   'C': tensor_contract(V, Q_c)} 


    t_b.reindex_(reindex_map) # make sure physical index 'qB' only appears once
    
    rix = bonds(new_tensors['C'], t_b)
    rix.add(physical_bonds['B'])
    Q_c, R_c = new_tensors['C'].split(left_inds=None, right_inds=rix,
                                method='qr', get='tensors',)


    new_tensors['B'] = tensor_contract(t_b, R_c)
    new_tensors['C'] = Q_c

    for k in 'ABC':
        # update the Gate|ket> tensors
        triangle_tensors[k].modify(
            data=new_tensors[k].data,
            inds=new_tensors[k].inds)           

        #add new tags, if any
        for gt in gate_tags: 
            triangle_tensors[k].add_tag(gt)
                                    




def main_test():
    psi = beeky.QubitEncodeVector.rand(3, 3)
    X, Y, Z = (qu.pauli(i) for i in 'xyz')
    where = (3, 4, 9) #which qubits to act on
    numsites = len(where) 
    dp = 2 #phys ind dimension
    gate = X & X & X

    ## take over from here ##
    g_xx = qtn.tensor_1d.maybe_factor_gate_into_tensor(gate, dp, numsites, where) #shape (2,2,2,2) or (2,2,2,2,2,2)

    site_inds = [psi.phys_ind_id.format(q) for q in where]
    bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
    reindex_map = dict(zip(site_inds, bond_inds))

    TG = qtn.Tensor(g_xx, inds=site_inds + bond_inds, 
            left_inds=bond_inds, tags=['GATE'])

    original_ts = [psi[q] for q in where]
    bonds_along = [next(iter(qtn.bonds(t1, t2))) for t1, t2 in qu.utils.pairwise(original_ts)]


    triangle_gate_absorb(gate=TG, reindex_map=reindex_map,
        vertex_tensors=(psi[where[0]], psi[where[1]]), 
        face_tensor=psi[where[2]], phys_inds=site_inds)


if __name__ == '__main__':

    main_test()