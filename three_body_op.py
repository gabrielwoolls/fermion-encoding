import quimb as qu
import quimb.tensor as qtn

from quimb.tensor.tensor_core import bonds, tensor_contract



def triangle_gate_absorb(
    gate,
    reindex_map,
    vertex_tensors, 
    face_tensor,
    ):
    '''
    #      \ a   b ╱ 
    #      ─●─────●─      
    #       │ \│╱ │       
    #       │  ● c│  
    #      \│ ╱ \ │╱      
    #      ─●─────●─      
    #       │     │ 

    vertex_tensors: sequence of 2 Tensors
        The vertex-site tensors getting acted on,
        shape [2] + [D]*6 (in the lattice bulk).
    
    face_tensor: Tensor, shape [2] + [D]*4
        The face-site tensor.

    gate: Tensor, shape [2]*6 
        Operator to be applied. Should be factorizable
        into shape (2,2, 2,2, 2,2)
    
    reindex_map: dict[physical_inds: gate_bond_inds]
        Maps physical inds (e.g. 'q3') to new index ids
        corresponding to site-to-gate bonds.
    
    **compress_opts: will be passed to `tensor_split()`
        for the main `blob` tensors.
    '''

    t_a, t_b = vertex_tensors
    t_c = face_tensor

    triangle_tensors = {'A': t_a,
                        'B': t_b,
                        'C': t_c}

    triangle_bonds = {'AC': bonds(t_a, t_c),
                      'CB': bonds(t_b, t_c),
                      'AB': bonds(t_a, t_b)}

    # map labels A,B,C to physical indices
    physical_bonds = {k: tuple(ix for ix in t.inds 
        if t.ind_size(ix) == 2)[0] 
        for k, t in triangle_tensors.items()}

    outer_tensors = []
    inner_tensors = []

    rix = (triangle_bonds['AC'], physical_bonds['A'])
    Q_a, R_a = t_a.split(left_inds=None, right_inds=rix,
                         method='qr', get='tensors')
    
    # outer_tensors.append(Q_a)
    inner_tensors.append(R_a.reindex_(reindex_map)) #??
    # R_a.reindex_(reindex_map)

    lix = (triangle_bonds['AC'], physical_bonds['C'])
    L_c, Q_c = t_c.split(left_inds=lix, method='lq',
                        get='tensors')

    # outer_tensors.append(Q_c)
    inner_tensors.append(L_c.reindex_(reindex_map))
    # L_c.reindex_(reindex_map)

    # merge gate, tL and tR tensors into `blob`
    blob = tensor_contract(*inner_tensors, gate)


    lix = bonds(blob, Q_a)
    lix.add(physical_bonds['A'])

    # NOTE: delete maybe_svals?
    U, *maybe_svals, V = blob.split(left_inds=lix,
            get='tensors', bond_ind=triangle_bonds['AC'], 
            **compress_opts)
    
    # contract: Q_a with U, structure as t_a
    #           Q_c with V, structure as t_c
    new_tensors = {'A': tensor_contract(Q_a, U, output_inds=t_a.inds),
                   'C': tensor_contract(V, Q_c, output_inds=t_c.inds)}


    t_b.reindex_(reindex_map) # make sure physical_ind_B only appears once
    rix = (triangle_bonds['CB'], physical_bonds['B'])
    Q_c, R_c = new_tensors['C'].split(left_inds=None, right_inds=rix,
                                method='qr', get='tensors')


    new_tensors['B'] = tensor_contract(t_b, R_c)
    new_tensors['C'] = Q_c

    for k in 'ABC':
        # update to new tensors' data
        triangle_tensors[k].modify(data=new_tensors[k].data)                                
                                    

    





