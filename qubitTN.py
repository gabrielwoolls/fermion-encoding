import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import spinlessQubit
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
    

def make_qubit_TN(Lx, Ly, chi=5, default_spin='up', show_graph=False):
    '''
    Make a qubit TensorNetwork from local states in `arrays`, i.e.
    the TN will be in a product state.

    arrays: sequence of arrays, optional
        Specify local qubit states in vertices, faces
    '''
    
    #default to `up` spin at every site
    vert_array = [[qu.up().reshape(2) 
                    for j in range(Ly)]
                    for i in range(Lx)]
        
    face_array = [[qu.plus().reshape(2) 
                    for j in range(Ly-1)]
                    for i in range(Lx-1)]


    vtensors = [[qtn.Tensor(data = vert_array[i][j], 
                            inds = [f'q{i*Ly+j}'],
                            tags = {f'Q{i*Ly+j}', 'VERT'}) 
                for j in range(Ly)] 
                for i in range(Lx)]


    ftensors = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):
        if i%2 == j%2:
            ftensors[i,j] = qtn.Tensor( data=face_array[i][j],
                                        inds=[f'q{k+Lx*Ly}'],
                                        tags={f'Q{k+Lx*Ly}','FACE'})
            k+=1

    

    for i,j in product(range(Lx), range(Ly)):
        # vtensors[i][j].new_ind(f'q{i*Ly+j}',size=2)
        # vtensors[i][j].add_tag(f'Q{i*Ly+j}')

        if i<=Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=chi)
        if j<=Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=chi)


    for i, j in product(range(Lx-1), range(Ly-1)):
        if not ftensors[i,j] is None:
            # ftensors[i,j].new_ind(f'q{k+Lx*Ly}',size=2)
            # ftensors[i,j].add_tag(f'Q{k+Lx*Ly}')

            ftensors[i,j].new_bond(vtensors[i][j],size=chi)
            ftensors[i,j].new_bond(vtensors[i][j+1],size=chi)
            ftensors[i,j].new_bond(vtensors[i+1][j+1],size=chi)
            ftensors[i,j].new_bond(vtensors[i+1][j],size=chi)


    alltensors = vtensors + [f for f in ftensors.flatten().tolist() if not f is None]
    vtn = qtn.TensorNetwork(alltensors)

    if show_graph:
        LAT_CONST = 50 #lattice constant for graphing

        fix = {
            **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
            }
        vtn.graph(color=['VERT','FACE'], show_tags=True, fix=fix)

    return vtn



class MyQubitTN():

    def __init__(self, Lx, Ly, chi, default_spin='up'):
        '''

        _edge_map: dict[string --> list]
            Gives list of edges for each direction  
            in {'u','d','r','l'}
        
        _psi: TensorNetwork 
            State of the qubit lattice
        
        _vert_coo_map: dict[tuple(int) --> int]
        '''
        
        verts, faces = spinlessQubit.gen_lattice_sites(Lx,Ly)
        
        tensor_net = make_qubit_TN(Lx, Ly, chi, default_spin)
        
        self._vert_coo_map = np.ndenumerate(verts)
        self._face_coo_map = np.ndenumerate(faces)
        
        self._psi = tensor_net

        self._Lx = Lx
        self._Ly = Ly

        self._Nverts = verts.size
        self._Nsites = verts.size + faces[faces!=None].size
        self._d_physical = 2


        self._edge_map = spinlessQubit.get_edge_map(verts, faces)
        
        self._site_tag_id = 'Q{}'
        self._phys_ind_id = 'q{}'

        
        #If we subclass TensorNetwork -- should we?
        # super().__init__(tensors)


    def vert_coo_map(self, i, j):
        return self._vert_coo_map[(i,j)]
    
    def face_coo_map(self, i, j):
        return self._face_coo_map[(i,j)]

    

    def vertex_coo_tag(self,i,j):
        return f'Q{self.vert_coo_map(i,j)}'
    
    def vertex_coo_ind(self,i,j):
        return f'q{self.vert_coo_map(i,j)}'



    def face_coo_tag(self,i,j):
        return f'Q{self.face_coo_map(i,j)}'

    def face_coo_ind(self,i,j):
        return f'q{self.face_site_map(i,j)}'


    def get_edges(self, key):
        '''
        Returns: list[tuple(int or None)]
            List of (three-tuple) edges,
            where edge (i,j,f) denotes vertices i,j 
            (ints) and face f  (int or None)

        key: {'u','d','r','l','r+l'}
            String specifying what edges of
            the graph to return.

        '''
        if key == 'r+l':
            return self._edge_map['r']+self._edge_map['l']
        
        else:
            return self._edge_map[key]



    def graph_psi(self, show_tags=False, auto=False):
        
        if auto:
            self._psi.graph(color=['VERT','FACE','GATE'])
        
        else:
            LAT_CONST = 50 #lattice constant for graphing
            Lx,Ly = self._Lx, self._Ly

            fix = {
                **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
                }
            self._psi.graph(color=['VERT','FACE','GATE'], show_tags=show_tags, fix=fix)


    def to_dense(self):
        '''Return self._psi as dense vector, i.e. a qarray with 
        shape (-1, 1)
        '''
        inds_seq = (f'q{i}' for i in range(self._Nsites))
        return self._psi.to_dense(inds_seq).reshape(-1,1)


    def apply_gate(self, psi, G, where, inplace=False):
        '''
        TODO: incorporate `physical_ind_id`?
        
        Apply gate `G` at sites specified in `where`,
        preserving physical indices.

        Params:
        psi: TensorNetwork
            
        G : array
            Gate to apply, should be compatible with 
            shape ``(.....)``
        
        where: sequence of ints
            The sites on which to act, using the 
            custom numbering that includes face 
            and vertex sites.
        '''
        # psi = self._psi if inplace else self._psi.copy()
        psi = psi if inplace else psi.copy()

        if isinstance(where, int): 
            where = (where,)

        numsites = len(where) #gate is `numsites`-local
        dp = self._d_physical

        G = maybe_factor_gate_into_tensor(G, dp, numsites, where)

        #new physical indices
        site_inds = [f'q{i}' for i in where] 
        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))

        TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=['GATE'])
        
        psi.reindex_(reindex_map)
        psi |= TG
        return psi


    def make_norm_tensor(self, psi=None):
        '''
        Return <psi|psi> as a TensorNetwork.
        '''

        if psi is None:
            psi = self._psi

        ket = psi.copy()
        ket.add_tag('KET')

        bra = ket.retag({'KET':'BRA'})
        bra.conj_()

        return ket | bra


    def compute_hop_expecs(self, psi=None):
        '''
        TODO: debug
        
        Return <psi|H_hop|psi> expectation for the
        hopping terms in (qubit) Hubbard
        '''
        if psi is None: 
            psi = self._psi

        E_hop = 0
        bra = psi.H

        X,Y,Z = (qu.pauli(mu) for mu in ['x','y','z'])

        #Horizontal edges
        for direction in ['r','l']:

            Of = Y #face operator

            for (i,j,f) in self.get_edges(direction):
                if f is None:
                    G = ((X&X) + (Y&Y))/2
                    G_ket = self.apply_gate(psi, G, where=(i,j))
            
                else:
                    G = ((X&X&Of) + (Y&Y&Of))/2
                    G_ket = self.apply_gate(psi, G, where=(i,j,f))

                E_hop += (bra|G_ket) ^ all


        #Vertical edges (sign changes!)
        for direction, sign in [('u', -1), ('d', 1)]:

            Of = X #face operator 

            for (i,j,f) in self.get_edges(direction):
                if f is None:
                    G = ((X&X) + (Y&Y))/2
                    G_ket = self.apply_gate(psi, G, where=(i,j))
            
                else:
                    G = ((X&X&Of) + (Y&Y&Of))/2
                    G_ket = self.apply_gate(psi, G, where=(i,j,f))

                E_hop += sign * (bra|G_ket) ^ all

        return E_hop

        # for (i,j,f) in self.get_edges('r+l'):
        #     Of = Y #operator to act on face qbit if it exists

        #     if f is None:
        #         G = 0.5 * sum(X&X, Y&Y)
        #         G_ket = self.apply_gate(psi, G, where=(i,j))
            
        #     else:
        #         G = 0.5 * sum(X&X&Of, Y&Y&Of)
        #         G_ket = self.apply_gate(psi, G, where=(i,j,f))

        #     # print(bra)
        #     # print(G_ket)
        #     E_hop += (bra|G_ket) ^ all
        
        ## DOWN

        # for (i,j,f) in self.get_edges('d'):
        #     Of = X #operator to act on face qbit if it exists
            
        #     if f is None:
        #         G = 0.5 * sum(X&X, Y&Y)
        #         G_ket = self.apply_gate(psi, G, where=(i,j))
            
        #     else:
        #         G = 0.5 * sum(X&X&Of, Y&Y&Of)
        #         G_ket = self.apply_gate(psi, G, where=(i,j,f))

        #     E_hop += (bra|G_ket) ^ all

        ## UP

        # for (i,j,f) in self.get_edges('u'):
        #     Of = X #operator to act on face qbit if it exists
            
        #     if f is None:
        #         G = 0.5 * sum(X&X, Y&Y)
        #         G_ket = self.apply_gate(psi, G, where=(i,j))
            
        #     else:
        #         G = 0.5 * sum(X&X&Of, Y&Y&Of)
        #         G_ket = self.apply_gate(psi, G, where=(i,j,f))

        #     E_hop += (bra|G_ket) ^ all

        # return E_hop            

    
    # def compute_nnint_expecs(self, psi=None):
    #     '''
    #     Return <psi|H_int|psi> for the nearest-neighbor
    #     repulsion terms in 1D-Hubbard.
    #     '''
    #     if psi is None: 
    #         psi = self._psi



        




     