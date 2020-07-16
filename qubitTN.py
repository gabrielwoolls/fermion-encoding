import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import spinlessQubit
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
    

def make_skeleton_net(Lx, Ly, chi=5, phys_dim=2, show_graph=False):
    '''
    TODO: fix ftensor type (change ndarray->list[list])
            change defaults?

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
        LAT_CONST = 50 #lattice constant for graph

        fix = {
            **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
            }
        vtn.graph(color=['VERT','FACE'], show_tags=True, fix=fix)

    return vtn


def make_random_net(Lx, Ly, bond_dim, phys_dim=2):
    '''TODO: take phys_dim into account?
    '''

    #dummy TN, sites to be replaced with random tensors
    tnet = make_skeleton_net(Lx, Ly, bond_dim)

    #replace vertex tensors
    for i, j in product(range(Lx), range(Ly)):
        
        tid = tuple(tnet.tag_map[f'Q{i*Ly+j}'])
        assert len(tid)==1
        tid = tid[0]

        old_tensor = tnet._pop_tensor(tid)

        shape = old_tensor.shape
        tags = old_tensor.tags #[f'Q{i*Ly + j}', 'VERT']
        inds  = old_tensor.inds #[..., f'q{i*Ly + j}']

                
        rand_data = qtn.array_ops.sensibly_scale(
                            qtn.array_ops.sensibly_scale(
                            qu.gen.rand.randn(shape)))
        
        tensor_ij = qtn.Tensor(rand_data, inds, tags)
        tnet |= tensor_ij
    
    
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):
        #replace face tensors
        if i%2 == j%2:
            
            tid = tuple(tnet.tag_map[f'Q{k+Lx*Ly}'])
            assert len(tid)==1
            tid = tid[0]

            old_tensor = tnet._pop_tensor(tid)

            shape = old_tensor.shape
            tags = old_tensor.tags 
            inds  = old_tensor.inds

                    
            rand_data = qtn.array_ops.sensibly_scale(
                                qtn.array_ops.sensibly_scale(
                                qu.gen.rand.randn(shape)))
            
            tensor_ij = qtn.Tensor(rand_data, inds, tags)
            tnet |= tensor_ij


        else: pass
    
    return tnet
            


class MyQubitTN():

    def __init__(self, Lx, Ly, chi):
        '''

        _edge_map: dict[string --> list(tuple(int))]
            Gives list of edges for each direction  
            in {'u','d','r','l'}
        
        _psi: TensorNetwork 
            State of the qubit lattice
        
        _vert_coo_map: dict[tuple(int) --> int]
            Numbering of the vertex sites, i.e.
            maps each location (i,j) in the vertex lattice
            to an integer.

            e.g. for 4x4 vertices

            0-----1-----2-----3
            |  x  |     |  x  |
            4-----5-----6-----7
            |     |  x  |     |
            8-----9----10----11
            |  x  |     |  x  |
            12----13----14----15
            :
            etc
        

        _face_coo_map: dict[tuple(int)-->int]
            Numbering of face sites. To each tuple
            (i,j) denoting a location in the *face*
            lattice, assigns an integer.

            e.g. for 4x4 vertices (3x3 faces)

            o----o----o----o
            | 16 |    | 17 |
            o----o----o----o
            |    | 18 |    |
            o----o----o----o
            | 19 |    | 20 |
            o----o----o----o
        '''
        
        verts, faces = spinlessQubit.gen_lattice_sites(Lx,Ly)
        
        # tensor_net = make_skeleton_net(Lx, Ly, chi)
        tensor_net = make_random_net(Lx, Ly, chi)
        

        self._vert_coo_map = dict(np.ndenumerate(verts))
        self._face_coo_map = dict(np.ndenumerate(faces))
        
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


    def get_edges(self, which):
        '''
        Returns: list[tuple(int or None)]
            List of (three-tuple) edges, where a tuple
            (i,j,f) denotes the edge with vertices i,j 
            (ints) and face f (int or None)


        which: {'u','d','r','l','all'}
            Which edges of the graph to return.

        '''
        if which == 'all':
            return list(self._edge_map['r'] + 
                        self._edge_map['l'] +
                        self._edge_map['u'] +
                        self._edge_map['d'] )

        else:
            return self._edge_map[which]



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
            shape ``([physical_dim, physical_dim]*len(where))``
        
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
 

    
    def compute_nnint_expecs(self, psi=None):
        '''
        Return <psi|H_int|psi> for the nearest-neighbor
        repulsion terms in spinless-Hubbard.
        '''
        if psi is None: 
            psi = self._psi
        
        E_int = 0
        bra = psi.H

        for (i, j, _) in self.get_edges('all'):
            #ignore all faces here

            G = self.number_op() & self.number_op()
            G_ket = self.apply_gate(psi, G, where=(i,j))

            E_int += (bra|G_ket)^all

        return E_int


    def compute_occs_expecs(self, psi=None, return_array=False):
        '''
        Compute local occupation/number expectations,
        <psi|n_xy|psi>

        return_array: bool
            Whether to return 2D array of local number 
            expectations. Defaults to false, in which case
            only the total sum is returned.
        '''
        Lx,Ly = self._Lx, self._Ly

        if psi is None: 
            psi = self._psi
        
        bra = psi.H

        nxy_array = [[None for y in range(Ly)] for x in range(Lx)]

        G = self.number_op()

        #only finds occupations at *vertices*!
        for x,y in product(range(Lx),range(Ly)):
            
            where = self.vert_coo_map(x,y)
            G_ket = self.apply_gate(psi, G, where=(where,))

            nxy_array[x][y] = (bra | G_ket) ^ all
            

        if return_array: 
            return nxy_array

        return np.sum(nxy_array)            
    


    def compute_energy(self, t, V, mu):
        return (t  * self.compute_hop_expecs() 
              + V  * self.compute_nnint_expecs()
              - mu * self.compute_occs_expecs())


    def number_op(self):
            '''
            Fermionic number operator is
            mapped to qubit spin-down
            projector acting on 2-dim qbit space.

            n_j --> (1-Vj)/2 
                    = (1-Zj)/2 
                    = |down><down|
            '''
            return qu.qu([[0, 0], [0, 1]])
            




     