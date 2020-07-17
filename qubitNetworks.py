import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import spinlessQubit
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
from collections import defaultdict
    

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
            


class QubitEncodeNet():

    def __init__(self, qlattice, chi=8):
        '''

        _edge_map: dict[string : list(tuple(int))]
            Gives list of edges for each direction  
            in {'u','d','r','l'}
        
        _psi: TensorNetwork 
            State of the qubit lattice
        
        _vert_coo_map: dict[tuple(int) : int]
            Numbering of the vertex sites, i.e.
            maps each location (i,j) in the vertex lattice
            to an integer.

            e.g. for 4x4 vertices

            0-----1-----2-----3
            |  x  |     |  x  |
            4-----5-----6-----7
            |     |  x  |     |
            8-----9-----10----11
            |  x  |     |  x  |
            12----13----14----15
            :
            etc
        

        _face_coo_map: dict[tuple(int)-->int]
            Numbering of face sites. Assigns an integer
            to each location (i,j) in the *face* array.

            e.g. for 4x4 vertices (3x3 faces)

            x----x----x----x
            | 16 |    | 17 |
            x----x----x----x
            |    | 18 |    |
            x----x----x----x
            | 19 |    | 20 |
            x----x----x----x
        '''
        
        # verts, faces = spinlessQubit.gen_lattice_sites(Lx,Ly)
        self.qlattice = qlattice

        verts = qlattice.vert_array()
        faces = qlattice.face_array()
        Lx, Ly = qlattice._Lx, qlattice._Ly 

        self._vert_coo_map = dict(np.ndenumerate(verts))
        self._face_coo_map = dict(np.ndenumerate(faces))
        
        # tensor_net = make_skeleton_net(Lx, Ly, chi)
        tensor_net = make_random_net(Lx, Ly, chi)

        self._psi = tensor_net

        self._Lx = Lx
        self._Ly = Ly


        self._Nverts = qlattice.num_verts()
        self._Nsites = qlattice.num_sites()

        self._d_physical = 2


        self._edge_map = spinlessQubit.make_edge_map(verts, faces)
        
        self._site_tag_id = 'Q{}'
        self._phys_ind_id = 'q{}'

        
        #If we subclass TensorNetwork -- should we?
        # super().__init__(tensors)


    def vert_coo_map(self, i, j):
        '''Maps location (i,j) in vertex lattice
        to the corresponding site number.
        '''
        return self._vert_coo_map[(i,j)]
    
    def face_coo_map(self, i, j):
        '''Maps location (i,j) in *face* lattice
        to the corresponding site number.
        '''
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


        Param:

        which: {'u', 'd', 'r', 'l',
                'he', 'ho', 've', 'vo',
                'horizontal', 'all'}
        '''
        return self.qlattice.get_edges(which)


    def Lx(self):
        return self.qlattice._Lx
    

    def Ly(self):
        return self.qlattice._Ly


    def graph_psi(self, show_tags=False, auto=False):
        
        if auto:
            self._psi.graph(color=['VERT','FACE','GATE'])
        
        else:
            LAT_CONST = 50 #lattice constant for graphing
            Lx,Ly = self.Lx(), self.Ly()

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
            The sites on which to act, using the (default) 
            custom numbering that labels/orders both face 
            and vertex sites.
        '''
        # psi = self._psi if inplace else self._psi.copy()
        psi = psi if inplace else psi.copy()

        #let G be a one-site gate
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

            G = number_op() & number_op()
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

        G = number_op()

        #only finds occupations at *vertices*!
        for x,y in product(range(Lx),range(Ly)):
            
            where = self.vert_coo_map(x,y)
            G_ket = self.apply_gate(psi, G, where=(where,))

            nxy_array[x][y] = (bra | G_ket) ^ all
            

        if return_array: 
            return nxy_array

        return np.sum(nxy_array)            
    

    def compute_energy_deprec(self, t, V, mu):
        '''Don't use anymore, see `compute_ham_expec`
        '''
        E_hop, E_int, E_occ = 0,0,0

        if t!=0.0: E_hop = t  * self.compute_hop_expecs()
        if V!=0.0: E_int = V  * self.compute_nnint_expecs()
        if mu!=0.0: E_occ = - mu * self.compute_occs_expecs()
        
        return E_hop + E_int + E_occ
              

    def compute_ham_expec(self, Ham):
        '''Return <psi|H|psi>
        Ham: [SimulatorHam]
            Specifies a two- or three-site gate for each edge in
            the lattice (third site is the possible face qubit)
        '''
        psi = self._psi
        bra = psi.H

        E = 0
        
        for (i,j,f) in self.get_edges('all'):
            
            G = Ham.get_gate((i,j,f))
            where = (i,j) if f is None else (i,j,f)
            G_ket = self.apply_gate(psi, G, where)

            E += (bra|G_ket) ^ all
        
        return E

    
    # def apply_trotter_gates(self):

    #     Ham = self._Ham2D

    #     for group in ['he', 'ho', 've', 'vo']:

    #         for (i,j,f), gate in Ham.trotter_gates(group).items()

    #             where = (i,j) if f is None else (i,j,f)
    #             self.apply_gate(gate, where, inplace=True)
        


def number_op():
    '''Fermionic number operator, aka
    qubit spin-down projector
    '''
    return qu.qu([[0, 0], [0, 1]])
    


class SimulatorHam():

    def __init__(self, qlattice, t, V, mu):
        '''
        qlattice: [SpinlessQubitLattice]
                The lattice of qubits specifying the geometry
                and vertex/face sites.
        
        t: hopping parameter
        V: nearest-neighbor repulsion
        mu: single-site chemical potential

        H =   t  * hopping
            + V  * repulsion
            - mu * occupation
        '''
        
        self.qlattice = qlattice

        self._t = t
        self._V = V
        self._mu = mu

        self._ham_terms = self.make_ham_terms()
        self._exp_gates = None
        


    def make_ham_terms(self):
        '''Store all terms in Ham as two/three-site gates, 
        in a dict() mapping edges to qarrays.
        
        ``terms``:  dict{ edge (i,j,f) : gate [qarray] }

        Iff `f` is None, the corresponding gate will be two-site.
        Otherwise, gate acts on three sites.
        '''
        t, V, mu = self.ham_params()

        terms = dict()

        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:

            for (i,j,f) in self.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] = sign * t * self.two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())
                
                #three-site
                else:
                    terms[(i,j,f)] = sign * t * self.three_site_hop_gate(edge_dir='vertical')
                    terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))


        #horizontal edges
        for (i,j,f) in self.get_edges('horizontal'):

            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self.two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())

            #three-site    
            else:
                terms[(i,j,f)] =  t * self.three_site_hop_gate(edge_dir='horizontal')
                terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))

        
        if mu == 0.0:
            return terms


        n_op = number_op() #one-site number operator 

        #map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))


        #for each vertex in lattice, absorb chemical potential term
        #uniformly into the edge terms that include it
        for vertex in self.qlattice.vertex_sites():
            
            #get edges that include this vertex
            edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(edges)

            assert num_edges > 1 #should appear in at least two edge terms!

            for (i,j,f) in edges:
                
                ham_term = terms[(i,j,f)]

                v_place = (i,j,f).index(vertex) #vertex is either i or j

                if f is None: #ham_term should act on two sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*2, inds=v_place)

                else: #act on three sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*3, inds=v_place)

        return terms




    def two_site_hop_gate(self):
        '''Hopping between two vertices, with no face site.
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        return 0.5* ((X&X) + (Y&Y))


    def three_site_hop_gate(self, edge_dir):
        '''Hop gate acting on two vertices and a face site.
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        O_face = {'vertical': X, 'horizontal':Y} [edge_dir]

        return 0.5 * ((X & X & O_face) + (Y & Y & O_face))
        

    def get_gate(self, edge):
        '''Term in Ham corresponding to ``edge``.
        '''
        return self._ham_terms[edge]


    def get_trotter_gates(self, group):
        '''Returns mapping of edges (in ``group``) to
        the corresponding Trotter gates
        
        Returns: dict[edge : exp(Ham gate)]
        '''
        edges = self.get_edges(group)
        gate_map = {edge : self._exp_gates[edge] for edge in edges}
        return gate_map
    

    def get_edges(self, which):
        '''Retrieves (selected) edges from internal qLattice object.
        '''
        return self.qlattice.get_edges(which)


    def ham_params(self):
        '''Gets Ham coupling constants
       (t: hopping parameter,
        V: nearest-neighbor repulsion,
        mu: chemical potential)
        '''
        return (self._t, self._V, self._mu)
    
    def Lx():
        return self.qlattice._Lx
    
    def Ly():
        return self.qlattice._Ly