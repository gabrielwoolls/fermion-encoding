import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import spinlessQubit
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
from collections import defaultdict
from autoray import do, dag
import tqdm
    

def make_skeleton_net(Lx, Ly, phys_dim, chi=5, show_graph=False):
    '''
    TODO: fix ftensor type (change ndarray->list[list])
            change defaults?
            tensor tag id

    Make a qubit TensorNetwork from local states in `arrays`, i.e.
    the TN will represent a product state.

    arrays: sequence of arrays, optional
        Specify local qubit states in vertices, faces
    '''
    
    #default to `up` spin at every site
    
    vert_array = [[qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim)
                    for j in range(Ly)]
                    for i in range(Lx)]
        
    face_array = [[qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim)
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


def make_random_net(Lx, Ly, phys_dim, chi=5):
    '''TODO: take phys_dim into account?
    '''

    #dummy TN, sites to be replaced with random tensors
    tnet = make_skeleton_net(Lx, Ly, phys_dim, chi)

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


def make_vertex_net(Lx,Ly, chi=5, show_graph=True):
    vert_array = [[qu.up().reshape(2) 
                    for j in range(Ly)]
                    for i in range(Lx)]
       

    vtensors = [[qtn.Tensor(data = vert_array[i][j], 
                            inds = [f'q{i*Ly+j}'],
                            tags = {f'Q{i*Ly+j}', 'VERT'}) 
                for j in range(Ly)] 
                for i in range(Lx)]

    

    for i,j in product(range(Lx), range(Ly)):
        # vtensors[i][j].new_ind(f'q{i*Ly+j}',size=2)
        # vtensors[i][j].add_tag(f'Q{i*Ly+j}')

        if i<=Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=chi)
        if j<=Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=chi)


    # alltensors = vtensors + [f for f in ftensors.flatten().tolist() if not f is None]
    vtn = qtn.TensorNetwork(vtensors)


    if show_graph:
        LAT_CONST = 50 #lattice constant for graph

        fix = {
            **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
            }
        vtn.graph(color=['VERT','FACE'], show_tags=False, fix=fix)

    return vtn


class iTimeTEBD:
    def __init__(
        self,
        qnetwork, 
        ham, 
        chi=8,
        tau=0.01,
        progbar=True,
        compute_every=None
    ):

        self.psi0 = qnetwork.state()
        self.qnet = qnetwork
        self.ham = ham
        self.tau = tau
        self.progbar = progbar
        

        self._n = 0
        self.iters = [] #iterations
        self.taus = []
        self.energies=[]

        self.compute_energy_every = compute_every
        

    def _check_energy(self):
        if self.iters and (self._n==self.iters[-1]):
            return self.energies[-1]
        
        en = self.compute_energy()
        
        self.energies.append(en)
        self.taus.append(float(self.tau))
        self.iters.append(self._n)

        return self.energies[-1]

    def _update_progbar(self, pbar):
        desc = f"n={self._n}, tau={self.tau}, energy~{float(self._check_energy()):.6f}"
        pbar.set_description(desc)


    def compute_energy(self):
        return self.qnet.compute_ham_expec(self.ham)


    def evolve(self, steps):
        tau = self.tau

        pbar = tqdm.tqdm(total=steps, disable=self.progbar is not True)

        try:
            for i in range(steps):

                should_compute_energy = (
                    bool(self.compute_energy_every) and
                    (i % self.compute_energy_every == 0))
                
                if should_compute_energy:
                    self._check_energy()
                    self._update_progbar(pbar)
                
                self.sweep()
                self._n += 1
                pbar.update()
        
        except KeyboardInterrupt:
            # allow the user to interupt early
            pass
        
        finally:
            pbar.close()
    
    def sweep(self):
        '''Perform a full sweep of gates at every edge.
        '''
        self.qnet.apply_trotter_gates_(self.ham, -self.tau)
        




class QubitEncodeNet:

    def __init__(self, qlattice, psi=None, phys_dim=2, chi=5):
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
        # wavefn = make_random_net(Lx, Ly, chi)


        #simulator wavefunction
        self._psi = psi if psi is not None else make_random_net(Lx, Ly, chi, phys_dim)

        # self._edge_map = spinlessQubit.make_edge_map(verts, faces)

        self._phys_dim = phys_dim
        
        self._site_tag_id = 'Q{}'
        self._phys_ind_id = 'q{}'

        
        #If we subclass TensorNetwork -- should we?
        # super().__init__(tensors)


    @classmethod
    def rand_network(cls, qlattice, phys_dim, chi=5):
        
        Lx, Ly = qlattice._Lx, qlattice._Ly
        randnet = make_random_net(Lx, Ly, phys_dim, chi)
        return cls(qlattice, randnet, phys_dim, chi)


    def sim_state(self):
        return self._psi.copy()


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
            List of (three-tuple) edges, where tuple
            (i,j,f) denotes the edge with vertices i,j 
            and face f (int or None)


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


    def Nsites(self):
        return self.qlattice.num_sites()


    def Nverts(self):
        return self.qlattice.num_verts()


    def graph_psi(self, show_tags=False, auto=False, **graph_opts):
        
        if auto:
            self._psi.graph(color=['VERT','FACE','GATE'], *graph_opts)
        
        else:
            LAT_CONST = 50 #lattice constant for graphing
            Lx,Ly = self.Lx(), self.Ly()

            fix = {
                **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
                }
            self._psi.graph(color=['VERT','FACE','GATE'], show_tags=show_tags, fix=fix, show_inds=True)



    def net_to_dense(self):
        '''Return self._psi as dense vector, i.e. a qarray with 
        shape (-1, 1)
        '''
        #TODO: check change from range(self._Nsites)
        
        inds_seq = (f'q{i}' for i in self.qlattice.all_sites())
        return self._psi.to_dense(inds_seq).reshape(-1,1)


    def apply_gate_(self, G, where):
        '''Inplace apply gate to internal
        wavefunction i.e. self._psi
        '''
        self.apply_gate(psi=self._psi, 
                        G=G, 
                        where=where,
                        inplace=True)


    def apply_gate(self, psi, G, where, inplace=False):
        '''
        TODO: incorporate `physical_ind_id`?
        
        Apply gate `G` at sites specified in `where`,
        preserving physical indices of `psi`.

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
        # site_inds = [self._phys_ind_id.format(i) for i in where] 

        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))

        TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=['GATE'])
        
        psi.reindex_(reindex_map)
        psi |= TG
        return psi


    def apply_stabilizer_gate_(self, vert_inds, face_ops, face_inds):
        '''Inplace application of a stabilizer gate that acts with 
        'ZZZZ' on `vert_inds`, and acts on `face_inds` with the operators
        specified in `face_ops`, e.g. 'YXY'.

        vert_inds: sequence of ints (length 4)
        face_ops: string 
        face_inds: sequence of ints (len face_inds==len face_ops)
        '''
        X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
        opmap = {'X': X, 'Y':Y, 'Z':Z, 'I':I}
        stab_op = qu.kron(*[opmap[Q] for Q in ('ZZZZ' + face_ops)])

        self.apply_gate(psi = self._psi, 
                        G = stab_op, 
                        where = vert_inds + face_inds,
                        inplace = True)



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
        Lx,Ly = self.Lx(), self.Ly()

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
            the lattice (third site if acting on the possible face
            qubit)
        '''
        psi = self._psi
        bra = psi.H

        E = 0
        
        for (i,j,f) in self.get_edges('all'):
            
            G = Ham.get_gate((i,j,f))
            
            #two vertices + possible face site
            where = (i,j) if f is None else (i,j,f)
            
            G_ket = self.apply_gate(psi, G, where)
            
            E += (bra|G_ket) ^ all
        
        return E

    
    def apply_trotter_gates_(self, Ham, tau):
        '''Inplace-apply the Ham gates, exponentiated by `tau`,
        in groups of {horizontal-even, horizontal-odd, etc}.
        '''
        for group in ['he', 'ho', 've', 'vo']:

            for edge, gate in Ham.trotter_gates(group, tau).items():
                
                i,j,f = edge
                where = (i,j) if f is None else (i,j,f)
                #inplace gate apply
                self.apply_gate_(gate, where)
        


def number_op():
    '''Fermionic number operator, aka
    qubit spin-down projector
    '''
    return qu.qu([[0, 0], [0, 1]])
    

### *********************** ###

class SimulatorHam():
    '''Parent class for simulator (i.e. qubit-space) 
    Hamiltonians. 

    Needs a `qlattice` object to handle lattice geometry/edges, 
    and a mapping `_ham_terms` of edges to two/three site gates.
    '''
    
    def __init__(self, qlattice, ham_terms):
        
        self.qlattice = qlattice
        self._ham_terms = ham_terms
        self._exp_gates = dict()

        # self._op_cache = defaultdict(dict)


    # def _expm_cached(self, gate, t):
    #     cache = self._op_cache['expm']
    #     key = (id(gate), t)
    #     if key not in cache:
    #         el, ev = do('linalg.eigh', gate)
    #         cache[key] = ev @ do('diag', do('exp', el * t)) @ dag(ev)
    #     return cache[key]

    def get_gate(self, edge):
        '''Local term corresponding to `edge`
        '''
        return self._ham_terms[edge]

    def get_expm_gate(self, edge, t):
        '''Local term for `edge`, matrix-exponentiated
        by `t`.
        '''
        # return self._expm_cached(self.get_gate(edge), x)
        key = (edge, t)
        if key not in self._exp_gates:
            gate = self.get_gate(edge)
            el, ev = do('linalg.eigh',gate)
            self._exp_gates[key] = ev @ do('diag', do('exp', el*t)) @ dag(ev)
        return self._exp_gates[key]

    

    def trotter_gates(self, group, x):
        '''Returns mapping of edges (in ``group``) to
        the corresponding exponentiated gates.
        
        Returns: dict[edge : exp(Ham gate)]
        '''
        edges = self.get_edges(group)
        gate_map = {edge : self.get_expm_gate(edge,x) for edge in edges}
        return gate_map
    

    def get_edges(self, which):
        '''Retrieves (selected) edges from internal 
        qlattice object.
        '''
        return self.qlattice.get_edges(which)


    def ham_params(self):
        '''Relevant parameters. Override for
         each daughter Hamiltonian.
        '''
        pass


    def Lx():
        return self.qlattice._Lx


    def Ly():
        return self.qlattice._Ly


## ******************* ##
# Subclass Hamiltonians
## ******************* ##

class SpinlessFermiSim(SimulatorHam):
    '''Encoded Hubbard Hamiltonian for spinless fermions,
    encoded as a qubit simulator Ham.

    H =   t  * hopping
        + V  * repulsion
        - mu * occupation
    '''

    def __init__(self, qlattice, t, V, mu):
        '''
        qlattice: [SpinlessQubitLattice]
                The lattice of qubits specifying the geometry
                and vertex/face sites.
        
        t: hopping parameter
        V: nearest-neighbor repulsion
        mu: single-site chemical potential

        '''
        
        self._t = t
        self._V = V
        self._mu = mu

        terms = self.make_ham_terms(qlattice)

        super().__init__(qlattice, terms)
        

    def make_ham_terms(self, qlattice):
        '''Get all terms in Ham as two/three-site gates, 
        in a dict() mapping edges to qarrays.
        
        ``terms``:  dict[edge (i,j,f) : gate [qarray] ]

        If `f` is None, the corresponding gate will be two-site 
        (vertices only). Otherwise, gate acts on three sites.
        '''
        t, V, mu = self.ham_params()

        terms = dict()

        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] = sign * t * self.two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())
                
                #three-site
                else:
                    terms[(i,j,f)] = sign * t * self.three_site_hop_gate(edge_dir='vertical')
                    terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('horizontal'):

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
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(edges)

            assert num_edges>1 or qlattice.num_faces()==0 #should appear in at least two edge terms!

            for (i,j,f) in edges:
                
                # ham_term = terms[(i,j,f)]

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
        

    def ham_params(self):
        '''Gets Ham coupling constants
       (t: hopping parameter,
        V: nearest-neighbor repulsion,
        mu: chemical potential)
        '''
        return (self._t, self._V, self._mu)
    
###

class SpinhalfHubbardSim(SimulatorHam):
    '''Simulator Hamiltonian, acting on qubit space,
    that encodes the Fermi-Hubbard model Ham for 
    spin-1/2 fermions. Each local site is 4-dimensional.

    Gates act on 2 or 3 sites (2 vertices + 1 possible face)
    '''

    def __init__(self, qlattice, t, U):
        
        self._t = t
        self._U = U

        terms = self.make_ham_terms(qlattice)

        super().__init__(qlattice, terms)
    

    def make_ham_terms(self, qlattice):
        '''Get all Hamiltonian terms, as two/three-site gates, 
        in a dict() mapping edges to arrays.
        
        `terms`:  dict[edge (i,j,f) : gate [qarray] ]

        If `f` is None, the corresponding gate will be two-site 
        (vertices only). Otherwise, gate acts on three sites.
        '''
        t, U = self.ham_params()

        terms = dict()

        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:
            
            spin_up_hop = self.two_site_hop_gate(spin=0)
            spin_down_hop = self.two_site_hop_gate(spin=1)

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] =  sign * t * spin_up_hop
                    terms[(i,j,f)] += sign * t * spin_down_hop
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
                #three-site
                else:
                    terms[(i,j,f)] =  sign * t * self.three_site_hop_gate(spin=0, edge_dir='vertical')
                    terms[(i,j,f)] += sign * t * self.three_site_hop_gate(spin=1, edge_dir='vertical')
                    # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('right+left'):
            
            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self.two_site_hop_gate(spin=0)
                    terms[(i,j,f)] += t * self.two_site_hop_gate(spin=1)
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
            #three-site    
            else:
                terms[(i,j,f)] =  sign * t * self.three_site_hop_gate(spin=0, edge_dir='horizontal')
                terms[(i,j,f)] += sign * t * self.three_site_hop_gate(spin=1, edge_dir='horizontal')
                # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)
        
        
        if U == 0.0:
            return terms


        G_onsite = self.onsite_int_gate() #on-site spin-spin interaction

        #map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))


        #for each vertex in lattice, absorb onsite repulsion term
        #uniformly into the edge terms that include it
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(edges)

            assert num_edges>1 or qlattice.num_faces()==0  #should appear in at least two edge terms!

            for (i,j,f) in edges:
                
                # ham_term = terms[(i,j,f)]

                v_place = (i,j,f).index(vertex) #vertex is either i or j

                if f is None: #term should act on two sites
                    terms[(i,j,f)] += U * (1/num_edges) * qu.ikron(G_onsite, dims=[4]*2, inds=v_place)

                else: #act on three sites
                    terms[(i,j,f)] += U * (1/num_edges) * qu.ikron(G_onsite, dims=[4]*3, inds=v_place)

        return terms


    def ham_params(self):
        '''t (hopping), U (repulsion)
        '''
        return (self._t, self._U)


    def two_site_hop_gate(self, spin):
        '''(Encoded) hopping between two vertex sites, for 
        fermions in the `spin` sector.
        '''
        spin = {0: 'u',
                1: 'd',
                'up': 'u',
                'down': 'd',
                'u': 'u',
                'd': 'd'
                }[spin]
            
        X, Y, I = (qu.pauli(mu) for mu in ['x','y','i'])

        #`spin` says which spin sector to act on

        X_s = { 'u': X & I, 
                'd': I & X}[spin]

        Y_s = { 'u': Y & I,
                'd': I & Y}[spin]

        return 0.5* ((X_s & X_s) + (Y_s & Y_s))
    

    def three_site_hop_gate(self, spin, edge_dir):
        '''Hop gate acting on two vertices and a face site,
        in `spin` sector. Action on the face site depends
        on `edge_dir`: {'vertical', 'horizontal'}
        '''
        spin = {0: 'up',
                1: 'down',
                'up': 'up',
                'down': 'down',
                'u': 'up',
                'd': 'down'
                }[spin]

        X, Y, I = (qu.pauli(mu) for mu in ['x','y','i'])
        
        #operators acting on the correct spin sector
        if spin == 'up':
            X_s = X & I
            Y_s = Y & I
        
        elif spin == 'down':
            X_s = I & X
            Y_s = I & Y

        #operator on face qubit (in `spin` sector)
        Of_s = {'vertical': X_s, 'horizontal':Y_s} [edge_dir]

        return 0.5 * ((X_s & X_s & Of_s) + (Y_s & Y_s & Of_s))


    def onsite_int_gate(self):
        '''Spin-spin interaction at a single vertex.
        '''
        return number_op() & number_op()