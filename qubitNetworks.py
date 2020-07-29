import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import denseQubits
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
from collections import defaultdict
from numbers import Integral
from autoray import do, dag
import tqdm
    

def make_skeleton_net(Lx, Ly, phys_dim, 
                    chi=5, show_graph=False):
    '''
    NOTE: uses 'Q{}' already!!
    
    TODO:
    fix ftensor type (change ndarray->list[list])
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
    
    vtn = qtn.TensorNetwork(alltensors, structure='Q{}')

    if show_graph:
        LAT_CONST = 50 #lattice constant for graph

        fix = {
            **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
            }
        vtn.graph(color=['VERT','FACE'], show_tags=True, fix=fix)

    return vtn


def make_random_net(qlattice, chi=5):
    '''Return a `TensorNetwork` made from random tensors
    structured like `qlattice` i.e. with the same local 
    qu(d)it degrees of freedom. 
    
    Each site has physical index dimension `d = qlattice._local_dim`,
    and is connected to its neighbors with a virtual bond of 
    dimension `chi`.
    '''
    Lx, Ly = qlattice._lat_shape
    phys_dim = qlattice._local_dim

    #dummy TN, site tensors to be replaced with randomized
    tnet = make_skeleton_net(Lx, Ly, phys_dim, chi)

    #replace vertex tensors with randoms
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
    

    #replace face tensors with randoms
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
    '''TODO: FIX HAMILTONIAN CLASS
    Object for TEBD imaginary-time evolution.
    Params:
    ------
    `qnetwork`: QubitEncodeNet
        The initial state of the qubits. Will be modified
        in-place.
    
    `ham`: `----`
        Hamiltonian for time-evolving. Should have a
        `gen_trotter_gates(tau)` method.
    
    `compute_extra_fns`: callable or dict of callables, optional
        If desired, can give extra callables to compute at each
        step of TEBD. Each function should take only the current 
        state `qnet` as parameter. Results of the callables will 
        be stored in `self._results`
    '''
    def __init__(
        self,
        qnetwork, 
        ham, 
        chi=8,
        tau=0.01,
        progbar=True,
        compute_every=None,
        compute_extra_fns=None
    ):

        self.psi0 = qnetwork.qbit_state()
        self.qnet = qnetwork

        self.ham = ham
        self.tau = tau
        self.progbar = progbar
        

        self._n = 0
        self.iters = [] #iterations
        self.energies=[]

        self.compute_energy_every = compute_every

        self._setup_callback(compute_extra_fns)



    def _setup_callback(self, fns):
        '''Setup for any callbacks to be computed during 
        imag-time evolution.
        
        `fns`: callable, or dict of callables, or None
            Callables should take only `qnet` as parameter, i.e.
            the current `QubitEncodeNetwork` state.

        '''

        if fns is None:
            self._step_callback = None
        
        #fns is callable or dict of callables
        else:

            if isinstance(fns, dict):
                self._results = {k: [] for k in fns}

                def step_callback(psi_t):
                    for k, func in fns.items():
                        fn_result = func(psi_t)
                        self._results[k].append(fn_result)
            
            #fns is a single callable
            else:
                self._results = []

                def step_callback(psi_t):
                    fn_result = fns(psi_t)
                    self._results.append(fn_result)
            

            self._step_callback = step_callback
            


    def _check_energy(self):
        '''Compute energy, unless we have already computed
        it for this time-step.
        '''
        if self.iters and (self._n==self.iters[-1]):
            return self.energies[-1]
        
        en = self.compute_energy()
        
        self.energies.append(en)
        self.iters.append(self._n)

        return self.energies[-1]


    def _update_progbar(self, pbar):
        desc = f"n={self._n}, tau={self.tau}, energy~{float(self._check_energy()):.6f}"
        pbar.set_description(desc)


    def _compute_extras(self):
        '''For any extra functions the TEBD object was 
        given to compute at each step of evolution, pass
        them the current state `self.qnet`
        '''
        if self._step_callback is not None:
            self._step_callback(self.qnet)


    def compute_energy(self):
        '''<psi|Ham|psi> / <psi|psi>
        '''
        return self.qnet.compute_ham_expec(self.ham, normalize=True)


    def evolve(self, steps):
        pbar = tqdm.tqdm(total=steps, disable=self.progbar is not True)

        try:
            for i in range(steps):

                should_compute_energy = (
                    bool(self.compute_energy_every) and
                    (i % self.compute_energy_every == 0))
                
                if should_compute_energy:
                    self._check_energy()
                    self._update_progbar(pbar)
                    self._compute_extras()
                
                self.sweep()

                # self.update_norm()
                # self.normsq = self.get_current_normsq()

                self._n += 1

                pbar.update()

            #compute final energy
            self._check_energy()
            self._compute_extras()

        except KeyboardInterrupt:
            # allow early interrupt
            pass
        
        finally:
            pbar.close()
    
    def sweep(self):
        '''Perform a full sweep, apply all `exp(gate)`s
        '''
        self.qnet.apply_trotter_gates_(self.ham, -self.tau)
        self.qnet.contract(tags=['GATE'], inplace=True)
    

    def results(self, which=None):
        '''Convenience property for testing.
        '''
        if which is None:
            return self._results
        
        elif which == 'energy':
            return self.energies

        return np.array(self._results[which])


    def get_final_data(self, data):
        '''Convenience method for testing.
        '''

        if data == 'Esim':
            return np.divide(np.real(self.results('sim')),
                            np.real(self.results('norm'))
                            )
        
        elif data == 'Estab':
            return np.divide(np.real(self.results('stab')),
                            np.real(self.results('norm'))
                            )
        
        





def compute_encnet_ham_expec(qnet, ham):
    '''Useful callable for TEBD
    '''
    return qnet.compute_ham_expec(ham, normalize=False)


def compute_encnet_normsquared(qnet):
    return np.real(qnet.get_norm_tensor()^all)


class QubitEncodeNet(qtn.TensorNetwork):
    '''
        Params:
        -------
        `tn`: TensorNetwork
            State of the tensors in the lattice.

        `qlattice`: QubitCodeLattice
            Specifies the underlying lattice geometry, in 
            particular the shape (Lx, Ly) of the lattice
            and the local dimension of the physical sites,
            e.g. d=2 (4) for simulating spinless (spin-1/2) 
            fermions.
        
        `chi`: int, optional
            The bond dimension to connect local site tensors.
            Defaults to 5.
        
        

        Attributes:
        ----------

        `_vert_coo_map`: dict[tuple(int) : int]
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
        

        `_face_coo_map`: dict[tuple(int)-->int]
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
        
    def __init__(
                self, 
                tn, 
                qlattice, 
                chi=5,
                site_tag_id = 'Q{}',
                phys_ind_id = 'q{}',
                 **tn_opts):
        
        #shortcut for copying QENs
        if isinstance(tn, QubitEncodeNet):
            self._qlattice = tn.qlattice
            self._vert_coo_map = tn.vert_coo_map
            self._face_coo_map = tn.face_coo_map
            self._site_tag_id = tn.site_tag_id
            self._phys_ind_id = tn.phys_ind_id
            super().__init__(tn)
            return

        self._qlattice = qlattice

        self._vert_coo_map = dict(np.ndenumerate(qlattice.vert_array))

        self._face_coo_map = dict(np.ndenumerate(qlattice.face_array))
        
        self._site_tag_id = site_tag_id

        self._phys_ind_id = phys_ind_id

        super().__init__(tn, **tn_opts)


    @property
    def qlattice(self):
        '''Internal qubit lattice object
        '''
        return self._qlattice

    @property
    def phys_dim(self):
        '''Local physical dimension of qu(d)it sites.
        '''        
        return self.qlattice.local_site_dim

    @property
    def site_tag_id(self):
        '''Format string for the tag identifiers of local sites
        '''
        return self._site_tag_id

    @property
    def phys_ind_id(self):
        '''Format string for the physical indices
        '''
        return self._phys_ind_id


    @classmethod
    def rand_network(cls, qlattice, chi=5, **tn_opts):
        '''Make a random QubitEncodeNet
        '''
        rand_tn = make_random_net(qlattice, chi)
        return cls(rand_tn, qlattice, chi)


    def copy(self):
        return self.__class__(self, self.qlattice)

    __copy__ = copy


    # def _site_tid(self, k):
    #     '''Given the site index `k` in {0,...,N},
    #     return the `tid` of the local tensor
    #     '''
    #     #'q{k}'
    #     index = self._phys_ind_id.format(k)
    #     return self.ind_map[index]


    def vert_coo_map(self, i=None, j=None):
        '''Maps location (i,j) in vertex lattice
        to the corresponding site number.
        '''
        if (i is not None) and (j is not None):
            return self._vert_coo_map[(i,j)]
        
        return self._vert_coo_map


    def face_coo_map(self, i=None, j=None):
        '''Maps location (i,j) in *face* lattice
        to the corresponding site number.
        '''
        if (i is not None) and (j is not None):
            return self._face_coo_map[(i,j)]

        return self._face_coo_map



    #TODO: TAKE INTERNAL TAG INTO ACCOUNT
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


    @property
    def Lx(self):
        return self.qlattice._Lx
    
    @property
    def Ly(self):
        return self.qlattice._Ly


    def num_sites(self):
        return self.qlattice.num_sites()


    def num_verts(self):
        return self.qlattice.num_verts()


    def codespace_dims(self):
        return self.qlattice.codespace_dims()


    def simspace_dims(self):
        return self.qlattice.simspace_dims()


    def graph_lat(self, show_tags=False, auto=False, **graph_opts):
        
        if auto:
            self.graph(color=['VERT','FACE','GATE'], *graph_opts)
        
        else:
            LAT_CONST = 50 #lattice constant for graphing
            Lx,Ly = self.Lx, self.Ly

            fix = {
                **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
                }
            self.graph(color=['VERT','FACE','GATE'], show_tags=show_tags, fix=fix, show_inds=True, **graph_opts)


    def check_dense_energy(self, Hdense, normalize=True):
        
        psi_d = self.net_to_dense()
        
        psiHpsi = (psi_d.H @ Hsim @ psi_d).item()

        if not normalize:
            return psiHpsi
        
        normsq = self.get_norm_tensor() ^ all
        return psiHpsi / normsq
        

    def dense_inner_product(self, dense_bra):

        dense_bra = qu.qu(dense_bra)/np.linalg.norm(dense_bra)
        
        dense_ket = self.net_to_dense(normalize=True)

        return dense_bra.H @ dense_ket


    # def exact_projector_from_matrix(self, Udag_matrix):
    #     Nfermi, Nqubit = self.num_verts(), self.num_sites()

    #     if Udag_matrix.shape != (2**Nfermi, 2**Nqubit):
    #         raise ValueError('Wrong U* shape')
        
    #     sim_dims = self.simspace_dims()
    #     code_dims = self.codespace_dims()
        
    #     sim_inds = [f'q{i}' for i in range(Nqubit)]
        
    #     Udagger = Udag_matrix.reshape(code_dims+sim_dims)

    #     Udagger = qtn.Tensor()


    def net_to_dense(self, normalize=True):
        '''Return state as dense vector, i.e. a qarray with 
        shape (-1, 1), in the order assigned to the local sites.
        '''
        inds_seq = (self._phys_ind_id.format(i) 
                    for i in self.qlattice.all_sites())

        psid = self.to_dense(inds_seq).reshape(-1,1)

        if not normalize:
            return psid
        
        return psid / np.linalg.norm(psid)


    def apply_gate_(self, G, where, contract=False):
        '''In-place apply gate `G` to the TN state
        '''
        return self.apply_gate( G=G, 
                                where=where,
                                inplace=True,
                                contract=contract)


    def apply_gate(
        self,
        G, 
        where,
        inplace=False, 
        contract=False
        ):
        '''Apply gate `G` acting at sites `where`,
        preserving physical indices. Note `G` has to be a
        "square" matrix!

        Params:
        ------            
        `G` : array
            Gate to apply, should be compatible with 
            shape ([physical_dim] * 2 * len(where))
        
        `where`: sequence of ints
            The sites on which to act, using the (default) 
            custom numbering that labels/orders both face 
            and vertex sites.
        
        `inplace`: bool, optional
            If False (default), return copy of TN with gate applied
        
        `contract`: bool, optional
            If false (default) leave all gates uncontracted
            If True, contract gates into one tensor in the lattice

        '''

        psi = self if inplace else self.copy()

        #G can be a one-site gate
        if isinstance(where, Integral): 
            where = (where,)

        numsites = len(where) #gate acts on `numsites`
        dp = self.phys_dim #local physical dimension

        G = maybe_factor_gate_into_tensor(G, dp, numsites, where)

        #new physical indices 'q{i}'
        site_inds = [self._phys_ind_id.format(i) for i in where] 

        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        
        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))

        TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=['GATE'])
        
        if contract is False:
            #attach gates w/out contracting any bonds
            psi.reindex_(reindex_map)
            psi |= TG
            return psi

        elif contract is True or numsites==1:
            #just contract the physical leg(s)

            psi.reindex_(reindex_map)
            
            #sites that used to have physical indices
            site_tids = psi._get_tids_from_inds(bond_inds, which='any')
           
            # pop the sites (inplace), contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
            psi |= qtn.tensor_contract(*pts, TG)

            return psi
        
        elif contract == 'split' and numsites==2:
    
            tensor_a, tensor_b = (psi[k] for k in where)

            
            
            string = where
            original_ts = [tensor_a, tensor_b]
            bonds_along = [next(iter(qtn.bonds(t1, t2)))
                       for t1, t2 in qu.utils.pairwise(original_ts)]
            site_ix = site_inds
            
            gss_opts = {'TG' : TG,
                        'where' : where,
                        'string': string,
                        'original_ts' : original_ts,
                        'bonds_along' : bonds_along,
                        'site_ix' : site_ix,
                        'reindex_map' : reindex_map,
                        'info' : None}

            qu.tensor.tensor_2d.gate_string_split_(**gss_opts)
            return psi                        


        else:
            raise ValueError('Failed to contract')

    def get_norm_tensor(self):
        '''<psi|psi> as an uncontracted TensorNetwork.
        '''

        ket = self.copy()
        ket.add_tag('KET')

        bra = ket.retag({'KET':'BRA'})
        bra.conj_()

        return ket | bra
    

    def norm_squared(self):
        '''Scalar quantity <psi|psi>
        '''
        return self.get_norm_tensor()^all


    def gate_sandwich(self, G, where, contract):
        '''Scalar quantity <psi|G|psi>
        '''
        bra = self.H
        Gket = self.apply_gate(G, where, contract)
        return (bra|Gket) ^ all


    def compute_hop_expecs(self):
        '''
        Return <psi|H_hop|psi> expectation for the
        hopping terms in (qubit) Hubbard
        '''
        
        psi=self

        E_hop = 0
        bra = psi.H

        X,Y,Z = (qu.pauli(mu) for mu in ['x','y','z'])

        #Horizontal edges
        for direction in ['r','l']:

            Of = Y #face operator

            for (i,j,f) in self.get_edges(direction):
                if f is None:
                    G = ((X&X) + (Y&Y))/2
                    G_ket = self.apply_gate(G, where=(i,j))
            
                else:
                    G = ((X&X&Of) + (Y&Y&Of))/2
                    G_ket = self.apply_gate(G, where=(i,j,f))

                E_hop += (bra|G_ket) ^ all


        #Vertical edges (sign changes!)
        for direction, sign in [('u', -1), ('d', 1)]:

            Of = X #face operator 

            for (i,j,f) in self.get_edges(direction):
                if f is None:
                    G = ((X&X) + (Y&Y))/2
                    G_ket = self.apply_gate(G, where=(i,j))
            
                else:
                    G = ((X&X&Of) + (Y&Y&Of))/2
                    G_ket = self.apply_gate(G, where=(i,j,f))

                E_hop += sign * (bra|G_ket) ^ all

        return E_hop
 
    
    def compute_nnint_expecs(self):
        '''
        Return <psi|H_int|psi> for the nearest-neighbor
        repulsion terms in spinless-Hubbard.
        '''
        
        psi = self
        
        E_int = 0
        bra = psi.H

        for (i, j, _) in self.get_edges('all'):
            #ignore all faces here

            G = number_op() & number_op()
            G_ket = self.apply_gate(G, where=(i,j))

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
        Lx, Ly = self.Lx, self.Ly
        psi = self
        bra = psi.H

        nxy_array = [[None for y in range(Ly)] for x in range(Lx)]

        G = number_op()

        #only finds occupations at *vertices*!
        for x,y in product(range(Lx),range(Ly)):
            
            where = self.vert_coo_map(x,y)
            G_ket = self.apply_gate(G, where=(where,))

            nxy_array[x][y] = (bra | G_ket) ^ all
            

        if return_array: 
            return nxy_array

        return np.sum(nxy_array)            
    

    
    def compute_ham_expec(self, Ham, normalize=True):
        '''Return <psi|H|psi>

        Ham: [SimulatorHam]
            Specifies a two- or three-site gate for each edge in
            the lattice (third site if acting on the possible face
            qubit)
        '''
        psi = self
        bra = psi.H

        E = 0
        
        for where, G in Ham.gen_ham_terms():
            G_ket = self.apply_gate(G, where)
            E += (bra|G_ket) ^ all


        if not normalize:
            return E
        
        normsq = self.get_norm_tensor() ^ all
        return E / normsq

    

    def apply_trotter_gates_(self, Ham, tau):

        for where, exp_gate in Ham.gen_trotter_gates(tau):
            self.apply_gate_(G=exp_gate, where=where)


    #TODO: RECOMMENT
    def apply_stabilizer_from_data_(self, loop_stab_data):
        '''Inplace application of a stabilizer gate that acts with 
        'ZZZZ' on `vert_inds`, and acts on `face_inds` with the operators
        specified in `face_ops`, e.g. 'YXY'.

        vert_inds: sequence of ints (length 4)
        face_ops: string 
        face_inds: sequence of ints (len face_inds==len face_ops)
        '''
        # X, Y, Z = (qu.pauli(mu) for mu in ['x','y','z'])
        # opmap = {'X': X, 'Y':Y, 'Z':Z}
        
        # stab_op = qu.kron(*[opmap[Q] for Q in ('ZZZZ' + face_ops)])
        gates = (qu.pauli(Q) for Q in loop_stab_data['opstring'])
        inds = loop_stab_data['inds']

        for G, where in zip(gates, inds):
            self.apply_gate_(G, where)



    def apply_all_stabilizers_(self, H_stab):
        '''``H_stab`` specifies the active sites and 
        gates, multiplied by ``StabModifier.multiplier``, 
        of all loop stabilizer operators in this lattice.
        '''
        for where, G in H_stab.gen_stabilizer_gates():
            self.apply_gate_(G, where)



def number_op():
    '''Fermionic number operator, aka
    qubit spin-down projector
    '''
    return qu.qu([[0, 0], [0, 1]])
    



# def loop_stab_to_tensor(loop_stab):
#     '''Convert `loop_stabilizer` [loopStabOperator]
#      to a qtn.Tensor of 8 gates.
#     '''
#     X, Y, Z = (qu.pauli(mu) for mu in ['x','y','z'])
#     opmap = {'X': X, 'Y':Y, 'Z':Z}

#     if isinstance(loop_stab, dict):
#         opstring = loop_stab['opstring']
#         inds = loop_stab['inds']
#         # vert_inds = loop_stab['verts']
#         # face_inds = loop_stab['faces']

#     elif isinstance(loop_stab, stabilizers.loopStabOperator):
#         opstring = loop_stab.op_string
#         inds = loop_stab.inds
#         # vert_inds = loop_stab.vert_inds
#         # face_inds = loop_stab.face_inds

#     else: ValueError('Unknown loop stabilizer')
    
#     # numsites = 4 + len(face_inds)

#     gates = (opmap[Q] for Q in opstring)
#     ind_list = (f'q{i}' for i in vert_inds + face_inds)

#     tensors = [qtn.Tensor(gate, inds=k)]
    
#     #new physical indices
#     site_inds = [f'q{i}' for i in where] 
#     # site_inds = [self._phys_ind_id.format(i) for i in where] 

#     #old physical indices joined to new gate
#     bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
#     #replace physical inds with gate/bond inds
#     reindex_map = dict(zip(site_inds, bond_inds))

#     TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=['GATE'])


class MasterHam():
    '''Commodity class that combines a simulator Ham `Hsim`
    and a stabilizer pseudo-Ham `Hstab` to generate each of 
    their gates in order.

    Attributes:
    ----------
    `gen_ham_terms()`: generate Hsim terms followed by Hstab terms

    `gen_trotter_gates(tau)`: trotter gates of Hsim followed by Hstab
    '''
    def __init__(self, Hsim, Hstab):
        self.Hsim=Hsim
        self.Hstab = Hstab


    def gen_ham_terms(self):
        return chain(self.Hsim.gen_ham_terms(),
                     self.Hstab.gen_ham_terms())

    def gen_trotter_gates(self, tau):
        return chain(self.Hsim.gen_trotter_gates(tau),
                     self.Hstab.gen_trotter_gates(tau))


### *********************** ###

class HamStab():
    '''TODO: INSTEAD can store lists of 8 1-site gates?

        Pseudo-Hamiltonian of stabilizers,
        
            `H_stab = multiplier * (S1 + S2 + ... + Sk)`, 
        
        i.e. sum of all the stabilizers in `qlattice` multiplied 
        by the Lagrange `multiplier`.

        Stores 8-site gates corresponding to the loop
        stabilizer operators, to be added to a simulator
        Hamiltonian `H_sim` for TEBD.
        '''

    def __init__(self, qlattice, multiplier = -1.0):
        
        
        self.qlattice = qlattice

        self.multiplier = multiplier
        
        #map coos to `loopStabOperator` objects
        coo_stab_map = qlattice.make_coo_stabilizer_map()

        self._stab_gates = self.make_stab_gate_map(coo_stab_map)
        self._exp_stab_gates = dict()



    def make_stab_gate_map(self, coo_stab_map):
        '''TODO: ALTERED!! NOW MAPS coos to (where, gate) tuples.

        Return
        -------
        `gate_map`: dict[tuple : (tuple, qarray)] 
            Maps coordinates (x,y) in the *face* array (empty 
            faces!) to pairs (where, gate) that specify the 
            stabilizer gate and the sites to be acted on.

        Param:
        ------
        `coo_stab_map`: dict[tuple : dict]
            Maps coordinates (x,y) in the face array of the lattice
            to `loop_stab` dictionaries of the form
            {'inds' : (indices),   'opstring' : (string)}
        '''
        gate_map = dict()

        for coo, loop_stab in coo_stab_map.items():
            #tuple e.g. (1,2,4,5,6)
            inds = tuple(loop_stab['inds'])
            #string e.g. 'ZZZZX'
            opstring = loop_stab['opstring']
            #qarray
            gate = qu.kron(*[qu.pauli(Q) for Q in opstring])
            
            gate *= self.multiplier
            
            # gatelist = [qu.pauli(Q) for Q in opstring]
            gate_map[coo] = (inds, gate)
        
        return gate_map


    def reset_multiplier(self, multiplier):
        '''Reset multiplier, re-compute stabilizer gates 
        and erase previous expm(gates)
        '''
        self.multiplier = multiplier
        self._stab_gates = self.make_stab_gate_map(coo_stab_map)
        self._exp_stab_gates = dict()


    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for acting with 
        the 8-site stabilizer gates on specified sites.

        Note the *keys* in the dictionary are coos and not used!
        '''
        for where, gate in self._stab_gates.values():
            yield (where, gate)


    
    def _get_exp_stab_gate(self, coo, tau):
        '''
        Returns 
        -------
        `exp(tau * multiplier * stabilizer)`: qarray
        
                Expm() of stabilizer centered on empty face at `coo`.

        Params
        -------
        `coo`: tuple (x,y)
            (Irrelevant) location of the empty face that 
            the stabilizer corresponds to. Just a label.
        
        `tau`: float
            Imaginary time for the exp(tau * gate)
        '''
        key = (coo, tau)

        if key not in self._exp_stab_gates:
            where, gate = self._stab_gates[coo]
            el, ev = do('linalg.eigh', gate)
            expgate = ev @ do('diag', do('exp', el*tau)) @ dag(ev)
            self._exp_stab_gates[key] = (where, expgate)
        
        return self._exp_stab_gates[key]


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau*gate)`) pairs for acting
        with exponentiated stabilizers on lattice.
        '''
        for coo in self.empty_face_coos():
            yield self._get_exp_stab_gate(coo, tau)


    def empty_face_coos(self):
        return self._stab_gates.keys()


###  **********************************  ###


class SimulatorHam():
    '''Parent class for simulator (i.e. qubit-space) 
    Hamiltonians. 

    Takes a `qlattice` object to handle lattice geometry/edges, 
    and a mapping `ham_terms` of edges to two/three site gates.
    '''
    
    def __init__(self, qlattice, ham_terms):
        
        self.qlattice = qlattice
        self._ham_terms = ham_terms
        self._exp_gates = dict()


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

    

    def _trotter_gate_group(self, group, x):
        '''Returns mapping of edges (in ``group``) to
        the corresponding exponentiated gates.
        
        Returns: dict[edge : exp(Ham gate)]
        '''
        edges = self.get_edges(group)
        gate_map = {edge : self.get_expm_gate(edge,x) for edge in edges}
        return gate_map
    

    #TODO: add _ internal method
    def get_edges(self, which):
        '''Retrieves (selected) edges from internal 
        qlattice object.
        '''
        return self.qlattice.get_edges(which)


    def Lx():
        return self.qlattice._Lx


    def Ly():
        return self.qlattice._Ly

    
    def ham_params(self):
        '''Relevant parameters. Override for
         each daughter Hamiltonian.
        '''
        pass

    def gen_ham_terms(self):
        pass
    
    def gen_trotter_gates(self, tau):
        pass

## ******************* ##
# Subclass Hamiltonians
## ******************* ##

class SpinlessSimHam(SimulatorHam):
    '''Encoded Hubbard Hamiltonian for spinless fermions,
    encoded as a qubit simulator Ham.

    H =   t  * hopping
        + V  * repulsion
        - mu * occupation
    '''

    def __init__(self, qlattice, t, V, mu):
        '''
        qlattice: [QubitCodeLattice]
                The lattice of qubits specifying the geometry
                and vertex/face sites.
        
        t: hopping parameter
        V: nearest-neighbor repulsion
        mu: single-site chemical potential

        '''
        
        self._t = t
        self._V = V
        self._mu = mu

        terms = self._make_ham_terms(qlattice)

        super().__init__(qlattice, terms)
        

    def _make_ham_terms(self, qlattice):
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
                    terms[(i,j,f)] = sign * t * self._two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())
                
                #three-site
                else:
                    terms[(i,j,f)] = sign * t * self._three_site_hop_gate(edge_dir='vertical')
                    terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('horizontal'):

            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())

            #three-site    
            else:
                terms[(i,j,f)] =  t * self._three_site_hop_gate(edge_dir='horizontal')
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

                v_place = (i,j,f).index(vertex) #vertex is either i or j

                if f is None: #ham_term should act on two sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*2, inds=v_place)

                else: #act on three sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*3, inds=v_place)

        return terms


    def _two_site_hop_gate(self):
        '''Hopping between two vertices, with no face site.
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        return 0.5* ((X&X) + (Y&Y))


    def _three_site_hop_gate(self, edge_dir):
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


    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for every location
        (edge) to be acted on with a Ham term
        '''
        for (i,j,f), gate in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, gate


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau * gate)`) pairs, for each 
        location (edge) `where` and gate exponentiated by
        `tau`. 

        Generated in ordered groups of edges:
        1. Horizontal-even
        2. Horizontal-odd
        3. Vertical-even
        4. Vertical-odd
        '''
        for group in ['he', 'ho', 've', 'vo']:

            for edge, exp_gate in self._trotter_gate_group(group, tau).items():
                
                i,j,f = edge
                where = (i,j) if f is None else (i,j,f)
                yield where, exp_gate

###

class SpinhalfSimHam(SimulatorHam):
    '''Simulator Hamiltonian, acting on qubit space,
    that encodes the Fermi-Hubbard model Ham for 
    spin-1/2 fermions. Each local site is 4-dimensional.

    Gates act on 2 or 3 sites (2 vertices + 1 possible face)
    '''

    def __init__(self, qlattice, t, U):
        
        self._t = t
        self._U = U

        terms = self._make_ham_terms(qlattice)

        super().__init__(qlattice, terms)
    

    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for every location
        (edge) to be acted on with a Ham term
        '''
        for (i,j,f), gate in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, gate


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau * gate)`) pairs, for each 
        location (edge) `where` and gate exponentiated by
        `tau`. 

        Generated in ordered groups of edges:
        1. Horizontal-even
        2. Horizontal-odd
        3. Vertical-even
        4. Vertical-odd
        '''
        for group in ['he', 'ho', 've', 'vo']:

            for edge, exp_gate in self._trotter_gate_group(group, tau).items():
                
                i,j,f = edge
                where = (i,j) if f is None else (i,j,f)
                yield where, exp_gate


    def _make_ham_terms(self, qlattice):
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
            
            spin_up_hop = self._two_site_hop_gate(spin=0)
            spin_down_hop = self._two_site_hop_gate(spin=1)

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] =  sign * t * spin_up_hop
                    terms[(i,j,f)] += sign * t * spin_down_hop
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
                #three-site
                else:
                    terms[(i,j,f)] =  sign * t * self._three_site_hop_gate(spin=0, edge_dir='vertical')
                    terms[(i,j,f)] += sign * t * self._three_site_hop_gate(spin=1, edge_dir='vertical')
                    # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('right+left'):
            
            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_gate(spin=0)
                    terms[(i,j,f)] += t * self._two_site_hop_gate(spin=1)
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
            #three-site    
            else:
                terms[(i,j,f)] =  sign * t * self._three_site_hop_gate(spin=0, edge_dir='horizontal')
                terms[(i,j,f)] += sign * t * self._three_site_hop_gate(spin=1, edge_dir='horizontal')
                # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)
        
        
        if U == 0.0:
            return terms


        G_onsite = self._onsite_int_gate() #on-site spin-spin interaction

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


    def _two_site_hop_gate(self, spin):
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
    

    def _three_site_hop_gate(self, spin, edge_dir):
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


    def _onsite_int_gate(self):
        '''Spin-spin interaction at a single vertex.
        '''
        return number_op() & number_op()
    
    

## ********** ##




def gate_string_split_(TG, where, string, original_ts, bonds_along,
                       reindex_map, site_ix, **compress_opts):

    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault('absorb', 'right')

    # the outer, neighboring indices of each tensor in the string
    neighb_inds = []

    # tensors we are going to contract in the blob, reindex some to attach gate
    contract_ts = []

    for t, coo in zip(original_ts, string):
        neighb_inds.append(tuple(ix for ix in t.inds if ix not in bonds_along))
        contract_ts.append(t.reindex(reindex_map) if coo in where else t)

    # form the central blob of all sites and gate contracted
    blob = qtn.tensor_contract(*contract_ts, TG)

    regauged = []

    # one by one extract the site tensors again from each end
    inner_ts = [None] * len(string)
    i = 0
    j = len(string) - 1

    while True:
        lix = neighb_inds[i]
        if i > 0:
            lix += (bonds_along[i - 1],)

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        inner_ts[i], *maybe_svals, blob = blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)


        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = neighb_inds[j]
        if j < len(string) - 1:
            lix += (bonds_along[j],)

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)


        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break


    # transpose to match original tensors and update original data
    for to, tn in zip(original_ts, inner_ts):
        tn.transpose_like_(to)
        to.modify(data=tn.data)
