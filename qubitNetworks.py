import quimb as qu
import numpy as np
from itertools import product, chain, starmap
import quimb.tensor as qtn
import denseQubits
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
from collections import defaultdict
from numbers import Integral
from autoray import do, dag
import tqdm
import functools
from quimb.tensor.tensor_core import tags_to_oset
from quimb.utils import pairwise, check_opt



def make_auxcon_net(
    Lx,
    Ly,
    phys_dim=2,
    bond_dim=3,
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    aux_tag_id='X{}',
    add_tags={},
    **tn_opts
):
    tag_id = site_tag_id
    ind_id = phys_ind_id
    D = bond_dim
    add_tags = tags_to_oset(add_tags)

    added_tensors = []

    dummy = np.random.rand(D,D,D,D)

    vertex_net = make_vertex_net(
        Lx=Lx, Ly=Ly, phys_dim=phys_dim,
        bond_dim=bond_dim, site_tag_id=site_tag_id,
        phys_ind_id=phys_ind_id, add_tags=add_tags,
        **tn_opts)
    
    k=0
    for fi, fj in product(range(Lx-1), range(Ly-1)):

        if fi % 2 == fj % 2: #add face site & splitting tensors
            
            face_bonds = [phys_ind_id.format(Lx*Ly + k)] + [qtn.rand_uuid()
                                                            for _ in range(2)]
            
            face_tensor = qtn.rand_tensor(
                            shape=[phys_dim, D, D],
                            inds=face_bonds,
                            tags=('FACE', tag_id.format(Lx*Ly + k), *add_tags)
            )
            
            up_left_corner = fi * Ly + fj
            T1 = vertex_net[up_left_corner]
            T2 = vertex_net[up_left_corner+1]
            
            tensor_upper_split = insert_split_tensor(
                T1=T1, T2=T2, face_ind=face_bonds[1],
                Tf=face_tensor, add_tags=['AUX', 'UPPER',
                aux_tag_id.format(up_left_corner), *add_tags]
            )

            down_left_corner = (fi + 1) * Ly + fj
            T1 = vertex_net[down_left_corner]
            T2 = vertex_net[down_left_corner + 1]

            tensor_lower_split = insert_split_tensor(
                T1=T1, T2=T2, face_ind=face_bonds[2],
                Tf=face_tensor, add_tags=['AUX', 'LOWER', 
                aux_tag_id.format(down_left_corner), *add_tags]
            )
            

            added_tensors.append([face_tensor, 
                                tensor_upper_split,
                                tensor_lower_split])
            k+=1
    
    vertex_net |= added_tensors
    return qtn.TensorNetwork(vertex_net.tensors, 
                            structure=site_tag_id)




    
def insert_split_tensor(T1, T2, face_ind, Tf, add_tags=None):
    '''Assuming `T1---T2` are connected by one bond, insert a 
    3-legged tensor `U` between T1 and T2 that connects to the 
    (face) tensor `Tf` through the `face_ind` bond.

    T1-----U------T2
           |
           | (face_ind)
           Tf
    
    In-place modifies the tensor T2, and returns the (random)
    tensor U.
    '''
    
    #current vertex---vertex bond
    bond, = qtn.bonds(T1, T2)

    Dv = T1.ind_size(bond) #size of v---v bond
    Df = Tf.ind_size(face_ind) #size of face-bond    
    
    newbond = qtn.rand_uuid()
    T2.reindex_({bond: newbond})

    return qtn.rand_tensor( shape=(Dv, Dv, Df),
                            inds=(bond, newbond, face_ind),
                            tags=add_tags)
    

def insert_identity_between_tensors(T1, T2, add_tags=None):
    '''Assuming T1---T2 share one bond, replace the bond
    and connect the tensors with an identity matrix.

    Modifies `T1, T2` in place and returns the identity tensor.
    '''
    bond, = qtn.bonds(T1, T2)
    D = T1.ind_size(bond)

    newbond = qtn.rand_uuid()
    T2.reindex_({bond: newbond})

    return qtn.Tensor(data=qu.eye(D),
                      inds=(bond, newbond),
                      tags=add_tags)



def get_halfcoo_between(coo1, coo2):
    i1, j1 = coo1
    i2, j2 = coo2

    return ((i1+i2)/2, (j1+j2)/2)

def make_skeleton_net(
    Lx, 
    Ly,
    phys_dim, 
    bond_dim=3,
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    add_tags={},
    **tn_opts
):
    '''Makes a product state qubit network, for a lattice with 
    dimensions `Lx, Ly` and local site dimension `phys_dim`.    

    Currently, every site is initialized to the `up x up x ...`
    state, i.e. `basis_vec(0)`
    
    Vertex tensors are tagged with 'VERT' and face tensors with 'FACE'.
    In addition, every tensor is tagged with any supplied in ``add_tags``,
    and with a unique site tag (e.g. `Q{k}` for the kth site)

    '''
    
    tag_id = site_tag_id
    ind_id = phys_ind_id

    add_tags = set(add_tags) #none by default

    #default to ``up`` spin at every site
    vert_array = [[qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim)
                    for j in range(Ly)]
                    for i in range(Lx)]
        
    face_array = [[qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim)
                    for j in range(Ly-1)]
                    for i in range(Lx-1)]


    vtensors = [[qtn.Tensor(data = vert_array[i][j], 
                            inds = [ind_id.format(i*Ly+j)],  #[f'q{i*Ly+j}'],f'Q{i*Ly+j}'
                            tags = {tag_id.format(i*Ly+j), 
                                    'VERT'} | add_tags) 
                for j in range(Ly)] 
                for i in range(Lx)]


    # ftensors = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    ftensors = [[None for fj in range(Ly-1)]
                      for fi in range(Lx-1)]
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):
        if i%2 == j%2:
            ftensors[i][j] = qtn.Tensor(data=face_array[i][j],
                                        inds=[ind_id.format(k+Lx*Ly)],
                                        tags={tag_id.format(k+Lx*Ly),
                                              'FACE'} | add_tags)
            k+=1

    

    for i,j in product(range(Lx), range(Ly)):
        
        if i<=Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=bond_dim)
        if j<=Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=bond_dim)


    for i, j in product(range(Lx-1), range(Ly-1)):
        if not ftensors[i][j] is None:
           
            ftensors[i][j].new_bond(vtensors[i][j], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i][j+1], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i+1][j+1], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i+1][j], size=bond_dim)


    vtensors = list(chain.from_iterable(vtensors))
    ftensors = list(chain.from_iterable(ftensors))
    
    alltensors = vtensors + [f for f in ftensors if f]
    # return alltensors
    return qtn.TensorNetwork(alltensors, structure=site_tag_id, **tn_opts)
    


def make_random_net(qlattice, 
                    bond_dim=3, 
                    site_tag_id='Q{}', 
                    phys_ind_id='q{}',
                    add_tags={},
                    **tn_opts
                    ):
    '''
    NOTE: can make much simpler with ``Tensor.randomize``?

    Return a `TensorNetwork` made from random tensors
    structured like `qlattice` i.e. with the same local 
    qu(d)it degrees of freedom. 
    
    Each site has physical index dimension `d = qlattice._local_dim`,
    and is connected to its neighbors with a virtual bond of 
    dimension ``bond_dim``.

    Vertex tensors are tagged with 'VERT' and face tensors with 'FACE'.
    In addition, every tensor is tagged with those supplied in ``add_tags``


    '''
    Lx, Ly = qlattice.lattice_shape
    phys_dim = qlattice.local_site_dim

    #dummy TN, site tensors to be replaced with randomized
    tnet = make_skeleton_net(Lx, Ly, 
                            phys_dim, 
                            bond_dim, 
                            site_tag_id, 
                            phys_ind_id, 
                            add_tags,
                            **tn_opts)

    #replace vertex tensors with randoms
    for i, j in product(range(Lx), range(Ly)):
        
        tid = tuple(tnet.tag_map[site_tag_id.format(i*Ly+j)])
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
            
            tid = tuple(tnet.tag_map[site_tag_id.format(k+Lx*Ly)])
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

    
    return tnet


def make_vertex_net(
    Lx,
    Ly, 
    phys_dim=2,
    bond_dim=3, 
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    add_tags={},
    qlattice=None,
    return_arrays=False
):
    '''Build 2D array of *vertex* tensors, without 
    face sites (i.e. essentially a PEPS).

    Returns: 
    -------
    vtensors: array [ array [qtn.Tensor] ] ]
        (Lx, Ly)-shaped array of qtn.Tensors connected by
        bonds of dimension ``bond_dim``. Vertex tensors are tagged 
        with 'VERT' *and* those supplied in ``add_tags``
    '''

    if qlattice is not None:
        Lx, Ly = qlat.lattice_shape

    add_tags = set(add_tags)

    vert_array = [[qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim) 
                    for j in range(Ly)]
                    for i in range(Lx)]
       

    vtensors = [[qtn.Tensor(data = vert_array[i][j], 
                            inds = [phys_ind_id.format(i*Ly+j)],
                            tags = {site_tag_id.format(i*Ly+j),
                                    'VERT'} | add_tags) 
                for j in range(Ly)] 
                for i in range(Lx)]

    

    for i,j in product(range(Lx), range(Ly)):
        if i < Lx-1:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=bond_dim)
        if j < Ly-1:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=bond_dim)

    vtn = qtn.TensorNetwork(vtensors, structure=site_tag_id)
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
        Hamiltonian for i-time evolution. Should have a
        `gen_trotter_gates(tau)` method.
    
    `compute_extra_fns`: callable or dict of callables, optional
        If desired, can give extra callables to compute at each
        step of TEBD. Each function should take only the current 
        state `qnet` as parameter. Results of the callables will 
        be stored in `self._results`
    
    `contract_opts`: 
        Supplied to 
        :meth:`~denseQubits.QubitLattice.apply_trotter_gates_`
    '''
    def __init__(
        self,
        qnetwork, 
        ham, 
        inplace=True,
        bond_dim=8,
        tau=0.01,
        progbar=True,
        compute_every=None,
        compute_extra_fns=None,
        **contract_opts
    ):

        self.qnet = qnetwork if inplace else qnetwork.copy()

        self.ham = ham #hamiltonian for i-time evolution
        self.tau = tau 
        self.progbar = progbar 
        
        self._n = 0 #current evolution step
        self.iters = [] #stored iterations
        self.energies=[] #stored energies <psi|H|psi>/<psi|psi>

        #how often to compute energy (and possibly other observables)
        self.compute_energy_every = compute_every

        #if other observables to be computed
        self._setup_callback(compute_extra_fns)

        #opts for how to contract gates on lattice tensors
        self._contract_opts = contract_opts



    def _setup_callback(self, fns):
        '''Setup for any callbacks to be computed during 
        imag-time evolution. Defines a function `self._step_callback`
        that computes desired quantities, to be called at each timestep
        (or at whatever steps the energy is evaluated).
        
        fns: callable, or dict of callables, or None
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
        self.qnet.apply_trotter_gates_(  self.ham, 
                                        -self.tau, 
                                        **self._contract_opts)

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
    return np.real(qnet.make_norm()^all)



## ******************************* ##



class QubitEncodeNet(qtn.TensorNetwork):
    '''
        Params:
        -------
        `tn`: TensorNetwork
            State of the tensors in the lattice.

        `qlattice`: QubitLattice
            Specifies the underlying lattice geometry, in 
            particular the shape (Lx, Ly) of the lattice
            and the local dimension of the physical sites,
            e.g. d=2 (4) for simulating spinless (spin-1/2) 
            fermions.
        
        `bond_dim`: int, optional
            The bond dimension to connect local site tensors.
            Defaults to 5.
        
        

        Attributes:
        ----------

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
        

        _face_coo_map: dict[tuple(int): int or None]
            Numbering of face sites. Assigns an integer
            to each location (i,j) in the *face* array,
            and `None` to the empty faces, i.e. lattice
            faces with no physical qubit.

            e.g. for 4x4 vertices (3x3 faces)

            x----x----x----x
            | 16 |    | 17 |
            x----x----x----x
            |    | 18 |    |
            x----x----x----x
            | 19 |    | 20 |
            x----x----x----x
        

        _supergrid: list[ list[ str or None ]], shape (2*Lx-1, 2*Ly-1)
            Array storing tag `tij` if there is a tensor at location
            (i,j), `None` otherwise. The supergrid has coordinates
            for both vertex and face sites -- effectively the vertices
            and faces live only at 'even' coos in the supergrid, while 
            'odd' supergrid coos connect physical sites.


        _face_coo_map_nonempty: dict[tuple(int): int]
            Like `_face_coo_map` but only contains face
            (coo, site) pairs that have a physical qubit
            associated, i.e. drops all (coo, None) pairs.

        '''
    _EXTRA_PROPS = (
        '_qlattice',
        '_site_tag_id',
    )

    def _is_compatible_lattice(self, other):
        return (
            isinstance(other, QubitEncodeNet) and
            all(getattr(self, e)==getattr(other, e)
                for e in QubitEncodeNet._EXTRA_PROPS)
            )
    

    def __and__(self, other):
        new = super().__and__(other)
        if self._is_compatible_lattice(other):
            new.view_as_(QubitEncodeNet,
                        like=self,
                        qlattice=self.qlattice,
                        site_tag_id=self.site_tag_id)
        return new
    

    def __or__(self, other):
        new = super().__or__(other)
        if self._is_compatible_lattice(other):
            new.view_as_(QubitEncodeNet,
                        like=self,
                        qlattice=self.qlattice,
                        site_tag_id=self.site_tag_id)
        return new



    @property
    def qlattice(self):
        '''Internal ``QubitLattice`` object
        '''
        return self._qlattice

    @property
    def phys_dim(self):
        '''Local physical dimension of qu(d)it sites.
        '''        
        return self._qlattice.local_site_dim

    @property
    def site_tag_id(self):
        '''Format string for the tag identifiers of local sites
        '''
        return self._site_tag_id


    # def copy(self):
    #     return self.__class__(
    #                     self, 
    #                     qlattice=self.qlattice,
    #                     site_tag_id=self.site_tag_id)

    # __copy__ = copy


    # def _site_tid(self, k):
    #     '''Given the site index `k` in {0,...,N},
    #     return the `tid` of the local tensor
    #     '''
    #     #'q{k}'
    #     index = self._phys_ind_id.format(k)
    #     return self.ind_map[index]

    
    def _build_supergrid(self):
        '''Sets up the 'bare' supergrid, with unique tags
        for all the vertex- and face-qubits.
        '''
        Lx, Ly = self.Lx, self.Ly

        supergrid = [[None for _ in range(2*Ly - 1)] 
                           for _ in range(2*Lx - 1)]

        # supergrid = dict()

        for i, j in product(range(Lx), range(Ly)):
            
            vertex_tag = self.vert_coo_tag(i, j)
            supergrid[2 * i][2 * j] = set([vertex_tag])

        for i, j in product(range(Lx-1), range(Ly-1)):
            
            if i%2==j%2:
                face_tag = self.face_coo_tag(i, j)
                supergrid[i + 1][j + 1] = set([face_tag])

        return supergrid

    #Depend on lattice geometry
    
    def vert_coo_map(self, i=None, j=None):
        '''Maps location (i,j) in vertex lattice
        to the corresponding site number.
        '''
        if not hasattr(self, '_vert_coo_map'):
            self._vert_coo_map = dict(np.ndenumerate(
                            self.qlattice.vert_array))


        if (i is not None) and (j is not None):
            return self._vert_coo_map[(i,j)]
        
        return self._vert_coo_map



    def face_coo_map(self, i=None, j=None):
        '''Maps location (i,j) in *face* lattice
        to the corresponding site number.
        '''
        if not hasattr(self, '_face_coo_map'):
            self._face_coo_map = dict(np.ndenumerate(
                            self.qlattice.face_array))


        if (i is not None) and (j is not None):
            return self._face_coo_map[(i,j)]

        return self._face_coo_map



    def supergrid(self, x=None, y=None):
        '''Returns the set of tags corresponding
        to supercoo `x, y`. If no coo is specified,
        returns the whole dictionary.
        '''
        if not hasattr(self, '_supergrid'):
            self._supergrid = self._build_supergrid()
        
        if (x is not None) and (y is not None):
            return self._supergrid[x][y]
        
        return self._supergrid


    def update_supergrid_(self, x, y, tag):
        '''Add `tag` to the set of tags at supergrid 
        coordinate `x, y`. Does not complain if the
        tag was already there.
        '''
        if not hasattr(self, '_supergrid'):
            self._supergrid = self._build_supergrid()
        
        if self._supergrid[x][y] is None:
            self._supergrid[x][y] = [tag]
        else:
            self._supergrid[x][y].add(tag)


    def vert_coo_tag(self,i,j):
        '''Tag for site at vertex-coo (i,j)
        '''
        k = self.vert_coo_map(i,j)
        return self.site_tag_id.format(k)
    

    def face_coo_tag(self, fi, fj):
        '''Tag for site at face-coo (fi,fj)
        '''
        k = self.face_coo_map(fi, fj)

        if k is None:
            return None

        return self.site_tag_id.format(k)

        

    def maybe_convert_face(self, where):
        '''Returns None if ``where`` is the coo
        of an empty face.
        Converts tuple coos (x,y) into the 
        corresponding 'Q{k}' tag
        '''
        # if isinstance(where, Integral):
        #     return self.site_tag_id.format(where)
        
        if not isinstance(where, str):
            try:
                fi, fj = map(int, where)
                return self.face_coo_tag(fi, fj)
            except (ValueError, TypeError):
                pass
        
        return where

        
    
    def gen_vertex_sites(self):
        ''' Generator, same as ``range(num_vertices)``
        '''
        return self.qlattice.gen_vertex_sites()

    
    def gen_face_sites(self):
        '''Generator, same as ``range(num_verts, num_sites)``
        '''
        return self.qlattice.gen_face_sites()
        # return product(range(self.Lx-1), range(self.Ly-1))
    

    def gen_all_sites(self):
        '''Generator, same as ``range(num_sites)``
        '''
        return self.qlattice.gen_all_sites()


    def gen_vertex_coos(self):
        '''Generate the (i,j) coordinates for all 
        vertices in the lattice.
        '''
        return product(range(self.Lx), range(self.Ly))
    
    
    def gen_face_coos(self):
        '''Generate the (i,j) coordinates for not-None
        (non-empty) face sites in the lattice
        '''
        for coo, site in self._face_coo_map.items():
            if site is not None:
                yield coo
    
    def gen_supergrid_coos(self):
        return product(range(-1 + 2*self.Lx), range(-1 + 2*self.Ly))



    def _canonize_supergrid_row(self, x, sweep, yrange=None, **canonize_opts):
        '''Canonize all bonds between tensors in the xth row
        of the supergrid.

        Automatically ignores the 'empty' coos in supergrid,
        i.e. picks only the nodes occupied by tensors.
        '''
        check_opt('sweep',sweep,('right','left'))
        
        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)

        ordered_row_tags = self._supergrid_row_slice(x, yrange, sweep)

        for tag1, tag2 in pairwise(ordered_row_tags):
            self.canonize_between(tag1, tag2, **canonize_opts)



    def _canonize_supergrid_column(self, y, sweep, xrange=None, **canonize_opts):
        check_opt('sweep',sweep,('up','down'))

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)
        
        ordered_col_tags = self._supergrid_column_slice(y, xrange, sweep)

        for tag1, tag2 in pairwise(ordered_col_tags):
            self.canonize_between(tag1, tag2, **canonize_opts)

    

    def _canonize_supergrid_row_around(self, x, around=(0,1)):
        #sweep to the right
        self._canonize_supergrid_row(x, sweep='right', yrange=(0, min(around)))
        #sweep to the left
        self._canonize_supergrid_row(x, sweep='left', yrange=(max(around), 2*self.Ly-2))
    
    


    def _compress_supergrid_row(self, x, sweep, yrange=None, **compress_opts):
        check_opt('sweep', sweep, ('right', 'left'))
        compress_opts.setdefault('absorb', 'right')

        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)
        
        ordered_row_tags = self._supergrid_row_slice(x, yrange, sweep) 

        for tag1, tag2 in pairwise(ordered_row_tags):
            self.compress_between(tag1, tag2, **compress_opts)



    def _compress_supergrid_column(self, y, sweep, xrange=None, **compress_opts):
        check_opt('sweep', sweep, ('up', 'down'))
        compress_opts.setdefault('absorb', 'right')

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        ordered_column_tags = self._supergrid_column_slice(y, xrange, sweep)

        for tag1, tag2 in pairwise(ordered_column_tags):
            self.compress_between(tag1, tag2, **compress_opts)


    # def _fill_column_with_identities(self, y, xrange=None):
    #     if xrange is None:
    #         xrange = (0, 2*self.Lx - 2)
        
    #     column_nodes = self._supergrid_column_slice(y, xrange, get='row', sweep='down')

    #     for row1, row2 in pairwise(column_nodes):
    #         where1, where2 = grid(row1, y), grid(row2, y)
            
    #         if row2 - row1 > 1 and bool(self.list_bonds_between(where1, where2)):
                
    #             midcoo = 
    #             self.insert_identity_between_(where1, where2, tags=[f''])


    def _shift_tensor_to_right(self, left_coo):
        x, yleft = left_coo

        yright = yleft + 1

        grid = self.supergrid
        
        if self.supergrid(x, yright) is not None:
            raise ValueError(f"Supercoo {x},{yright} is already occupied!")

        
        elif (x > 0) and (x < 2*self.Lx-2):
            #in the bulk, check for a bond to 'merge' with
            try:
                bond, = self.bond(grid(x-1, yright), grid(x+1, yright))
                newid_tag = f'IX{x},Y{yright}'

                self.insert_identity_between_(where1=grid(x-1, yright), 
                                            where2=grid(x+1, yright),
                                            tags=[newid_tag])

                self.update_supergrid_(x=x, y=yright, tag=newid_tag)

                self.contract_((grid(x, yleft), grid(x, yright)), which='any')

            except ValueError:
                #there was no bond, just move right
                self.update_supergrid_(x=x, y=yright, tag=grid(x, yleft))

            
        else:
            #add tensor's tag to location on the right
            self.update_supergrid_(x=x, y=yright, tag=grid(x, yleft))





    def _supergrid_row_slice(self, x, yrange, sweep, get='tag'):
        '''Inclusive, 'directed' slice of self._supergrid, where 
        `yrange` is INCLUSIVE. Automatically drops the `None` tags.

        Params:
        ------
        x: int
            The supergrid row, from {0, 1, ..., 2*Lx-2}
        
        yrange: tuple(int, int)
            Inclusive range of columns to select, max yrange
            is (0, 2*Ly-2)
        
        sweep: {'left', 'right'}
            Direction of the slice.
        
        get: {'tag', 'col'}
            What to generate. TODO:Get rid of this?
        
        Returns:
        -------
        Generator of non-None tags/supercolumn labels
        '''

        if sweep == 'right':
            for y in range(min(yrange), max(yrange)+1):
                tag = self.supergrid(x,y)
                
                if tag is not None:
                    yield {'tag': tag, 'col': y}[get]


        elif sweep == 'left':
            for y in range(max(yrange), min(yrange)-1, -1):
                tag = self.supergrid(x, y)
                
                if tag is not None:
                    yield {'tag': tag, 'col': y}[get]
    


    def _supergrid_column_slice(self, y, xrange, sweep, get='tag'):
        '''Inclusive, 'directed' slice of self._supergrid, where 
        `xrange` is INCLUSIVE. Automatically drops the `None` tags.

        y: int
            The supergrid column
        
        xrange: tuple(int, int)
            Inclusive range of rows to select. Max
            xrange is (0, 2*self.Lx - 2)
        
        sweep: {'up', 'down'}
            Direction/order of the slice.
        
        get: {'tag', 'row'}
            Whether to generate the str `tag` or the integer `row`.
            TODO: Get rid of this?
        '''

        if sweep == 'down':
            for x in range(min(xrange), max(xrange) + 1):
                tag = self.supergrid(x, y)
                if tag is not None:
                    yield {'tag': tag, 'row': x}[get]
        
        elif sweep == 'up':
            for x in range(max(xrange), min(xrange) - 1, -1):
                tag = self.supergrid(x, y)
                if tag is not None:
                    yield {'tag': tag, 'row': x}[get]


    def _contract_boundary_from_left_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='up',
        layer_tag=None,
        **compress_opts
    ):
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
        }[compress_sweep]


        grid = self.supergrid


        for y in range(min(yrange), max(yrange)):
            #
            #     ●──●──       ●──
            #     │  │         ║
            #     I  ●──  ==>  ●──
            #     │  │         ║
            #     ●──●──       ●──
            #
            left_col = list(self._supergrid_column_slice(y, xrange, sweep='down', get='row'))
            right_col = list(self._supergrid_column_slice(y + 1, xrange, sweep='down', get='row'))
            
            for x in range(min(xrange), max(xrange)+1):
                
                occupation = (x in left_col, x in right_col)


                if occupation == (True, True):
                    #     │  │         ║
                    #     ●──●──       ●──
                    #     │  │         ║
                    self.contract_((grid(x, y), grid(x, y+1)), which='any')
                    # self.update_supergrid_(x, y+1, grid(x,y))

                elif occupation == (True, False):
                    #     │  │         │
                    #     ●  │         ●──
                    #     │  │         │
                    # no contraction, but shift tensor rightward
                    # so that it doesn't 'fall behind'
                    self._shift_tensor_to_right(left_coo=(x,y))
                    
                    
                elif occupation == (False, True):
                    #     │  │         │
                    #     |  ●──       ●──
                    #     │  │         │
                    self.insert_identity_between_(where1=grid(x, y), 
                                                  where2=grid(x+1, y), 
                                                  tags=[f'IX{x},Y{y}'])

                    self.update_supergrid_(x, y, f'IX{x},Y{y}')
                    
                    self.contract_(grid(x, y), grid(x, y+1))

                else: # no tensor on either column
                    continue
    
    

    def get_edges(self, which):
        '''
        Returns: 
        --------
        edges: list[ tuple(int or None)]
            List of 3-tuples, where (i,j,f) 
            denotes vertices i,j and face f
            (int or None)

        Param:
        ------
        which: {'u', 'd', 'r', 'l', 'he', 'ho', 
                've', 'vo', 'horizontal', 'all'}
        '''
        return self.qlattice.get_edges(which)


    @property
    def Lx(self):
        return self.qlattice._Lx
    
    @property
    def Ly(self):
        return self.qlattice._Ly

    @property
    def num_sites(self):
        return self.qlattice.num_sites

    @property
    def num_verts(self):
        return self.qlattice.num_verts

    @property
    def num_faces(self):
        return self.qlattice.num_faces

    @property
    def codespace_dims(self):
        '''List like [phys_dim] * num_verts,
        i.e. dimensions for vertex subspace
        '''
        return self.qlattice.codespace_dims

    @property
    def simspace_dims(self):
        '''List like [phys_dim] * num_sites,
        i.e. dimensions for full qubit space
        '''
        return self.qlattice.simspace_dims


    def bond(self, where1, where2):
        '''Get index (should only be one!) of bond connecting 
        the sites. Can take ints or tags, but not coos.
        '''
        bond, = self[where1].bonds(self[where2])
        return bond


    def list_bonds_between(self, where1, where2):
        return list(self[where1].bonds(self[where2]))


    def graph(self, fix_lattice=True, fix_tags=[], **graph_opts):
        '''
        TODO: DEBUG ``fix_tags`` PARAM

        Overloading TensorNetwork.graph() for convenient
        lattice-fixing when ``fix_lattice`` is True (default).
        '''
        
        graph_opts.setdefault('color', ['VERT','FACE','GATE'])
        graph_opts.setdefault('show_tags', False)

        if fix_lattice == False: 
            super().graph(**graph_opts)
        
        else:
            # try:                
            Lx,Ly = self.Lx, self.Ly
            
            LATCX, LATCY = 1.5, 2

            fix_verts = {(f'Q{i*Ly+j}', *fix_tags): (LATCY*j, -LATCX*i) 
                        for i,j in product(range(Lx),range(Ly))}
            
            fix_faces, k = dict(), 0
            for i,j in product(range(Lx-1), range(Ly-1)):
                if i%2 == j%2:
                    fix_faces.update({(f'Q{k+(Lx*Ly)}', *fix_tags): 
                                      (LATCY*(j+0.5), -LATCX*(i+0.5)) })
                    k+=1

            fix = {**fix_verts, **fix_faces}
                
            super().graph(fix=fix, show_inds=True,**graph_opts)
            
            # except:
            #     super().graph(**graph_opts)


    # def exact_projector_from_matrix(self, Udag_matrix):
    #     Nfermi, Nqubit = self.num_verts(), self.num_sites()

    #     if Udag_matrix.shape != (2**Nfermi, 2**Nqubit):
    #         raise ValueError('Wrong U* shape')
        
    #     sim_dims = self.simspace_dims()
    #     code_dims = self.codespace_dims()
        
    #     sim_inds = [f'q{i}' for i in range(Nqubit)]
        
    #     Udagger = Udag_matrix.reshape(code_dims+sim_dims)

    #     Udagger = qtn.Tensor()



    # def add_tags(self, tags, inplace=False):

    #     net = self if inplace else self.copy()

    #     newtags = tags_to_oset(tags)

    #     for T in net.tensors:
    #         T.modify(tags = newtags | T.tags)
        
    #     return net
    
    # add_tags_ = functools.partialmethod(add_tags, inplace=True)



    def flatten(self, inplace=False, fuse_multibonds=True):
        '''Contract all tensors corresponding to each site into one
        '''
        net = self if inplace else self.copy()

        for k in net.gen_all_sites():
            net ^= k
        
        if fuse_multibonds:
            net.fuse_multibonds_()
        
        return net
        # return net.view_as_(QubitEncodeNetFlat)

    flatten_ = functools.partialmethod(flatten, inplace=True)


    def absorb_face_left(self, face_coo, inplace=False, fuse_multibonds=True):
        '''NOTE: CURRENTLY ONLY FOR FLAT NETWORKS
        Need way to do one layer at a time for sandwiches.
        partition? 
        '''
        tn = self if inplace else self.copy()

        face_tag = tn.maybe_convert_face(face_coo)
        
        fi, fj = face_coo

        #tags for corner vertex sites
        ul_tag = tn.vert_coo_tag(fi, fj)
        ur_tag = tn.vert_coo_tag(fi, fj + 1)
        dl_tag = tn.vert_coo_tag(fi + 1, fj)
        dr_tag = tn.vert_coo_tag(fi + 1, fj + 1)

        #corner bonds
        ul_bond = tn.bond(ul_tag, face_tag)
        ur_bond = tn.bond(ur_tag, face_tag)
        dl_bond = tn.bond(dl_tag, face_tag)
        dr_bond = tn.bond(dr_tag, face_tag)


        face_tensor = tn[face_tag]
        
        #split face tensor into two, upper/lower tensors
        tensors = face_tensor.split(
                    left_inds=(ul_bond, ur_bond),
                    right_inds=(dl_bond, dr_bond),
                    get=None,
                    ltags=['UPPER','SPLIT'],
                    rtags=['LOWER','SPLIT'])

        tn.delete(face_tag)
        tn |= tensors

        # return tn

        # Absorb split-tensors into the vertices
        tn.contract_((ul_tag, 'UPPER'))
        tn.contract_((dl_tag, 'LOWER'))

        tn[ul_tag].drop_tags([face_tag, 'FACE', 'UPPER'])
        tn[dl_tag].drop_tags([face_tag, 'FACE', 'LOWER'])

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn


    absorb_face_left_ = functools.partialmethod(absorb_face_left,
                                                inplace=True)

    def reshape_face_to_cross(self, i, j, inplace=False):
        tn = self if inplace else self.copy()

        face_tag = tn.face_coo_tag(i, j)
    
        corner_coos = tn.corner_coos_around_face(i, j)
        corner_tags = list(starmap(tn.vert_coo_tag, corner_coos))
        # corner_tags = [tn.vert_coo_tag(x, y) for (x, y) in
        #                 tn.corner_coos_around_face(i, j)]
        

        for k, ctag in enumerate(corner_tags):
            
            tn.insert_identity_between_(face_tag, ctag, tags=[f'ID{k}'])
            
            next_ctag = corner_tags[0] if k==3 else corner_tags[k+1]
            #insert I between this and 'next' corner in the square
            tn.insert_identity_between_(ctag, next_ctag, tags=[f'ID{k}'])        


        for k, ccoo in enumerate(corner_coos):
            tn ^= f'ID{k}'

            next_ccoo = corner_coos[0] if k==3 else corner_coos[k+1]

            mid_supercoo = tn.supergrid_coo_between_verts(ij1=ccoo, ij2=next_ccoo)
            
            # v1 = tn.vert_coo_map(*ccoo)
            # v2 = tn.vert_coo_map(*next_ccoo)

            mid_tag = 'IX{},Y{}'.format(*mid_supercoo)

            tn.retag_({f'ID{k}': mid_tag})
            tn.update_supergrid_(*mid_supercoo, tag=mid_tag)

        return tn




    def insert_identity_between(self, where1, where2, tags=None, inplace=False):
        '''Inserts an identity tensor at the bond between tags `where1, where2`.

        The new identity is tagged with the given `ident_tags`.
        '''
        tn = self if inplace else self.copy()

        T1, T2 = tn[where1], tn[where2]
        I = insert_identity_between_tensors(T1, T2, add_tags=tags)
        tn |= I


    insert_identity_between_ = functools.partialmethod(insert_identity_between, 
                                                       inplace=True)
    
    def corner_coos_around_face(self, i, j, order='clockwise'):
        '''For a face-site ij (in the *face* array),
        return the *vertex* coos of the corners bounding
        the face, generated starting in the upper-left.

        e.g. (if clockwise:)

        (i, j) ------> (i, j+1)
          |               |
          ^     [i,j]     v
          |               |
        (i+1, j) <--- (i+1, j+1)
        '''
        return {'clockwise': [(i,j), (i, j+1), (i+1, j+1), (i+1, j)],
                'countwise': [(i,j), (i+1, j), (i+1, j+1), (i, j+1)]
            }[order]


    def supergrid_coo_between_verts(self, ij1, ij2):
        '''Returns the *supergrid* coordinates lying halfway between
        the two given *vertex* coordinates.
        '''
        i1, j1 = ij1 #supercoo = (2i, 2j)
        i2, j2 = ij2

        x = i1 + i2
        y = j1 + j2
        #supercoo in-between is (2i1+2i2)/2, (2j1+2j2)/2
        return (x, y)


    # def canonize_superrow(self, i, sweep, yrange=None, **canonize_opts):
    #     check_opt('sweep',sweep,('right','left'))

    #     if yrange is None:
    #         yrange = (0, self._Ly-1)
        
    #     if sweep == 'right':
    #         for y in range(min(yrange), max(yrange), +1):
    #             # if 
    #             self.canonize_between((i, j), (i, j+1), **canonize_opts)
        
    #     else:
    #         for j in range(max(yrange), min(yrange),-1):
    #             self.canonize_between((i, j), (i, j-1), **canonize_opts)


## ********************* ##
## End QubitEncodeNet class
## ********************* ##


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


class QubitEncodeVector(QubitEncodeNet,
                        qtn.TensorNetwork):

    _EXTRA_PROPS = (
        '_qlattice',
        '_site_tag_id',
        '_phys_ind_id'
    )
    
            
    def __init__(
            self, 
            tn, 
            qlattice, 
            site_tag_id = 'Q{}',
            phys_ind_id = 'q{}',
            **tn_opts
        ):
        
        #shortcut for copying QEN vectors
        if isinstance(tn, QubitEncodeVector):
            self._qlattice = tn.qlattice
            self._site_tag_id = tn.site_tag_id
            self._phys_ind_id = tn.phys_ind_id
            
            super().__init__(tn)
            return

        self._qlattice = qlattice
        self._site_tag_id = site_tag_id
        self._phys_ind_id = phys_ind_id

        
        self._vert_coo_map = self.vert_coo_map()
        self._face_coo_map = self.face_coo_map()

        # self._face_coo_map_nonempty = self.face_coo_map_nonempty()
            
        super().__init__(tn, **tn_opts)

    

    def copy(self):
        return self.__class__(
                        tn=self, 
                        qlattice=self.qlattice,
                        site_tag_id=self.site_tag_id,
                        phys_ind_id=self.phys_ind_id)

    __copy__ = copy
        

    @classmethod
    def rand_from_qlattice(cls, qlattice, bond_dim=3, **tn_opts):
        '''Make a random QubitEncodeVector from specified `qlattice`.

        Params:
        -------
        qlattice: QubitLattice
            Specify lattice geometry and local site dimension
        
        bond_dim: int
            Size of lattice bonds (like D in PEPS)
        '''
        rand_tn = make_random_net(qlattice=qlattice,
                                bond_dim=bond_dim,
                                **tn_opts)
                                
        return cls(tn=rand_tn, qlattice=qlattice)


    @classmethod
    def rand(cls, Lx, Ly, phys_dim=2, bond_dim=3, **tn_opts):
        qlat = denseQubits.QubitLattice(Lx=Lx, Ly=Ly, local_dim=phys_dim)
        rand_tn = make_random_net(qlattice=qlat,
                                bond_dim=bond_dim,
                                **tn_opts)
        return cls(tn=rand_tn, qlattice=qlat)                                




    @property
    def phys_ind_id(self):
        '''Format string for the physical indices
        '''
        return self._phys_ind_id
    
    def _vert_coo_ind(self,i,j):
        '''Index id for site at vertex-coo (i,j)
        '''
        k = self.vert_coo_map(i,j)
        return self.phys_ind_id.format(k)


    def _face_coo_ind(self, fi, fj):
        '''Index id for site at face-coo (fi,fj)
        '''
        k = self.face_coo_map(fi, fj)
        return None if k is None else self.phys_ind_id.format(k)
        # if k is None:
        #     return None
        # return self.phys_ind_id.format(k)
        
        
    def vec_to_dense(self, normalize=True):
        '''Return this state as dense vector, i.e. a qarray with 
        shape (-1, 1), in the order assigned to the local sites.
        '''
        inds_seq = (self._phys_ind_id.format(i) 
                    for i in self.qlattice.all_sites())

        psid = self.to_dense(inds_seq).reshape(-1,1)

        if not normalize:
            return psid
        
        return psid / np.linalg.norm(psid)



    def make_norm(self, layer_tags=('KET','BRA')):
        '''<psi|psi> as an uncontracted ``QubitEncodeNet``.
        '''
        ket = self.copy()
        ket.add_tag(layer_tags[0])

        bra = ket.H.retag({layer_tags[0]: layer_tags[1]})
        return ket | bra
    


    def _norm_scalar(self):
        '''Scalar quantity <psi|psi>
        '''
        return self.make_norm()^all


    def apply_gate(
        self,
        G, 
        where,
        inplace=False, 
        contract=False
        ):
        '''Apply array `G` acting at sites `where`,
        preserving physical indices. Uses `self._phys_ind_id`
        and the integer(s) `where` to apply `G` at the
        correct sites.

        Params:
        ------            
        G : array
            Gate to apply, should be compatible with 
            shape ([physical_dim] * 2 * len(where))
        
        where: sequence of ints
            The sites on which to act, using the (default) 
            custom numbering that labels both face and vertex
            sites.
        
        inplace: bool, optional
            If False (default), return copy of TN with gate applied
        
        contract: {False, True, 'split'}, optional
            False (default) leave all gates uncontracted
            True contract gates into one tensor in the lattice
            'split' uses tensor_2d.gate_split method for two-site gates

        '''

        # if isinstance(G, qtn.TensorNetwork):
        #     self.apply_mpo(G, where, inplace, contract)

        psi = self if inplace else self.copy()

        #G can be a one-site gate
        if isinstance(where, Integral): 
            where = (where,)

        numsites = len(where) #gate acts on `numsites`

        dp = self.phys_dim #local physical dimension

        G = maybe_factor_gate_into_tensor(G, dp, numsites, where)

        #new physical indices 'q{i}'
        site_inds = [self.phys_ind_id.format(i) for i in where] 

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
    
            original_ts = [psi[k] for k in where]

            bonds_along = [next(iter(qtn.bonds(t1, t2)))
                       for t1, t2 in qu.utils.pairwise(original_ts)]

            
            gss_opts = {'TG' : TG,
                        'where' : where,
                        'string': where,
                        'original_ts' : original_ts,
                        'bonds_along' : bonds_along,
                        'reindex_map' : reindex_map,
                        'site_ix' : site_inds,
                        'info' : None}

            qu.tensor.tensor_2d.gate_string_split_(**gss_opts)
            return psi                        


        else:
            raise ValueError('Unknown contraction requested')


    apply_gate_ = functools.partialmethod(apply_gate, inplace=True)


    def apply_mpo(self, mpo, where, inplace=False, contract=False):
        '''For now assume mpo and self have same tagging conventions
        '''
        psi = self if inplace else self.copy()

        nsites = mpo.nsites

        if len(where) != nsites:
            raise ValueError("Wrong number of sites!")
        
        #reset index id (automatically updates tensors)
        mpo.lower_ind_id = 'b{}'
        mpo.upper_ind_id = 'k{}'

        mpo_site_tag = mpo.site_tag_id

        #physical indices like 'q1', 'q4', etc
        site_inds = [psi.phys_ind_id.format(i) for i in where]

        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(nsites)]
        
        
        Ts = [None]*nsites
        for j in range(nsites):
            Ts[j] = mpo[j].reindex({f'k{j}': site_inds[j],
                                    f'b{j}': bond_inds[j]})
            Ts[j].retag_({mpo_site_tag.format(j): 'GATE'})
        

        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))
        

        if contract == False:
                
            psi.reindex_(reindex_map)
            for T in Ts:
                psi |= T
            
            return psi            

        
        elif contract == True:
            #just contract the physical leg(s)
            psi.reindex_(reindex_map)
            
            #sites that used to have physical indices
            site_tids = psi._get_tids_from_inds(bond_inds, which='any')
           
            # pop the sites (inplace), contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]

            #contract physical indices ~ MPO|psi
            for k in range(nsites):
                psi |= qtn.tensor_contract(pts[k], Ts[k])

            return psi

        else:
            raise NotImplementedError('Approx. contraction for MPO gates')
        

    apply_mpo_ = functools.partialmethod(apply_mpo, inplace=True)
    

    def _exact_local_gate_sandwich(self, G, where, contract):
        '''Exactly contract <psi|G|psi>
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


    def compute_occs_expecs(self, return_array=False):
        '''
        Compute local occupation/number expectations,
        <psi|n_xy|psi>

        return_array: bool
            Whether to return 2D array of local number 
            expectations. Defaults to false, in which case
            only the total sum is returned.
        '''
        Lx, Ly = self.Lx, self.Ly
        bra = self.H

        nxy_array = [[None for _ in range(Ly)] for _ in range(Lx)]

        G = number_op()

        #only finds occupations at *vertices*
        for i,j in product(range(Lx), range(Ly)):
            
            where = self.vert_coo_map(i,j)
            G_ket = self.apply_gate(G, where=(where,))

            nxy_array[i][j] = (bra | G_ket) ^ all
            

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
        bra = self.H

        E = 0

        for where, G in Ham.gen_ham_terms():
            G_ket = self.apply_gate(G, where)
            E += (bra|G_ket) ^ all


        if not normalize:
            return E
        
        normsq = self.make_norm() ^ all
        return E / normsq


    def compute_mpo_ham_expec(self, Hmpo, normalize=True):
        '''Same as ``compute_ham_expec`` but for Hamiltonian 
        made from a sum of MPOs rather than raw qarrays.
        '''
        bra = self.H

        E = 0
        for where, mpo in Hmpo.gen_ham_terms():
            E += (bra|self.apply_mpo(mpo, where, contract=True))^all
        
        if not normalize:
            return E
        
        normsquared = self.make_norm()^all
        return E / normsquared


    def apply_trotter_gates_(self, Ham, tau, **contract_opts):

        for where, exp_gate in Ham.gen_trotter_gates(tau):
            self.apply_gate_(G=exp_gate, where=where, **contract_opts)



    #TODO: RECOMMENT
    def apply_stabilizer_from_data_(self, loop_stab_data):
        '''Inplace application of a stabilizer gate.
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

    
    def check_dense_energy(self, Hdense, normalize=True):
        
        psi_d = self.net_to_dense()
        
        psiHpsi = (psi_d.H @ Hsim @ psi_d).item()

        if not normalize:
            return psiHpsi
        
        normsq = self.make_norm() ^ all
        return psiHpsi / normsq
        

    def dense_inner_product(self, dense_bra):

        dense_bra = qu.qu(dense_bra)/np.linalg.norm(dense_bra)
        
        dense_ket = self.net_to_dense(normalize=True)

        return dense_bra.H @ dense_ket


####################################################

class QubitEncodeVectorAUX(QubitEncodeVector,
                            QubitEncodeNet,
                            qtn.TensorNetwork):
    
    _EXTRA_PROPS = (
        '_qlattice',
        '_site_tag_id',
        '_phys_ind_id',
        '_aux_tag_id'
    )
    
    def __init__(
            self, 
            tn, 
            qlattice, 
            site_tag_id='Q{}',
            phys_ind_id='q{}',
            aux_tag_id='X{}',
            **tn_opts
        ):
        
        #shortcut for copying QEN vectors
        if isinstance(tn, QubitEncodeVectorAUX):
            self._aux_tag_id = tn.aux_tag_id
            super().__init__(tn)
            return

        self._aux_tag_id = aux_tag_id
        
        self._aux_tensor_tags = self._find_aux_tensors()

        super().__init__(tn=tn, qlattice=qlattice,
                        site_tag_id=site_tag_id, 
                        phys_ind_id=phys_ind_id, 
                        **tn_opts)

    @property
    def aux_tag_id(self):
        return self._aux_tag_id

    @property
    def aux_tensor_tags(self):
        return self._aux_tensor_tags
    


    def aux_coo_map(self, x=None, y=None):
        '''Maps coordinates like (i, j + 1/2) to a
        tag 'X{k}' if there is an auxiliary tensor
        there.
        '''
        if not hasattr(self, '_aux_coo_map'):

            aux_coos = dict()

            #check every vertex for an associated aux-tensor
            for vcoo, vsite in self.vert_coo_map().items():
                
                maybe_aux_tag = self.aux_tag_id.format(vsite)

                if maybe_aux_tag in self.tags:
                    i,j = vcoo
                    aux_coos.update({(i, j + 0.5): maybe_aux_tag})


            self._aux_coo_map = aux_coos


        if (x is not None) and (y is not None):
            return self._aux_coo_map[(x, y)]
        
        return self._aux_coo_map
        







####################################################


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
        qlattice: QubitLattice
            Lattice of qubits specifying the geometry
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
        
    
    def get_term_at(self, i, j, f=None):
        '''Array acting on edge `(i,j,f)`.
        `i,j` are vertex sites, optional `f` is 
        the face site.
        '''
        return self._ham_terms[(i,j,f)]


    def ham_params(self):
        '''Ham coupling constants

        t: hopping parameter,
        V: nearest-neighbor repulsion,
        mu: chemical potential
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
        


################################

class MPOSpinlessHam():

    def __init__(self, qlattice, t, V, mu):

        self._t = t
        self._V = V
        self._mu = mu

        self._site_tag_id = 'Q{}'
        self._ham_terms = self._make_ham_terms(qlattice)


    def ham_params(self):
        return (self._t, self._V, self._mu)
    

    def get_term_at(self, i, j, f=None):
        '''MPO acting on edge `(i,j,f)`.
        `i,j` are vertex sites, optional `f` is 
        the face site.
        '''
        return self._ham_terms[(i,j,f)]
    

    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for every location
        (i.e. edge) to be acted on with a Ham term
        '''
        for (i,j,f), mpo in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, mpo


    def _make_ham_terms(self, qlattice):
        
        site_tag = self._site_tag_id
        t, V, mu = self.ham_params()

        terms = dict()
        
        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    mpo = sign * t * self._two_site_hop_mpo()
                    mpo += V * self._two_site_nnint_mpo()
                    terms[(i,j,f)] = mpo
                
                #three-site
                else:
                    mpo = sign * t * self._three_site_hop_mpo(edge_dir='vertical')
                    mpo += V * self._three_site_nnint_mpo()
                    terms[(i,j,f)] = mpo

        
        #horizontal edges
        for (i,j,f) in qlattice.get_edges('horizontal'):

            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_mpo()
                    terms[(i,j,f)] += V * self._two_site_nnint_mpo()

            #three-site    
            else:
                terms[(i,j,f)] =  t * self._three_site_hop_mpo(edge_dir='horizontal')
                terms[(i,j,f)] += V * self._three_site_nnint_mpo()

        

        if mu == 0.0:
            return terms


        n_op = number_op() #one-site number operator 
        Ident = qu.eye(2)

        #map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))

        
        mpo_NI = qtn.MatrixProductOperator(arrays=[n_op.reshape(2,2,1),
                                                  Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)

        mpo_IN = qtn.MatrixProductOperator(arrays=[Ident.reshape(2,2,1),
                                                   n_op.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        mpo_NII = qtn.MatrixProductOperator(arrays=[n_op.reshape(2,2,1),
                                                    Ident.reshape(1,2,2,1),
                                                    Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        mpo_INI = qtn.MatrixProductOperator(arrays=[Ident.reshape(2,2,1),
                                                   n_op.reshape(1,2,2,1),
                                                   Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        two_site_mpos = (mpo_NI, mpo_IN)
        three_site_mpos = (mpo_NII, mpo_INI)

        #for each vertex in lattice, absorb occupation term
        #evenly into the edge terms that include it
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            cover_edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(cover_edges)

            assert num_edges>1 or qlattice.num_faces()==0 #should appear in at least two edge terms!


            for (i,j,f) in cover_edges:
                
                #Number op acts on either the i or j vertex (first or second site)
                which_vertex = (i,j,f).index(vertex) #`which` can be 0 or 1


                #no face, so use 2-site MPO
                if f is None: 
                    #choose either NI or IN
                    terms[(i,j,f)] -= mu * (1/num_edges) * two_site_mpos[which_vertex]
                                
                else: #include face, use 3-site MPO
                    #choose either NII or INI
                    terms[(i,j,f)] -= mu * (1/num_edges) * three_site_mpos[which_vertex]

        return terms


    def _two_site_hop_mpo(self):
        ''' 0.5 * (XX + YY)
        '''
        X, Y = (qu.pauli(q) for q in ['x','y'])
        site_tag = self._site_tag_id
    
        mpoXX = qtn.MatrixProductOperator(arrays=[X.reshape(2,2,1), 
                                                  X.reshape(1,2,2)], 
                                        shape='ludr',
                                        site_tag_id=site_tag)
        
        mpoYY = qtn.MatrixProductOperator(arrays=[Y.reshape(2,2,1), 
                                                  Y.reshape(1,2,2)], 
                                        shape='ludr',
                                        site_tag_id=site_tag)
        
        return (mpoXX + mpoYY) / 2


    def _three_site_hop_mpo(self, edge_dir):
        '''(XXO + YYO)/2
        '''
        X, Y = (qu.pauli(q) for q in ['x','y'])
        O = {'vertical': X, 'horizontal':Y} [edge_dir]
        site_tag = self._site_tag_id

        mpo_XXO = qtn.MatrixProductOperator(arrays=[X.reshape(2,2,1), 
                                                    X.reshape(1,2,2,1),
                                                    O.reshape(1,2,2)], 
                                            shape='ludr',
                                            site_tag_id=site_tag)

        mpo_YYO = qtn.MatrixProductOperator(arrays=[Y.reshape(2,2,1), 
                                                    Y.reshape(1,2,2,1),
                                                    O.reshape(1,2,2)], 
                                            shape='ludr',
                                            site_tag_id=site_tag)

        return (mpo_XXO + mpo_YYO) / 2                                                    


    def _two_site_nnint_mpo(self, third_site_identity=False):
        '''Two-site nearest-neighbor interaction, optionally padded
        with an identity to act on a third site.
        '''
        Nop = number_op()
        site_tag = self._site_tag_id

        if third_site_identity:
            oplist = [Nop.reshape(2,2,1), 
                      Nop.reshape(1,2,2,1), 
                      qu.eye(2).reshape(1,2,2)]
        
        else:
            oplist = [Nop.reshape(2,2,1), 
                      Nop.reshape(1,2,2)]


        return qtn.MatrixProductOperator(arrays=oplist,
                                        shape='ludr',
                                        site_tag_id=site_tag)
    

    _three_site_nnint_mpo = functools.partialmethod(_two_site_nnint_mpo,
                                third_site_identity=True)


    

        

################################

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
