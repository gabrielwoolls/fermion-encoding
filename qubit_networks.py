import quimb as qu
import quimb.tensor as qtn
from itertools import product, chain, starmap, cycle, combinations
import dense_qubits
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
from collections import defaultdict
from numbers import Integral
from autoray import do, dag
import tqdm
import functools
from quimb.tensor.tensor_core import tags_to_oset
from quimb.utils import pairwise, check_opt, oset
import opt_einsum as oe
from operator import add
import re
import numpy as np
from random import randint



def make_auxcon_net(
    Lx,
    Ly,
    phys_dim=2,
    bond_dim=3,
    grid_tag_id='S{},{}',
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    aux_tag_id='X{}',
    add_tags=None,
    **tn_opts
):
    # tag_id = site_tag_id
    # ind_id = phys_ind_id
    D = bond_dim
    add_tags = tags_to_oset(add_tags)

    added_tensors = []

    dummy = np.random.rand(D,D,D,D)

    vertex_net = make_product_state_net(
        vertices_only=True, Lx=Lx, Ly=Ly, phys_dim=phys_dim,
        bond_dim=bond_dim, site_tag_id=site_tag_id,
        phys_ind_id=phys_ind_id, add_tags=add_tags,
        **tn_opts)
    
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):

        if i % 2 == j % 2: #add face site & splitting tensors
            
            face_bonds = [phys_ind_id.format(Lx*Ly + k)] + [qtn.rand_uuid()
                                                            for _ in range(2)]
            
            face_tensor = qtn.rand_tensor(
                            shape=[phys_dim, D, D],
                            inds=face_bonds,
                            tags=('FACE', 
                                  grid_tag_id.format(2*i+1, 2*j+1),
                                  site_tag_id.format(Lx*Ly + k), 
                                  *add_tags)
            )
            
            up_left_corner = i * Ly + j
            T1 = vertex_net[up_left_corner]
            T2 = vertex_net[up_left_corner+1]
            
            tensor_upper_split = insert_split_tensor(
                T1=T1, T2=T2, face_ind=face_bonds[1],
                Tf=face_tensor, add_tags=['AUX', 'UPPER',
                aux_tag_id.format(up_left_corner),
                grid_tag_id.format(2*i, 2*j+1), *add_tags]
            )

            down_left_corner = (i + 1) * Ly + j
            T1 = vertex_net[down_left_corner]
            T2 = vertex_net[down_left_corner + 1]

            tensor_lower_split = insert_split_tensor(
                T1=T1, T2=T2, face_ind=face_bonds[2],
                Tf=face_tensor, add_tags=['AUX', 'LOWER', 
                aux_tag_id.format(down_left_corner), 
                grid_tag_id.format(2*i + 2, 2*j+1), *add_tags]
            )
            

            added_tensors.append([face_tensor, 
                                tensor_upper_split,
                                tensor_lower_split])
            k+=1
    
    vertex_net |= added_tensors
    # return qtn.TensorNetwork(vertex_net.tensors, structure=site_tag_id)
    return vertex_net




    
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
                      tags=add_tags,)


# TODO: implement bitstring functionality
def make_product_state_net(
    Lx, 
    Ly,
    phys_dim, 
    bond_dim=3,
    bitstring=None,
    grid_tag_id='S{},{}',
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    add_tags=None,
    dtype='complex128',
    vertices_only=False,
    **tn_opts
):
    '''Makes a product state qubit network, for a lattice with 
    dimensions ``Lx, Ly`` and local site dimension ``phys_dim``.    

    Currently, every site is initialized to the `up x up x ...`
    state, i.e. `basis_vec(0)`
    
    Every qubit tensor is tagged with:
        1. 'VERT' or 'FACE' according to the site
        2. A 'supergrid' tag (e.g. "S{x}{y}" for supercoo (x, y)
        3. A qubit-number tag (e.g. "Q{k}" for the kth qubit)
        4. Any additional supplied in ``add_tags``
    '''
    # note this only works if either Lx or Ly is odd!
    num_vertices = Lx * Ly
    num_faces = int((Lx-1) * (Ly-1) / 2)

    # optional additional tags, none by default
    add_tags = tags_to_oset(add_tags) 
    
    spin_map = {'0': qu.basis_vec(i=0, dim=phys_dim, dtype=dtype).reshape(phys_dim),
                '1': qu.basis_vec(i=1, dim=phys_dim, dtype=dtype).reshape(phys_dim)}

    #default to spin-up at every site
    if bitstring is None:
        spin_up = spin_map['0']
        vertex_data = [spin_up] * num_vertices
        face_data = [spin_up] * num_faces


    elif all(s in '01' for s in bitstring) and len(bitstring) == num_vertices + num_faces:
        vertex_data, face_data = [], []

        for k, bit in enumerate(bitstring):

            if k < num_vertices:
                vertex_data.append(spin_map[bit])
            
            if k >= num_vertices:
                face_data.append(spin_map[bit])
                
    else:
        raise ValueError(f"{bitstring} is not a proper spin configuration")


    vtensors = [[None for _ in range(Ly)] for _ in range(Lx)]

    for i, j in product(range(Lx), range(Ly)):
        
        qubit_number = i * Ly + j
        supergrid_coo = (2*i, 2*j)

        ind_ij = (phys_ind_id.format(qubit_number),)

        tags_ij = ('VERT', 'QUBIT',
                   site_tag_id.format(qubit_number),
                   grid_tag_id.format(*supergrid_coo),
                   *add_tags)

        vertex_ij = qtn.Tensor(data = vertex_data[qubit_number], 
                               inds = ind_ij,
                               tags = tags_ij)
        
        vtensors[i][j] = vertex_ij

    
    for i, j in product(range(Lx), range(Ly)):
     
        if i <= Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j], size=bond_dim)
        
        if j <= Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1], size=bond_dim)


    if vertices_only:
        vtensors = list(chain.from_iterable(vtensors))
        return qtn.TensorNetwork(vtensors, structure=site_tag_id, **tn_opts)

    ##
    # add face sites + bonds 
    ##
    
    ftensors = [[None for _ in range(Ly-1)] for _ in range(Lx-1)]
    
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):
        
        if i % 2 == j % 2: #tensors on the 'even' faces
            qubit_number = k + Lx * Ly
            supergrid_coo = (2*i + 1, 2*j + 1)

            ind_ij = (phys_ind_id.format(qubit_number),)

            tags_ij = ('FACE', 'QUBIT',
                        site_tag_id.format(qubit_number),
                        grid_tag_id.format(*supergrid_coo),
                        *add_tags)

            face_ij = qtn.Tensor(data = face_data[k], #note index k, not k + Lx.Ly
                                 inds = ind_ij,
                                 tags = tags_ij)

            ftensors[i][j] = face_ij
            k += 1


    for i, j in product(range(Lx-1), range(Ly-1)):
        
        if ftensors[i][j] is not None:
           
            ftensors[i][j].new_bond(vtensors[i][j], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i][j+1], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i+1][j+1], size=bond_dim)
            ftensors[i][j].new_bond(vtensors[i+1][j], size=bond_dim)


    vtensors = list(chain.from_iterable(vtensors))
    ftensors = list(chain.from_iterable(ftensors))
    
    alltensors = vtensors + [f for f in ftensors if f]
    # return alltensors
    return qtn.TensorNetwork(alltensors, structure=site_tag_id, **tn_opts)



def local_product_state_net(
    Lx, 
    Ly,
    phys_dim, 
    bond_dim=3,
    grid_tag_id='S{},{}',
    site_tag_id='Q{}',
    phys_ind_id='q{}',
    add_tags=None,
    dtype='complex128',
    **tn_opts):
    '''Makes a "local product state" qubit network, for a lattice with 
    dimensions ``Lx, Ly`` and local site dimension ``phys_dim``.    

    For now, makes each "square" of the form
    
    ●─────●        
    │  ●  │  
    ●─────● 
    
    start in a random state, and "products" all the squares together.
    Entanglement within the squares, but each square unentangled with 
    the other squares.
    '''


    add_tags = tags_to_oset(add_tags) #none by default
    spin_up = qu.basis_vec(i=0, dim=phys_dim).reshape(phys_dim)

    vtensors = [[None for _ in range(Ly)] for _ in range(Lx)]

    ## make vertex sites ##
    for i, j in product(range(Lx), range(Ly)):

        ind_ij = (phys_ind_id.format(i * Ly + j),)

        tags_ij = ('VERT', 'QUBIT',
                   site_tag_id.format(i * Ly + j),
                   grid_tag_id.format(2*i, 2*j),
                   *add_tags)

        vertex_ij = qtn.Tensor(data = spin_up, 
                               inds = ind_ij,
                               tags = tags_ij)
        
        vtensors[i][j] = vertex_ij

    ftensors = [[None for _ in range(Ly-1)] for _ in range(Lx-1)]
    
    ## make all face sites
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):

        if i % 2 == j % 2: #put tensors on 'even' faces
            ind_ij = (phys_ind_id.format(k + Lx * Ly),)
            tags_ij = ('FACE', 'QUBIT',
                        site_tag_id.format(k + Lx * Ly),
                        grid_tag_id.format(2*i + 1, 2*j + 1),
                        *add_tags)
            face_ij = qtn.Tensor(data = spin_up,
                                 inds = ind_ij,
                                 tags = tags_ij)

            ftensors[i][j] = face_ij
            k += 1

    ## connect faces to vertices
    #NOTE: is there a reason not to do this in previous loop?
    for i, j in product(range(Lx-1), range(Ly-1)):
        
        if i % 2 == j % 2:
           
            #connect face tensor to four corner vertices
            face_tensor = ftensors[i][j]
            vertex_A = vtensors[i][j]
            vertex_B = vtensors[i][j+1]
            vertex_C = vtensors[i+1][j+1]           
            vertex_D = vtensors[i+1][j]

            
            face_tensor.new_bond(vertex_A, size=bond_dim)
            face_tensor.new_bond(vertex_B, size=bond_dim)
            face_tensor.new_bond(vertex_C, size=bond_dim)
            face_tensor.new_bond(vertex_D, size=bond_dim)
            
            vertex_A.new_bond(vertex_D, size=bond_dim)
            vertex_A.new_bond(vertex_B, size=bond_dim)

            vertex_C.new_bond(vertex_D, size=bond_dim)
            vertex_C.new_bond(vertex_B, size=bond_dim)
    
    for vt in list(chain.from_iterable(vtensors)):
        vt.randomize(dtype=dtype, inplace=True)
    for ft in list(chain.from_iterable(ftensors)):
        if ft: ft.randomize(dtype=dtype, inplace=True) #excludes `None` tensors (empty faces)
    
    ## add zero-entanglement (dim=1) bonds
    for i,j in product(range(Lx), range(Ly)):
     
        if i <= Lx-2 and 0 == len(qtn.bonds(vtensors[i][j], vtensors[i+1][j])):
            vtensors[i][j].new_bond(vtensors[i+1][j], size=2)
        
        if j <= Ly-2 and 0 == len(qtn.bonds(vtensors[i][j], vtensors[i][j+1])):
            vtensors[i][j].new_bond(vtensors[i][j+1], size=2)

    vtensors = list(chain.from_iterable(vtensors))
    ftensors = list(chain.from_iterable(ftensors))
    
    alltensors = vtensors + [f for f in ftensors if f]
    return qtn.TensorNetwork(alltensors, structure=site_tag_id, **tn_opts)





    
        
            



def make_random_net(Lx, Ly, phys_dim, 
                    bond_dim=3, 
                    grid_tag_id='S{},{}',
                    site_tag_id='Q{}', 
                    phys_ind_id='q{}',
                    add_tags=None,
                    dtype='complex128',
                    **tn_opts
                    ):
    '''
    NOTE: can make much simpler with ``Tensor.randomize``?

    Return a `TensorNetwork` made from random tensors
    structured like `qlattice` i.e. with the same local 
    qudit degrees of freedom. 
    
    Each site has physical index dimension `d = qlattice._local_dim`,
    and is connected to its neighbors with a virtual bond of 
    dimension ``bond_dim``.

    Vertex tensors are tagged with 'VERT' and face tensors with 'FACE'.
    In addition, every tensor is tagged with those supplied in ``add_tags``
    '''

    #dummy TN, site tensors to be replaced with randomized
    tnet = make_product_state_net(Lx=Lx, Ly=Ly, phys_dim=phys_dim,
                            bond_dim=bond_dim, 
                            grid_tag_id=grid_tag_id,
                            site_tag_id=site_tag_id, 
                            phys_ind_id=phys_ind_id,
                            add_tags=add_tags,
                            **tn_opts)

    #replace vertex tensors with randoms
    for i, j in product(range(Lx), range(Ly)):
        
        tid = tuple(tnet.tag_map[site_tag_id.format(i*Ly+j)])
        tid = tid[0]

        old_tensor = tnet._pop_tensor(tid)

        shape = old_tensor.shape
        tags = old_tensor.tags #[f'Q{i*Ly + j}', 'VERT']
        inds  = old_tensor.inds #[..., f'q{i*Ly + j}']

                
        rand_data = qtn.array_ops.sensibly_scale(
                            qtn.array_ops.sensibly_scale(
                            qu.gen.rand.randn(shape, dtype)))
        
        tensor_ij = qtn.Tensor(rand_data, inds, tags)
        tnet |= tensor_ij
    

    #replace face tensors with randoms
    k=0
    for i, j in product(range(Lx-1), range(Ly-1)):
        #replace face tensors
        if i%2 == j%2:
            
            tid = tuple(tnet.tag_map[site_tag_id.format(k+Lx*Ly)])
            tid = tid[0]

            old_tensor = tnet._pop_tensor(tid)

            shape = old_tensor.shape
            tags = old_tensor.tags 
            inds  = old_tensor.inds

                    
            rand_data = qtn.array_ops.sensibly_scale(
                                qtn.array_ops.sensibly_scale(
                                qu.gen.rand.randn(shape, dtype)))
            
            tensor_ij = qtn.Tensor(rand_data, inds, tags)
            tnet |= tensor_ij

    
    return tnet





class iTimeTEBD:
    '''Object for TEBD imaginary-time evolution.

    Params:
    ------
    `qnetwork`: QubitEncodeNet
        The initial state of the qubits. Will be modified
        in-place.
    
    `ham`: MasterHam, or SimulatorHam, or StabHam
        Hamiltonian for time evolution. Should have a
        `gen_trotter_gates(tau)` method.
    
    `compute_extra_fns`: callable or dict of callables, optional
        If desired, can give extra callables to compute at each
        step of TEBD. Each function should take only the current 
        state `qnet` as parameter. Results of the callables will 
        be stored in `self._results`
    
    `contract_opts`: 
        Supplied to 
        :meth:`~dense_qubits.QubitLattice.apply_trotter_gates_`
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
        
        else:
            #fns is dict of callables
            if isinstance(fns, dict):
                self._results = {k: [] for k in fns}

                #given state at time t, compute observables
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


    def get_normalized_data(self, data):
        '''Convenience method for testing.
        '''

        if data == 'Esim':
            return np.divide(np.real(self.results('sim')),
                            np.real(self.results('norm')))
        
        elif data == 'Estab':
            return np.divide(np.real(self.results('stab')),
                            np.real(self.results('norm')))


def compute_encnet_ham_expec(qnet, ham):
    '''Useful callable for TEBD
    '''
    return qnet.compute_ham_expec(ham, normalize=False)


def compute_encnet_normsquared(qnet):
    return np.real(qnet.make_norm()^all)



## ******************************* ##



class QubitEncodeNet(qtn.TensorNetwork):
    '''
        Tensors at physical sites are tagged with 'QUBIT' while
        auxiliary tensors (e.g. dummy identities) are tagged with
        'AUX' tag.
        

        Attributes:
        ----------

        `qlattice`: QubitLattice
            Specifies underlying lattice geometry, in 
            particular the shape (Lx, Ly) of the qu(d)it
            lattice and the local dimension of the physical 
            sites, e.g. d=2 (4) for simulating spinless 
            (spin-1/2) fermions.


        _vertex_coo_map: dict[tuple(int) : int]
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
        

        _supergrid: list[ list[ str or None ]], shape (2 * Lx - 1, 2 * Ly - 1)
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
        '_Lx',
        '_Ly',
        '_phys_dim',
        '_grid_tag_id',
        '_site_tag_id',
        '_aux_tag_id',
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
                        like=self) #,
                        # qlattice=self.qlattice,
                        # site_tag_id=self.site_tag_id)
        return new
    

    def __or__(self, other):
        new = super().__or__(other)
        if self._is_compatible_lattice(other):
            new.view_as_(QubitEncodeNet,
                        like=self)#,
                        # qlattice=self.qlattice,
                        # site_tag_id=self.site_tag_id)
        return new


    @classmethod
    def random_flat(cls, Lx, Ly, bond_dim=3, **tn_opts):

        rand_net = QubitEncodeVector.rand(Lx=Lx, Ly=Ly, phys_dim=1,
                    bond_dim=bond_dim, **tn_opts)

        rand_net.squeeze_()
        
        return rand_net.view_as(cls, inplace=True)
 

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly

    @property
    def phys_dim(self):
        '''Local physical dimension of qu(d)it sites.
        '''        
        return self._phys_dim

    @property
    def grid_Lx(self):
        '''Total num rows in the ``supergrid``, i.e.
        vertex rows + face rows.
        '''
        return 2 * self.Lx - 1

    @property
    def grid_Ly(self):
        '''Number of columns in ``supergrid``, i.e. vertex 
        cols + face cols
        '''
        return 2 * self.Ly - 1


    def qlattice(self):
        '''Internal ``QubitLattice`` object
        '''
        if not hasattr(self, '_qlattice'):
            self._qlattice = dense_qubits.QubitLattice(
                            Lx=self.Lx,
                            Ly=self.Ly,
                            local_dim=self.phys_dim)
        return self._qlattice

    @property
    def site_tag_id(self):
        '''Format string for the tag identifiers of local sites,
        i.e. the qubit numbers like 'Q{k}'
        '''
        return self._site_tag_id
    
    @property
    def aux_tag_id(self):
        '''Format string for the tags of 'auxiliary' tensors
        (e.g. dummy identities) anywhere in supergrid.

        >>> 'IX{}Y{}'
        '''
        return self._aux_tag_id

    @property
    def grid_tag_id(self):
        '''Format string for tag at a given supercoo, like 'S{x}{y}'
        '''
        return self._grid_tag_id

    @property
    def row_tag_id(self):
        return "ROW{}"
    
    @property
    def col_tag_id(self):
        return "COL{}"
    
    
    
    # def copy(self, virtual=False, deep=False):
    #     return self.__class__(
    #                     self,
    #                     virtual=virtual)
    #                     # qlattice=self.qlattice,
    #                     # site_tag_id=self.site_tag_id)

    # __copy__ = copy


    # def _site_tid(self, k):
    #     '''Given the site index `k` in {0,...,N},
    #     return the `tid` of the local tensor
    #     '''
    #     #'q{k}'
    #     index = self._phys_ind_id.format(k)
    #     return self.ind_map[index]    


    def calc_supergrid(self):
        '''Infer the 'supergrid' of this tensor network by looking
        for tags like 'S{x}{y}' and storing tags in a 2D array.
        Leaves ``None`` at any grid coo without matching tensors.
        '''
        
        supergrid = [[None for _ in range(2 * self.Ly - 1)] 
                           for _ in range(2 * self.Lx - 1)]
        
        grid_tags = {(x, y): self.grid_coo_tag(x,y) for x, y in
                product(range(2 * self.Lx - 1), range(2 * self.Ly - 1))}

        for (x, y), tag_xy in grid_tags.items():
            #at most one tensor should match 'S{x}{y}' tag
            if tag_xy in self.tags:
                supergrid[x][y] = tags_to_oset(tag_xy)

        return supergrid


    def show_supergrid(self):
        '''Print out supergrid for debugging
        '''
        for x in range(self.grid_Lx):
            
            for y in range(self.grid_Ly):
                tags = self.supergrid(x,y)
                
                if tags is None:
                    tags = '....'
                
                else:
                     tags = tags[0]

                print(f'  {tags}  ', end='')
            
            print('\n')

    
    # def _update_supergrid(self, x, y, tag):
    #     '''Add `tag` to the set of tags at supergrid 
    #     coordinate `x, y`. Does not complain if the
    #     tag was already there.
    #     '''
    #     if not hasattr(self, '_supergrid'):
    #         self._supergrid = self.calc_supergrid()
        
    #     tag = tags_to_oset(tag)

    #     if self._supergrid[x][y] is None:
    #         self._supergrid[x][y] = tag
        
    #     else:
    #         self._supergrid[x][y].update(tag)
    

    # def _move_supergrid_tags(self, from_coo, to_coo):
    #     '''Add the tags at `from_coo` in the supergrid
    #     to those at `to_coo`. Keeps the previous tags 
    #     at `to`, and removes the tags at `from`.
    #     '''
    #     if not hasattr(self, '_supergrid'):
    #         self._supergrid = self.calc_supergrid()

    #     x, y = from_coo
    #     x2, y2 = to_coo
        
    #     from_tag = self._supergrid[x][y]

    #     if from_tag is None:
    #         return
        
    #     elif self._supergrid[x2][y2] is None:
    #         self._supergrid[x2][y2] = from_tag
    #         self._supergrid[x][y] = None
        
    #     else:
    #         self._supergrid[x2][y2] |= from_tag
    #         self._supergrid[x][y] = None




    def vertex_coo_map(self, i=None, j=None):
        '''Maps location (i,j) in vertex lattice
        to the corresponding site number,  e.g.
        
        >>> vertex_coo_map(0, 1)
        >>> 1
        '''
        if not hasattr(self, '_vertex_coo_map'):
            self._vertex_coo_map = {(2*i, 2*j): i * self.Ly + j
                    for i, j in product(range(self.Lx), range(self.Ly))}
            
        if (i is None) and (j is None):
            return self._vertex_coo_map        

        elif (i % 2 == 0) and (j % 2 == 0):
            return self._vertex_coo_map[(i, j)]        
        
        else:
            raise ValueError(f"{i},{j} not a proper vertex coordinate")



    def is_vertex_coo(self, xy):
        x, y = xy

        return all((
            x % 2 == 0,
            y % 2 == 0,
            self.valid_supercoo(xy)
        ))
    
    def is_face_coo(self, xy):
        x, y = xy

        return all((
            x % 2 == 1,
            y % 2 == 1,
            self.valid_supercoo(xy)
        ))
    
    def is_qubit_coo(self, xy):
        '''Whether `xy` is a vertex or face coo (not auxiliary)
        Note: will count empty face sites as ``True``
        '''
        x, y = xy
        return (x % 2 == y % 2) and self.valid_supercoo(xy)

    def face_coo_map(self, x=None, y=None):
        '''Maps location (x, y) in grid to the 
        corresponding face-qubit 'number', or ``None`` 
        if xy is an empty face.
        '''
        if not hasattr(self, '_face_coo_map'):

            Lx, Ly = self.Lx, self.Ly
            n_vertices = Lx * Ly

            #empty faces first
            self._face_coo_map = {(2*i + 1, 2*j + 1): None for i, j in product(range(Lx-1), range(Ly-1))
                              if i % 2 != j % 2}

            # add face qubits
            face_sites = [(2*i + 1, 2*j + 1) for i,j in product(range(Lx-1), range(Ly-1)) 
                          if  i % 2 == j % 2]
            
            for num, coo in enumerate(face_sites):
                self._face_coo_map.update({coo: num + n_vertices})


        if (x is None) and (y is None):
            return self._face_coo_map
        

        elif (x % 2 == 1) and (y % 2 == 1):
            return self._face_coo_map[(x,y)]
        
        else:
            raise ValueError(f"{x},{y} not a proper face coordinate")


    def qubit_to_coo_map(self, qnumber=None):
        """Mapping of qubit numbers to corresponding grid coordinates, e.g.

        >>> qubit_to_coo_map(0)
        (0, 0)

        >>> qubit_to_coo_map(1)
        (0, 1)

        or, if no qubit is specified, return the whole dict:

        >>> qubit_to_coo_map()
        {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), ...}
        """

        if not hasattr(self, '_qubit_to_coo_map'):
            q2coo = {q: coo for coo, q in self.vertex_coo_map().items()}
            q2coo.update({q: coo for coo, q in self.face_coo_map().items()
                                    if q is not None})
            self._qubit_to_coo_map = q2coo


        if qnumber is not None:
            return self._qubit_to_coo_map[qnumber]


        return self._qubit_to_coo_map


    def set_supergrid(self, arrays):
        self._supergrid = arrays


    def supergrid(self, x=None, y=None, layer=None):
        '''Returns *tuple* of tags corresponding
        to supercoo ``x, y``. If no coo is specified
        (default) returns the whole dictionary.
        '''
        if not hasattr(self, '_supergrid'):
            # automatically infer supergrid structure
            self._supergrid = self.calc_supergrid()
        

        if (x is not None) and (y is not None):
            tags = self._supergrid[x][y]
            
            if tags is None:
                return None
            
            else:
                return tuple(tags)
            
        
        return self._supergrid




    def vert_coo_tag(self, x, y):
        '''Tag 'Q{k}' for vertex site at supercoo (x, y)
        '''
        qnumber = self.vertex_coo_map(x,y)

        return self.site_tag_id.format(qnumber)
    

    def face_coo_tag(self, x, y):
        '''Tag 'Q{k}' for site at face-coo (x, y), or 
        `None` if the face is empty
        '''
        qnumber = self.face_coo_map(x, y)

        if qnumber is None:
            return None

        return self.site_tag_id.format(qnumber)

    
    def aux_tensor_tag(self, x, y):
        '''Tag for 'auxiliary' tensor at supergrid[x][y]
        '''
        return self.aux_tag_id.format(x,y)


    def grid_coo_tag(self, x, y):
        '''Default tag for tensor at supergrid (x, y),
        e.g. S{x},{y}
        '''
        return self.grid_tag_id.format(x, y)


    def row_tag(self, x):
        '''ROW{x}
        '''
        return self.row_tag_id.format(x)
    

    def col_tag(self, y):
        '''COL{y}
        '''
        return self.col_tag_id.format(y)

    def qubit_tag(self, k):
        '''Q{k}
        '''
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

        

    def gen_vertex_coos(self):
        '''Generate supergrid coordinates for all 
        vertex sites in the lattice.
        '''
        for i,j in product(range(self.Lx), range(self.Ly)):
            yield (2*i, 2*j)

        
    
    def gen_face_coos(self, including_empty=False):
        '''Generate supergrid coordinates of 'face' sites,
        optionally including the empty faces as well.
        '''

        for i, j in product(range(self.Lx-1), range(self.Ly-1)):
        
            if (i % 2 == j % 2) or including_empty:
        
                yield (2 * i + 1, 2 * j + 1)
    

    def gen_supergrid_coos(self):
        '''All the coos in the (2*Lx-1, 2*Ly-1) supergrid, 
        including 'unoccupied' coos.
        '''
        return product(range(2 * self.Lx - 1), range(2 * self.Ly -1))


    def gen_occupied_grid_tags(self, with_coo=True):
        '''Generate the supergrid tags corresponding to locations
        occupied by tensors. If ``with_coo == True``, also yield
        the coordinate (x,y) along with the tag.
        '''
        coo2tag = {(x, y): self.grid_coo_tag(x, y) 
            for x,y in self.gen_supergrid_coos()\
            if self.grid_coo_tag(x,y) in self.tags}
        
        
        return coo2tag.items() if with_coo else coo2tag.values()

        


        # for (x, y), tag_xy in coo2tag.items():
        #     # if there is a tensor matching this grid location
        #     if self.check_for_matching_tags(tag_xy):

        #         yield ((x, y), tag_xy) if with_coo else tag_xy
            
                    

    def valid_supercoo(self, xy):
        '''Whether ``xy`` is inside the supergrid boundary
        '''
        x, y = xy
        return (0 <= x < self.grid_Lx) and (0 <= y < self.grid_Ly)


    def check_for_matching_tags(self, tags):
        '''Whether this TN contains a tensor having all of
        the specified ``tags``. Converts tags to oset first.
        '''
        tags = tags_to_oset(tags)

        for t in tags:
            if t not in self.tag_map:
                return False
        
        return bool(self._get_tids_from_tags(tags=tags, which='all'))


    def _canonize_supergrid_row(self, x, sweep, yrange=None, **canonize_opts):
        '''Canonize all bonds between tensors in the xth row
        of the supergrid.

        Automatically ignores the 'empty' coos in supergrid,
        i.e. picks only the nodes occupied by tensors.
        '''
        check_opt('sweep',sweep,('right','left'))
        
        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)

        ordered_row_tags = self.supergrid_row_slice(x, yrange, sweep=sweep)

        for tag1, tag2 in pairwise(ordered_row_tags):
            self.canonize_between(tag1, tag2, **canonize_opts)



    def _canonize_supergrid_column(self, y, sweep, xrange=None, **canonize_opts):
        check_opt('sweep',sweep,('up','down'))

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)
        
        ordered_col_tags = self.supergrid_column_slice(y, xrange, sweep=sweep)

        for tag1, tag2 in pairwise(ordered_col_tags):
            self.canonize_between(tag1, tag2, **canonize_opts)

    

    def _canonize_supergrid_row_around(self, x, around=(0,1)):
        #sweep to the right
        self._canonize_supergrid_row(x, sweep='right', yrange=(0, min(around)))
        #sweep to the left
        self._canonize_supergrid_row(x, sweep='left', yrange=(max(around), 2*self.Ly-2))
    
    


    def _compress_supergrid_row(
        self, 
        x, 
        sweep, 
        yrange=None, 
        split_method='svd', 
        **compress_opts):
        
        check_opt('sweep', sweep, ('right', 'left'))
        compress_opts.setdefault('absorb', 'right')
        compress_opts.setdefault('method', split_method)

        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)
        
        ordered_row_tags = self.supergrid_row_slice(x, yrange, sweep=sweep) 

        for tag1, tag2 in pairwise(ordered_row_tags):
            self.compress_between(tag1, tag2, **compress_opts)
            

    def _compress_supergrid_column(
        self, 
        y, 
        sweep, 
        xrange=None, 
        split_method='svd', 
        **compress_opts):

        check_opt('sweep', sweep, ('up', 'down'))
        compress_opts.setdefault('absorb', 'right')
        compress_opts.setdefault('method', split_method)

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        ordered_column_tags = self.supergrid_column_slice(y, xrange, sweep=sweep)

        
        for tag1, tag2 in pairwise(ordered_column_tags):
            self.compress_between(tag1, tag2, **compress_opts)


    def comp(self, t1, t2, **compress_opts):
        self.compress_between(t1, t2, **compress_opts)


    def supergrid_row_slice(
        self, 
        x, 
        yrange, 
        layer_tag=None,
        sweep='right', 
        get='tag'
    ):
        '''Directed slice of the grid, where `yrange` is INCLUSIVE. 
        Ignores the empty sites in the lattice.

        Params:
        ------
        x: int
            The supergrid row, from {0, 1, ..., 2 * Lx-2}
        
        yrange: tuple(int, int)
            Inclusive range of columns to select, max yrange
            is (0, 2*Ly-2)
        
        layer_tag: str, optional
            Can specify a layer to look in, like 'KET'  
        
        sweep: {'left', 'right'}
            Direction of the slice.
        
        get: {'tag', 'col'}
            What to generate, either tags (str) or
            column numbers (int)
        
        Returns:
        -------
        Generator of non-None tags/supercolumn labels
        '''
        if layer_tag is None:
            coo_tags = self.grid_coo_tag

        else: #include layer_tag in every set of tags
            coo_tags = lambda x,y: (layer_tag, self.grid_coo_tag(x,y))


        ys = {'right': range(min(yrange), max(yrange)+1),
             'left': range(max(yrange), min(yrange)-1, -1)}[sweep]


        for y in ys:
            # look for a tensor with these tags
            tag_xy = coo_tags(x, y)

            if self.check_for_matching_tags(tag_xy):
                yield {'tag': tag_xy, 'col': y}[get]


    def supergrid_column_slice(
        self, 
        y, 
        xrange, 
        layer_tag=None, 
        sweep='down', 
        get='tag'
    ):
        '''Generate a directed slice of the supergrid, 
        where ``xrange`` is INCLUSIVE and 'empty' sites are omitted.

        y: int
            The supergrid column
        
        xrange: tuple(int, int)
            Inclusive range of rows to select. Maximum
            xrange is (0, 2 * self.Lx - 2)
        
        layer_tag: str, optional
            Can specify a layer to look in, like 'KET'  

        sweep: {'up', 'down'}
            Direction of the slice.
        
        get: {'tag', 'row'}
            Whether to generate strings (tags) or 
            integers (row numbers).
        '''
        if layer_tag is None:
            coo_tags = self.grid_coo_tag
        else:
            coo_tags = lambda x,y: (layer_tag, self.grid_coo_tag(x,y))
        

        xs = {'down': range(min(xrange), max(xrange) + 1),
              'up': range(max(xrange), min(xrange) - 1, -1)}[sweep]

        for x in xs:
            tag_xy = coo_tags(x, y)
            # look for a tensor with these tags
            if self.check_for_matching_tags(tag_xy):
                yield {'tag': tag_xy, 'row': x}[get]
        



    def setup_bmps_contraction(self, layer_tags=None, inplace=False):
        '''Prepare for boundary-MPS-style contraction by 'rotating'
        face qubits, inserting identities to make this TN look
        more 'PEPS-like'.
        '''
        tn = self if inplace else self.copy()
        
        #first check if there are 'AUX'-tagged tensors already
        
        maybe_already_rotated = self.check_if_bmps_setup()
        
        if maybe_already_rotated:
            print("'setup_bmps' exited, face qubits already rotated!")
            return tn

        # otherwise proceed with setup

        if layer_tags is None:
            tn.rotate_face_qubits_()
            tn.fill_cols_with_identities_()
            tn.fill_rows_with_identities_()
        
        else:
            for layer in layer_tags:
                tn.rotate_face_qubits_(layer_tag=layer)
                tn.fill_cols_with_identities_(layer_tag=layer)
                tn.fill_rows_with_identities_(layer_tag=layer)

        tn._supergrid = tn.calc_supergrid()
        return tn
    
    setup_bmps_contraction_ = functools.partialmethod(
                                    setup_bmps_contraction,
                                    inplace=True)


    def check_if_bmps_setup(self):
        '''Checks whether 'AUX' is in the TN tags, as
        indication of whether face qubits are rotated.
        '''
        return ('AUX' in self.tags)


    def _contract_boundary_from_left_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='up',
        layer_tag=None,
        retag_boundary=True,
        **compress_opts
    ):
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
        }[compress_sweep]

        
        def contract_any_layers(grid_tag1, grid_tag2):

            ''' Contract any tensors with these tags, i.e. 
            on any layer (bra or ket)'''
            self.contract_((grid_tag1, grid_tag2,), which='any')

        def contract_chosen_layer(grid_tag1, grid_tag2):
            ''' Only contract tensors living on the specified
            layer via `layer_tag`.

            Looks for `layer_tag` on the *second* tensor given.
            '''
            self.contract_between(tags1=grid_tag1,
                                  tags2=(grid_tag2, layer_tag))
                                

        
        # maps x, y to grid tag like 'S{x},{y}'
        grid_tag = self.grid_coo_tag
        
        if layer_tag is None:
            contract_step = contract_any_layers
            maybe_with_layer = lambda t: t
        
        else:
            contract_step = contract_chosen_layer
            maybe_with_layer = lambda t: (t, layer_tag)


        for y in range(min(yrange), max(yrange)):
            #
            #     ●──●──       ●──
            #     │  │         ║
            #     ●  ●──  ==>  ●──
            #     │  │         ║
            #     ●──●──       ●──
            #
            retag_map = dict()

            
            slice_opts = dict(xrange=xrange, get='row', layer_tag=layer_tag)
            # get lists of 'nonempty' row indices in each column
            left_col = tuple(self.supergrid_column_slice(y, **slice_opts))
            right_col = tuple(self.supergrid_column_slice(y + 1, **slice_opts))
            
            # do a 'prescan' to check where there are tensors and bonds
            scan_for_tensors, scan_for_bonds = dict(), dict()

            for x in range(min(xrange), max(xrange)+1):
                
                scan_for_tensors[x] = (x in left_col, x in right_col)

                # store if there is a bond 'floating' over the supergrid locations
                scan_for_bonds[x] = (self.check_for_vertical_bond_across(x, y, layer_tag),
                              self.check_for_vertical_bond_across(x, y+1, layer_tag))
            
            
            for x in range(min(xrange), max(xrange)+1):
                
                # whether there are tensors at (left, right) sites
                tensors_here = scan_for_tensors[x]
                
                # whether there are bonds floating over (left, right) sites
                bonds_here = scan_for_bonds[x] 

                if tensors_here == (True, True):
                    # CASE 1
                    # 
                    #     │  │           ║
                    #     ●──●──   ==>   ●──
                    #     │  │           ║
                    # 
                    # tensors on both sides, contract bond

                    left_tag, right_tag = grid_tag(x, y), grid_tag(x, y + 1)
                    
                    # looking for layer_tag on right-hand tensor
                    contract_step(left_tag, right_tag)                    
                    
                    # drop right-hand tag for now
                    # new_tensor = self[maybe_with_layer(left_tag)]
                    # new_tensor.drop_tags(right_tag)

                    # for retagging later
                    retag_map[left_tag] = right_tag


                elif tensors_here == (True, False):
                    
                    if bonds_here[1] == True:
                        # CASE 2
                        # 
                        #     ●──●──     ●──●──      ●──
                        #     │  │       │  │        ║
                        #     ●  │  ==>  ●--i  ==>   ●  
                        #     │  │       │  │        ║
                        #     ●──●──     ●──●──      ●──
                        # 
                        # no right-tensor, but need to absorb a bond
                        # Insert a dummy identity and contract w/ left-tensor
                        
                        dummy_tags = ('AUX', grid_tag(x, y+1))
                        
                        if layer_tags is not None:
                            dummy_tags = (*dummy_tags, layer_tag)

                        self.insert_identity_between_(where1=grid_tag(x-1, y+1), 
                                                      where2=grid_tag(x+1, y+1),
                                                      layer_tag=layer_tag,
                                                      add_tags=dummy_tags)

                        contract_step(dummy_tags, grid_tag(x, y))

                        retag_map[left_tag] = right_tag

                    
                    else: 
                        # CASE 3
                        # 
                        #     ●──●──       ●──
                        #     │            │
                        #     ●      ==>   ●  
                        #     │            │
                        #     ●──●──       ●──
                        # 
                        # no bond to absorb, just shift left-tensor
                        # rightward so we have it on the 'next' column
                        left_tag, right_tag = grid_tag(x, y), grid_tag(x, y + 1)
                        self[left_tag].add_tag(right_tag)
                        retag_map[left_tag] = right_tag

                
                elif tensors_here == (False, True):
                    
                    if bonds_here[0] == True:
                        # CASE 4
                        # 
                        #     ●──●──        ●──●──         ●──
                        #     │  │          │  │           ║
                        #     |  ●──   ==>  i--●──   ==>   ●──
                        #     │  │          │  │           ║
                        #     ●──●──        ●──●──         ●──     
                        # 
                        # No left-tensor, but there is a bond to absorb rightwards
                        # Insert a dummy identity and contract to the right
                        
                        dummy_tags = ('AUX', grid_tag(x, y))
                        
                        self.insert_identity_between_(where1=grid_tag(x - 1, y), 
                                                      where2=grid_tag(x + 1, y), 
                                                      add_tags=dummy_tags)

                        # look for ``layer_tag`` on right-tensor
                        contract_step(dummy_tags, grid_tag(x, y+1))
                        retag_map[grid_tag(x, y)] = grid_tag(x, y+1)
                        


                    else:
                        # CASE 5
                        # 
                        #     ●──●──         ●──
                        #        │           │
                        #        ●──   ==>   ●──
                        #        │           │
                        #     ●──●──         ●──     
                        #
                        # no bond or tensor to push rightward
                        continue

                
                elif tensors_here == (False, False): 
                    # CASE 6
                    # no tensor on either side
                    continue
                

                else:
                    raise ValueError('Missed a case!')

            
            #end loop over rows

            self.fuse_multibonds_()


            if canonize:
                #
                #     ●──       v──
                #     ║         ║
                #     ●──  ==>  v──
                #     ║         ║
                #     ●──       ●──
                #
                self._canonize_supergrid_column(y, canonize_sweep, xrange)
            #
            #     v──       ●──
            #     ║         │
            #     v──  ==>  ^──
            #     ║         │
            #     ●──       ^──
            #
            self._compress_supergrid_column(y, compress_sweep, 
                                    xrange, **compress_opts)

            # optionally, drop 'previous' boundary tags (i.e. column y)
            # to effectively move the boundary rightward (y --> y+1)
            
            # TODO: CAN GET RID OF RETAG_MAP ?
            if retag_boundary:
                # for x in range(min(xrange), max(xrange)+1):
                #     self.drop_tags(self.grid_coo_tag(x, y))
                for outer_tag in map(maybe_with_layer, retag_map.keys()):
                    self[outer_tag].retag_(retag_map)

            #on to next column
        
        #end loop over columns
                                                

    def _contract_boundary_from_left_multi(
        self,
        yrange,
        xrange,
        layer_tags,
        canonize=True,
        compress_sweep='up',
        **compress_opts
    ):
        for y in range(min(yrange), max(yrange)):
            # make sure the exterior sites are a single tensor
            #
            #     ○──○──           ●──○──
            #     │╲ │╲            │╲ │╲       (for two layer tags)
            #     ●─○──○──         ╰─●──○──
            #      ╲│╲╲│╲     ==>    │╲╲│╲
            #       ●─○──○──         ╰─●──○──
            #        ╲│ ╲│             │ ╲│
            #         ●──●──           ╰──●──
            #
            for left_tag in self.supergrid_column_slice(y, xrange, get='tag'):
                self ^= left_tag
            
            for layer in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #        ○──
                #      ╱╱ ╲        (first contraction if there are two tags)
                #     ●─── ○──
                #      ╲ ╱╱ ╲
                #       ^─── ○──
                #        ╲ ╱╱
                #         ^─────
                #
                # contract layer with boundary, without dropping outer tags
                self._contract_boundary_from_left_single(
                    yrange=(y, y + 1), xrange=xrange, retag_boundary=False,
                    canonize=canonize, compress_sweep=compress_sweep, 
                    layer_tag=layer, **compress_opts)

                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for x in range(min(xrange), max(xrange) + 1):
                    inner_tag = self.grid_coo_tag(x, y + 1)
                    if (inner_tag in self.tags) and len(self.tag_map[inner_tag]) > 1:
                        outer_tensor_xy = self[self.grid_coo_tag(x,y)]
                        outer_tensor_xy.drop_tags(inner_tag)
                
                # on to next layer
            
            # end loop over layers

            # drop 'outer' tags to move boundary rightward
            for x in range(min(xrange), max(xrange)+1):
                self.drop_tags(self.grid_coo_tag(x, y))
            
            # on to next column
        
        # end loop over columns


    def contract_boundary_from_left(
        self, 
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='up',
        layer_tags=None,
        inplace=False,
        **compress_opts):

        tn = self if inplace else self.copy()

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        if layer_tags is None:
            tn._contract_boundary_from_left_single(
                yrange=yrange, xrange=xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        
        else:
            tn._contract_boundary_from_left_multi(
                yrange=yrange, xrange=xrange, layer_tags=layer_tags, 
                canonize=canonize, compress_sweep=compress_sweep,
                **compress_opts)

        return tn

    contract_boundary_from_left_ = functools.partialmethod(
        contract_boundary_from_left, inplace=True)



    
    def _contract_boundary_from_right_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='down',
        layer_tag=None,
        retag_boundary=True,
        **compress_opts
     ):
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
        }[compress_sweep]

        
        def contract_any_layers(grid_tag1, grid_tag2):

            ''' Contract any tensors with these tags, i.e. 
            on any layer (bra or ket)'''
            self.contract_((grid_tag1, grid_tag2), which='any')

        def contract_chosen_layer(grid_tag1, grid_tag2):
            ''' Only contract tensors living on the specified
            layer via `layer_tag`.

            Looks for `layer_tag` on the *second* tensor given.
            '''
            self.contract_between(tags1=grid_tag1,
                                  tags2=(grid_tag2, layer_tag))
                                
        #bound method that maps x,y to grid tag like 'S{x}{y}'
        grid_tag = self.grid_coo_tag

        if layer_tag is None:
            contract_step = contract_any_layers
            maybe_with_layer = lambda t: t

        else:
            contract_step = contract_chosen_layer
            maybe_with_layer = lambda t: (t, layer_tag)
        

        for y in range(max(yrange), min(yrange), -1):
            #
            #     ──●──●       ──●
            #       │  │         ║
            #     ──●──●  ==>  ──●
            #       │  │         ║
            #     ──●──●       ──●
            #
            retag_map = dict() # keep track of any retagging to be done afterwards

            slice_opts = dict(xrange=xrange, get='row', layer_tag=layer_tag)
            # get lists of 'nonempty' row indices in each column
            left_col = tuple(self.supergrid_column_slice(y-1, **slice_opts))
            right_col = tuple(self.supergrid_column_slice(y, **slice_opts))
            
            # do a 'prescan' to check where there are tensors and bonds
            scan_for_tensors, scan_for_bonds = dict(), dict()
            
            for x in range(min(xrange), max(xrange)+1):
                
                scan_for_tensors[x] = (x in left_col, x in right_col)

                # store if there is a bond 'floating' over the supergrid locations
                scan_for_bonds[x] = (self.check_for_vertical_bond_across(x, y-1, layer_tag),
                                     self.check_for_vertical_bond_across(x, y, layer_tag))
            
            


            for x in range(min(xrange), max(xrange)+1):
                
                # whether there are tensors to contract
                tensors_here = scan_for_tensors[x]

                # whether there are bonds 'floating' over the sites
                bonds_here = scan_for_bonds[x]
                
                if tensors_here == (True, True):
                    #
                    #       │  │         ║
                    #     ──●──●  ==>  ──●
                    #       │  │         ║
                    #
                    # tensors on both sides, contract:
                    left_tag, right_tag = grid_tag(x, y - 1), grid_tag(x, y)
                    
                    # looking for ``layer_tag`` on left-hand tensor
                    contract_step(right_tag, left_tag)
                    
                    # for retagging later
                    retag_map[right_tag] = left_tag


                elif tensors_here == (False, True):
                    
                    if bonds_here[0] == True:
                        # 
                        #   ──●──●        ──●──●         ──●
                        #     │  │          │  │           ║
                        #     |  ●     ==>  i--●     ==>   ●  
                        #     │  │          │  │           ║
                        #   ──●──●        ──●──●         ──●
                        # 
                        # no left tensor, but need to absorb a bond.
                        # Insert dummy identity and contract w/ right-tensor
                        dummy_tags = ('AUX', grid_tag(x, y - 1))

                        if layer_tags is not None:
                            dummy_tags = (*dummy_tags, layer_tag)

                        self.insert_identity_between_(where1=grid_tag(x-1, y-1),
                                                      where2=grid_tag(x+1, y-1),
                                                      layer_tag=layer_tag,
                                                      add_tags=dummy_tags)

                        contract_step(dummy_tags, grid_tag(x, y))

                        # for retagging later
                        retag_map[right_tag] = left_tag


                    else:
                        # 
                        #   ──●──●       ──●
                        #        │         │
                        #        ●   ==>   ●  
                        #        │         │
                        #   ──●──●       ──●
                        # 
                        # no left tensor or bond, just shift right-tensor
                        # leftward so we have it on the next column pass
                        # right_tensor = self[self.grid_coo_tag(x, y)]
                        # right_tensor.add_tag(self.grid_coo_tag(x, y-1))
                        # continue
                        left_tag, right_tag = grid_tag(x, y - 1), grid_tag(x, y)
                        self[right_tag].add_tag(left_tag)
                        retag_map[right_tag] = left_tag
                        
                elif tensors_here == (True, False):
                    
                    if bonds_here[1] == True:
                        # 
                        #   ──●──●        ──●──●          ──●
                        #     │  │          │  │            ║
                        #   ──●  │  ==>   ──●--i    ==>   ──●  
                        #     │  │          │  │            ║
                        #   ──●──●        ──●──●          ──●
                        # 
                        # no right-tensor but there is a bond to absorb.
                        # insert dummy identity and contract to the left
                        
                        dummy_tags = ('AUX', grid_tag(x,y))

                        self.insert_identity_between_(where1=grid_tag(x - 1, y),
                                                      where2=grid_tag(x + 1, y),
                                                      add_tags=dummy_tags)


                        # layer_tag will be on left-tensor
                        contract_step(dummy_tags, grid(x, y-1))
                        retag_map[grid_tag(x, y)] = grid_tag(x, y-1)                                                

                    else:
                        # 
                        #   ──●──●        ──●
                        #     │             │
                        #   ──●     ==>   ──●
                        #     │             │
                        #   ──●──●        ──●
                        # 
                        # no bond or tensor to push leftward
                        continue
                
                elif tensors_here == (False, False):
                    #no tensor on either side
                    continue
                    
                else:
                    raise ValueError('Missed a case!')

            #end loop over rows

            self.fuse_multibonds_()

            if canonize:
                #
                #   ──●       ──v
                #     ║         ║
                #   ──●  ==>  ──v
                #     ║         ║
                #   ──●       ──●
                #
                self._canonize_supergrid_column(y, canonize_sweep, xrange)
            
            #
            #   ──v       ──●
            #     ║         │
            #   ──v  ==>  ──^
            #     ║         │
            #   ──●       ──^
            #
            self._compress_supergrid_column(y, compress_sweep, 
                                    xrange, **compress_opts)

            # should be true if contracting more than one column at a time
            if retag_boundary:
                # drop previous tags to move boundary (previously at
                # supergrid[:][y]) to the left (y -> y-1)
                
                # for x in range(min(xrange), max(xrange)+1):         
                #     self.drop_tags(self.grid_coo_tag(x, y))       
                for outer_tag in map(maybe_with_layer, retag_map.keys()):
                    self[outer_tag].retag_(retag_map)
            
            # on to next column
        
        # end loop over columns


    def _contract_boundary_from_right_multi(
        self,
        yrange,
        xrange,
        layer_tags,
        canonize=True,
        compress_sweep='down',
        **compress_opts):

        ## update?
        self._supergrid = self.calc_supergrid()
        ##

        for y in range(max(yrange), min(yrange), -1):
            # make sure the exterior sites are a single tensor
            #
            #         ──○──○           ──○──●
            #          ╱│ ╱│            ╱│ ╱│    (for two layer tags)
            #       ──○──○─●         ──○──●─╯
            #        ╱│╱╱│╱   ==>     ╱│╱╱│
            #     ──○──○─●         ──○──●─╯
            #       │╱ │╱            │╱ │
            #     ──●──●           ──●──╯
            #
            for right_tag in self.supergrid_column_slice(y, xrange, get='tag'):
                self ^= right_tag
            

            for layer in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #         ──○
                #          ╱ ╲╲     (first contraction if there are two tags)
                #       ──○────v
                #        ╱ ╲╲ ╱
                #     ──○────v
                #        ╲╲ ╱
                #     ─────●
                #
                # contract layer with boundary, forcing ``supergrid``` to stay 
                # the same until both BRA and KET have been contracted
                self._contract_boundary_from_right_single(
                    yrange=(y, y-1), xrange=xrange, retag_boundary=False,
                    canonize=canonize, compress_sweep=compress_sweep, 
                    layer_tag=layer, **compress_opts)
                
                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for x in range(min(xrange), max(xrange) + 1):
                    inner_tag = self.grid_coo_tag(x, y - 1)
                    if (inner_tag in self.tags) and len(self.tag_map[inner_tag]) > 1:
                        outer_tensor_xy = self[self.grid_coo_tag(x,y)]
                        outer_tensor_xy.drop_tags(inner_tag)

                # on to next layer

            # end loop over layers

            for x in range(min(xrange), max(xrange)+1):
                self.drop_tags(self.grid_coo_tag(x, y))
            
            # on to next column
        
        # end loop over columns


    def contract_boundary_from_right(
        self,
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='down',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        
        tn = self if inplace else self.copy()

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        if layer_tags is None:
            tn._contract_boundary_from_right_single(
                yrange=yrange, xrange=xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        else:
            tn._contract_boundary_from_right_multi(
                yrange=yrange, xrange=xrange, layer_tags=layer_tags,
                canonize=canonize, compress_sweep=compress_sweep, 
                **compress_opts)

        return tn

    contract_boundary_from_right_ = functools.partialmethod(
        contract_boundary_from_right, inplace=True)




    def _contract_boundary_from_top_single(
        self,
        xrange,
        yrange,
        canonize=True,
        compress_sweep='right',
        layer_tag=None,
        retag_boundary=True,
        **compress_opts
    ):
        canonize_sweep = {
                'left': 'right',
                'right': 'left'
            }[compress_sweep]


        def contract_any_layers(grid_tag1, grid_tag2):

            ''' Contract any tensors with these tags, i.e. 
            on any layer (bra or ket)'''
            self.contract_((grid_tag1, grid_tag2), which='any')

        def contract_chosen_layer(grid_tag1, grid_tag2):
            ''' Only contract tensors living on the specified
            layer via `layer_tag`.

            Assumes `layer_tag` will be found on *second* given tensor.
            '''
            self.contract_between(tags1=grid_tag1,
                                  tags2=(grid_tag2, layer_tag))
                                

        # maps x,y to grid tag like 'S{x}{y}'
        grid_tag = self.grid_coo_tag

        if layer_tag is None:
            contract_step = contract_any_layers
            maybe_with_layer = lambda t:t

        else:
            contract_step = contract_chosen_layer
            maybe_with_layer = lambda t: (t, layer_tag)

        
        for x in range(min(xrange), max(xrange)):
            #
            #     ●──●──●──●──●
            #     |  |  |  |  |  ==>  ●══●══●══●══●
            #     ●──●──●──●──●       |  |  |  |  |
            #     |  |  |  |  |
            #
            retag_map = dict()

            slice_opts = dict(yrange=yrange, get='col', layer_tag=layer_tag)

            # get lists of nonempty column numbers for this row
            upper_row = list(self.supergrid_row_slice(x, **slice_opts))
            lower_row = list(self.supergrid_row_slice(x + 1, **slice_opts))

            # do a 'prescan' to check where there are tensors and bonds
            scan_for_tensors, scan_for_bonds = dict(), dict()
            
            for y in range(min(yrange), max(yrange) + 1):
                
                scan_for_tensors[y] = (y in upper_row, y in lower_row)

                scan_for_bonds[y] = (self.check_for_horizontal_bond_across(x, y, layer_tag),
                              self.check_for_horizontal_bond_across(x + 1, y, layer_tag),)


            for y in range(min(yrange), max(yrange) + 1):

                # check whether there are tensors to contract above, below
                tensors_here = scan_for_tensors[y]

                # check if there are horizontal bonds 'floating' above or below
                bonds_here = scan_for_bonds[y]

                upper_tag, lower_tag = grid_tag(x, y), grid_tag(x+1, y)

                if tensors_here == (True, True):
                    #
                    #     ──●──
                    #       |    ==>  ══●══
                    #     ──●──         |  
                    #       |
                    # tensors on both top & bottom, contract bond
                    contract_step(upper_tag, lower_tag)
                    retag_map[upper_tag] = lower_tag


                elif tensors_here == (True, False):

                    if bonds_here[1] == True:
                        #
                        #     ●──●──●       ●──●──●
                        #     |     |  ==>  |  :  |   ==>   ●══●══●
                        #     ●─────●       ●──i──●         |     |
                        #     |     |       |     |
                        #
                        # no lower-tensor, but need to absorb a bond
                        # insert dummy identity and contract w/ upper-tensor

                        dummy_tags = ('AUX', lower_tag)

                        if layer_tags is not None:
                            dummy_tags = (*dummy_tags, layer_tag)

                        self.insert_identity_between_(where1=grid(x+1, y-1),
                                                      where2=grid(x+1, y+1),
                                                      layer_tag=layer_tag,
                                                      add_tags=dummy_tags)

                        contract_step(dummy_tags, upper_tag)
                        retag_map[upper_tag] = lower_tag

                    else:
                        #
                        #     ●──●──●       
                        #     |     |  ==>  ●──●──●
                        #     ●     ●       |     |
                        #     |     |       
                        #
                        # no bond to absorb, just shift upper-tensor
                        # downward so we have it on the next row pass
                        
                        # top_tensor = self[self.grid_coo_tag(x, y)]
                        # top_tensor.add_tag(self.grid_coo_tag(x + 1, y))
                        # continue
                        self[upper_tag].add_tag(lower_tag)
                        retag_map[upper_tag] = lower_tag
                        
                elif tensors_here == (False, True):

                    if bonds_here[0] == True:
                        #
                        #     ●─────●       ●──i──●
                        #     |     |  ==>  |  :  |   ==>   ●══●══●
                        #     ●──●──●       ●──●──●         |     |
                        #     |  |  |       |     |
                        #
                        # no upper-tensor, but need to absorb a bond downward
                        # insert dummy identity and contract w/ lower-tensor

                        dummy_tags = ('AUX', upper_tag)

                        self.insert_identity_between_(where1=grid_tag(x, y-1),
                                                      where2=grid_tag(x, y+1),
                                                      add_tags=dummy_tags)

                        # look for ``layer_tag`` on lower tensor
                        contract_step(dummy_tags, lower_tag)
                        retag_map[upper_tag] = lower_tag
                        

                    else:
                        #
                        #     ●     ●       
                        #     |     |  ==>  ●──●──●
                        #     ●──●──●       |     |
                        #     |  |  |       
                        #
                        # no bond or tensor to push downward
                        continue
                
                elif tensors_here == (False, False):
                    # no tensor either above or below
                    continue
                    
                else:
                    raise ValueError('Missed a case!')
            
            # end loop over columns

            self.fuse_multibonds_()

            if canonize:
                #
                #     ●══●══<══<══<
                #     |  |  |  |  |
                #
                self._canonize_supergrid_row(x, sweep=canonize_sweep, yrange=yrange)
            
            #
            #     >──●══●══●══●  -->  >──>──●══●══●  -->  >──>──>──●══●
            #     |  |  |  |  |  -->  |  |  |  |  |  -->  |  |  |  |  |
            #     .  .           -->     .  .        -->        .  .
            #
            self._compress_supergrid_row(x, sweep=compress_sweep, 
                                        yrange=yrange, **compress_opts)

            # should be true if contracting more than one row at a time
            if retag_boundary:
                # drop previous tags to move boundary (previously at 
                # supergrid[x][:]) down by one (x -> x+1)
                # for y in range(min(yrange), max(yrange)+1):
                #     self.drop_tags(self.grid_coo_tag(x, y))
                for outer_tag in map(maybe_with_layer, retag_map.keys()):
                    self[outer_tag].retag_(retag_map)
            
            # on to next row
        
        # end loop over rows


    
    def _contract_boundary_from_top_multi(
        self,
        xrange,
        yrange,
        layer_tags,
        canonize=True,
        compress_sweep='left',
        **compress_opts
    ):
        self._supergrid = self.calc_supergrid()

        for x in range(min(xrange), max(xrange)):
            # make sure the exterior sites are a single tensor
            #
            #    ●─○●─○●─○●─○●─○         ●══●══●══●══●
            #    │ ││ ││ ││ ││ │  ==>   ╱│ ╱│ ╱│ ╱│ ╱│
            #    ●─○●─○●─○●─○●─○       ●─○●─○●─○●─○●─○
            #    │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │   (for two layer tags)
            #
            for top_tag in self.supergrid_row_slice(x, yrange, get='tag'):
                self ^= top_tag
            

            for layer in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #    ●══<══<══<══<
                #    │╲ │╲ │╲ │╲ │╲
                #    │ ○──○──○──○──○
                #    │ ││ ││ ││ ││ │  (first contraction if there are two tags)
                #
                self._contract_boundary_from_top_single(
                    xrange=(x, x + 1), yrange=yrange, retag_boundary=False,
                    canonize=canonize, compress_sweep=compress_sweep, 
                    layer_tag=layer, **compress_opts)
                
                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for y in range(min(yrange), max(yrange) + 1):
                    inner_tag = self.grid_coo_tag(x + 1, y)
                    if (inner_tag in self.tags) and len(self.tag_map[inner_tag]) > 1:
                        outer_tensor_xy = self[self.grid_coo_tag(x,y)]
                        outer_tensor_xy.drop_tags(inner_tag)

                # on to next layer
            
            # end loop over layers

            for y in range(min(yrange), max(yrange) + 1):
                self.drop_tags(self.grid_coo_tag(x, y))
            
            # self._supergrid = self.calc_supergrid()

            # on to next row
        
        # end loop over rows


    def contract_boundary_from_top(
        self, 
        xrange, 
        yrange=None, 
        canonize=True,
        compress_sweep='right',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
    
        tn = self if inplace else self.copy()

        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)

        if layer_tags is None:
            tn._contract_boundary_from_top_single(
                yrange=yrange, xrange=xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        else:
            tn._contract_boundary_from_top_multi(
                yrange=yrange, xrange=xrange, layer_tags=layer_tags,
                canonize=canonize, compress_sweep=compress_sweep, 
                **compress_opts)

        return tn

    contract_boundary_from_top_ = functools.partialmethod(
        contract_boundary_from_top, inplace=True)
                

    def _contract_boundary_from_bottom_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='left',
        layer_tag=None,
        retag_boundary=True,
        **compress_opts
    ):
        canonize_sweep = {
            'left': 'right',
            'right': 'left',
        }[compress_sweep]


        def contract_any_layers(grid_tag1, grid_tag2):

            ''' Contract any tensors with these tags, i.e. 
            on any layer (bra or ket)'''
            self.contract_((grid_tag1, grid_tag2), which='any')

        def contract_chosen_layer(grid_tag1, grid_tag2):
            ''' Only contract tensors living on the specified
            layer via `layer_tag`.

            Assumes `layer_tag` is on the *second* tensor given.
            '''
            self.contract_between(tags1=grid_tag1,
                                  tags2=(grid_tag2, layer_tag))

        
        #bound method that maps (x,y) to tuple[tags_xy,] or None
        grid_tag = self.grid_coo_tag

        if layer_tag is None:
            contract_step = contract_any_layers
            maybe_with_layer = lambda t: t

        else:
            contract_step = contract_chosen_layer
            maybe_with_layer = lambda t: (t, layer_tag)


        for x in range(max(xrange), min(xrange), -1):
            #
            #     │  │  │  │  │
            #     ●──●──●──●──●       │  │  │  │  │
            #     │  │  │  │  │  ==>  ●══●══●══●══●
            #     ●──●──●──●──●
            #
            retag_map = dict()
            slice_opts = dict(yrange=yrange, get='col', layer_tag=layer_tag)
            
            # get lists of nonempty column numbers for this row
            upper_row = tuple(self.supergrid_row_slice(x-1, yrange, get='col'))
            lower_row = tuple(self.supergrid_row_slice(x, yrange, get='col'))

            # do a 'prescan' to check where there are tensors and bonds
            scan_for_tensors, scan_for_bonds = dict(), dict()
            
            for y in range(min(yrange), max(yrange) + 1):
                
                scan_for_tensors[y] = (y in upper_row, y in lower_row)

                scan_for_bonds[y] = (self.check_for_horizontal_bond_across(x-1, y, layer_tag),
                                      self.check_for_horizontal_bond_across(x, y, layer_tag),)

            for y in range(min(yrange), max(yrange)+1):

                # check whether there are tensors to contract
                tensors_here = scan_for_tensors[y]

                # check if there are horizontal bonds 'floating' over the supergrid locations
                bonds_here = scan_for_bonds[y]

                upper_tag, lower_tag = grid_tag(x-1, y), grid_tag(x, y)
                
                if tensors_here == (True, True):
                    #
                    #       │  
                    #     ──●──         │  
                    #       │    ==>  ══●══
                    #     ──●──
                    #
                    # tensors on both bottom & top
                    
                    # look for ``layer_tag`` on the top tensor
                    contract_step(lower_tag, upper_tag)
                    retag_map[lower_tag] = upper_tag
                    
                

                elif tensors_here == (False, True):
                    # no tensor above

                    if bonds_here[0] == True:
                        #
                        #     │     │       │     │
                        #     ●─────●       ●──i──●         │     │
                        #     │     │  ==>  │  :  │    ==>  ●══●══●
                        #     ●──●──●       ●──●──●
                        #
                        # no upper-tensor but need to absorb a bond
                        # insert dummy identity above, contract w/ lower-tensor
                        dummy_tags = ('AUX', upper_tag) 

                        if layer_tags is not None:
                            dummy_tags = (*dummy_tags, layer_tag)

                        self.insert_identity_between_(where1=grid(x-1, y-1),
                                                      where2=grid(x-1, y+1),
                                                      layer_tag=layer_tag,
                                                      add_tags=dummy_tags)

                        contract_step(dummy_tags, lower_tag)
                        retag_map[lower_tag] = upper_tag

                    else:
                        #
                        #     │     │       
                        #     ●     ●       │     │
                        #     │     │  ==>  ●──●──●
                        #     ●──●──●       
                        #
                        # no upper-tensor or bond, just shift lower-tensor
                        # upward so we have it on the next row pass
                        # bottom_tensor = self[self.grid_coo_tag(x, y)]
                        # bottom_tensor.add_tag(self.grid_coo_tag(x - 1, y))
                        # continue
                        self[lower_tag].add_tag(upper_tag)
                        retag_map[lower_tag] = upper_tag
                
                
                elif tensors_here == (True, False):
                    
                    
                    if bonds_here[1] == True:
                        #
                        #     │  │  │       │  │  │
                        #     ●──●──●       ●──●──●         │  │  │
                        #     │     │  ==>  │  :  │    ==>  ●══●══●
                        #     ●─────●       ●──i──●
                        #
                        # no lower-tensor but there is a bond to absorb.
                        # insert dummy identity and contract w/ upper-tensor

                        dummy_tags = ('AUX', lower_tag)

                        self.insert_identity_between_(where1=grid(x, y-1),
                                                      where2=grid(x, y+1),
                                                      add_tags=dummy_tags)

                        # ``layer_tag`` will be on upper-tensor
                        contract_step(dummy_tags, upper_tag)
                        retag_map[lower_tag] = upper_tag

                    else:
                        #
                        #     │  │  │       
                        #     ●──●──●       │  │  │
                        #     │     │  ==>  ●──●──●
                        #     ●     ●       
                        #
                        # no tensor or bond to push upward
                        continue

                elif tensors_here == (False, False):
                    
                    #     │     │       
                    #     ●─────●       │     │
                    #     │     │  ==>  ●═════●
                    #     ●─────●       
                    #
                    # no tensor either above or below
                    continue

                else:
                    raise ValueError('Missed a case!')
            
            # end loop over columns

            self.fuse_multibonds_()


            if canonize:
                #
                #     │  │  │  │  │
                #     ●══●══<══<══<
                #
                self._canonize_supergrid_row(x, sweep=canonize_sweep, yrange=yrange)
            
            #
            #     │  │  │  │  │  -->  │  │  │  │  │  -->  │  │  │  │  │
            #     >──●══●══●══●  -->  >──>──●══●══●  -->  >──>──>──●══●
            #     .  .           -->     .  .        -->        .  .
            #
            self._compress_supergrid_row(x, sweep=compress_sweep,
                                        yrange=yrange, **compress_opts)

            
            # should be true if contracting more than one row at a time
            if retag_boundary:
                # drop previous tags to move boundary (previously 
                # at supergrid[x, :]) up by one (x -> x-1)
                for outer_tag in map(maybe_with_layer, retag_map.keys()):
                    self[outer_tag].retag_(retag_map)
            

            # on to next row        
        
        # end loop over rows                                            
        
    def _contract_boundary_from_bottom_multi(
        self, 
        xrange,
        yrange,
        layer_tags,
        canonize=True,
        compress_sweep='left',
        **compress_opts        
    ):  
        ## update?
        self._supergrid = self.calc_supergrid()
        ##

        for x in range(max(xrange), min(xrange), -1):
            # first ensure the exterior sites are a single tensor
            #
            #    │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │   (for two layer tags)
            #    ●─○●─○●─○●─○●─○       ●─○●─○●─○●─○●─○
            #    │ ││ ││ ││ ││ │  ==>   ╲│ ╲│ ╲│ ╲│ ╲│
            #    ●─○●─○●─○●─○●─○         ●══●══●══●══●
            #
            
            for bottom_tag in self.supergrid_row_slice(x, yrange, get='tag'):
                self ^= bottom_tag
            
            for tag in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #    │ ││ ││ ││ ││ │  (first contraction if there are two tags)
                #    │ ○──○──○──○──○
                #    │╱ │╱ │╱ │╱ │╱
                #    ●══<══<══<══<
                #
                # contract layer with boundary, forcing ``supergrid``` to stay 
                # the same until both BRA and KET have been contracted
                self._contract_boundary_from_bottom_single(
                    yrange=yrange, xrange=(x, x-1), retag_boundary=False,
                    canonize=canonize, compress_sweep=compress_sweep, 
                    layer_tag=tag, **compress_opts)
                
                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for y in range(min(yrange), max(yrange) + 1):
                    inner_tag = self.grid_coo_tag(x - 1, y)
                    if (inner_tag in self.tags) and len(self.tag_map[inner_tag]) > 1:
                        outer_tensor_xy = self[self.grid_coo_tag(x,y)]
                        outer_tensor_xy.drop_tags(inner_tag)

                # on to next layer
            
            # end loop over layers

            for y in range(min(yrange), max(yrange)+1):
                    self.drop_tags(self.grid_coo_tag(x, y))

            # self._supergrid = self.calc_supergrid()

            # on to next row
        
        # end loop over rows



    def contract_boundary_from_bottom(
        self, 
        xrange, 
        yrange=None, 
        canonize=True,
        compress_sweep='left',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
    
        tn = self if inplace else self.copy()

        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)

        if layer_tags is None:
            tn._contract_boundary_from_bottom_single(
                yrange=yrange, xrange=xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)

        else:
            tn._contract_boundary_from_bottom_multi(
                yrange=yrange, xrange=xrange, layer_tags=layer_tags, 
                canonize=canonize, compress_sweep=compress_sweep, 
                **compress_opts)

        return tn

    contract_boundary_from_bottom_ = functools.partialmethod(
        contract_boundary_from_bottom, inplace=True)


    def contract_boundary(
        self,
        around=None,
        layer_tags=None,
        max_separation=1,
        sequence=None,
        bottom=None,
        top=None,
        left=None,
        right=None,
        inplace=False,
        contract_optimize='random-greedy',
        **boundary_contract_opts,
    ):
        """Contract boundary inwards::
            ●──●──●──●       ●──●──●──●       ●──●──●
            │  │  │  │       │  │  │  │       ║  │  │
            ●──●──●──●       ●──●──●──●       ^──●──●       >══>══●       >──v
            │  │ij│  │  ==>  │  │ij│  │  ==>  ║ij│  │  ==>  │ij│  │  ==>  │ij║
            ●──●──●──●       ●══<══<══<       ^──<──<       ^──<──<       ^──<
            │  │  │  │
            ●──●──●──●
            
        Optionally from any or all of the boundary, in multiple layers, and
        stopping around a region.

        Parameters
        ----------
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        max_separation : int, optional
            If ``around is None``, when any two sides become this far apart
            simply contract the remaining tensor network.
        sequence : sequence of {'b', 'l', 't', 'r'}, optional
            Which directions to cycle throught when performing the inwards
            contractions: 'b', 'l', 't', 'r' corresponding to *from the*
            bottom, left, top and right respectively. If ``around`` is
            specified you will likely need all of these!
        bottom : int, optional
            The initial bottom boundary row, defaults to ``2 Lx - 2``.
        top : int, optional
            The initial top boundary row, defaults to 0.
        left : int, optional
            The initial left boundary column, defaults to 0.
        right : int, optional
            The initial right boundary column, defaults to ``2 Ly - 2``..
        inplace : bool, optional
            Whether to perform the contraction in place.
        boundary_contract_opts
            Supplied to
            :meth:`~QubitEncodeNetwork._contract_boundary_from_bottom`,
            :meth:`~QubitEncodeNetwork._contract_boundary_from_left`,
            :meth:`~QubitEncodeNetwork._contract_boundary_from_top`,
            or
            :meth:`~QubitEncodeNetwork._contract_boundary_from_right`,
            including compression and canonization options.
        """
        
        tn = self if inplace else self.copy()

        boundary_contract_opts['layer_tags'] = layer_tags

        # starting borders (default to *supergrid* dimensions!)
        if bottom is None:
            bottom = 2 * tn.Lx - 2
        if top is None:
            top = 0
        if left is None:
            left = 0
        if right is None:
            right = 2 * tn.Ly - 2
        
        # setting contraction sequence
        if around is not None:
            if sequence is None:
                sequence = 'bltr'
            stop_i_min = min(x[0] for x in around)
            stop_i_max = max(x[0] for x in around)
            stop_j_min = min(x[1] for x in around)
            stop_j_max = max(x[1] for x in around)
        
        elif sequence is None:
            # contract in along short dimension
            if self.Lx >= self.Ly:
                sequence = 'b'
            else:
                sequence = 'l'


        # keep track of whether we have hit the ``around`` region.
        reached_stop = {direction: False for direction in sequence}

        for direction in cycle(sequence):

            if direction == 'b':
                # check if we have reached the 'stop' region
                if (around is None) or (bottom - 1 > stop_i_max):
                    tn.contract_boundary_from_bottom_(
                        xrange=(bottom, bottom - 1),
                        yrange=(left, right),
                        compress_sweep='left',
                        **boundary_contract_opts
                    )
                    bottom -= 1
                else:
                    reached_stop[direction] = True
            
            elif direction == 'l':
                if (around is None) or (left + 1 < stop_j_min):
                    tn.contract_boundary_from_left_(
                        yrange=(left, left + 1),
                        xrange=(bottom, top),
                        compress_sweep='up',
                        **boundary_contract_opts
                    )
                    left += 1
                else:
                    reached_stop[direction] = True
            
            elif direction == 't':
                if (around is None) or (top + 1 < stop_i_min):
                    tn.contract_boundary_from_top_(
                        xrange=(top, top + 1),
                        yrange=(left, right),
                        compress_sweep='right',
                        **boundary_contract_opts
                        )
                    top += 1
                else:
                    reached_stop[direction] = True
            
            elif direction == 'r':
                if (around is None) or (right - 1 > stop_j_max):
                    tn.contract_boundary_from_right_(
                        yrange=(right, right - 1),
                        xrange=(bottom, top),
                        compress_sweep='down',
                        **boundary_contract_opts
                    )
                    right -= 1
                else:
                    reached_stop[direction] = True
            
            else:
                raise ValueError("Sequence can only be from bltr")


            if around is None:
                # check whether TN is thin enough to just contract
                thin_strip = (
                    (bottom - top <= max_separation) or
                    (right - left <= max_separation)
                )
                if thin_strip:
                    return tn.contract(all, optimize=contract_optimize)
            
            elif all(reached_stop.values()):
                break

        return tn

    contract_boundary_ = functools.partialmethod(
            contract_boundary, inplace=True)
                    
                

    def compute_row_environments(self, dense=False, **compress_opts):
        """Compute the ``2 * grid_Lx `` 1D boundary tensor networks describing
        the lower and upper environments of each row in this 2D tensor network,
        *assumed to represent the norm*.

        The 'above' environment for row ``i`` will be a contraction of all
        rows ``i - 1, i - 2, ...`` etc::

             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
        
        The 'below' environment for row ``i`` will be a contraction of all
        rows ``i + 1, i + 2, ...`` etc::

            ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
        
        
        Such that
        ``envs['above', i] & self.select(self.row_tag(i)) & envs['below', i]``
        would look like::

             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
            o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o
            ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●

        And be (an approximation of) the norm centered around row ``i``
        Parameters
        ----------
        dense : bool, optional
            If true, contract the boundary in as a single dense tensor.
        compress_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_bottom`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_top`
            .
        Returns
        -------
        row_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of row ``i`` will be stored in
            ``row_envs['below', i]`` and ``row_envs['above', i]``.
        """
        ## NOTE: should this be automatic?
        # self.fill_rows_with_identities_()
        ##

        row_envs = dict()

        grid_Lx = self.grid_Lx
        grid_Ly = self.grid_Ly

        self._add_row_col_tags()
        # first_row = self.supergrid_row_slice(x=0, yrange=(0, 2*self.Ly-2), get='tag')

        # downwards pass #
        row_envs['above', 0] = qtn.TensorNetwork([])
        first_row_tag = self.row_tag(0)
        env_top = self.copy()

        if dense:
            env_top ^= first_row_tag

        row_envs['above', 1] = env_top.select(first_row_tag).copy()

        for x in range(2, grid_Lx):

            if dense:
                env_top ^= (self.row_tag(x-2), self.row_tag(x-1))
            else:
                env_top.contract_boundary_from_top_(
                    xrange=(x - 2, x - 1), **compress_opts)

            row_envs['above', x] = env_top.select(first_row_tag).copy()


        # upwards pass #
        row_envs['below', grid_Lx - 1] = qtn.TensorNetwork([])
        last_row_tag = self.row_tag(grid_Lx - 1)
        env_bottom = self.copy()
        
        if dense:
            env_bottom ^= last_row_tag
        
        row_envs['below', grid_Lx - 2] = env_bottom.select(last_row_tag).copy()

        for x in range(grid_Lx - 3, -1, -1):
            
            if dense:
                env_bottom ^= (self.row_tag(x + 1), self.row_tag(x + 2))
            else:
                env_bottom.contract_boundary_from_bottom_(
                        xrange=(x + 1, x + 2), **compress_opts)
            
            row_envs['below', x] = env_bottom.select(last_row_tag).copy()

        return row_envs



    def compute_col_environments(self, dense=False, **compress_opts):
        r"""Compute the ``2 * self.Ly`` 1D boundary tensor networks describing
        the left and right environments of each column in this 2D tensor
        network, assumed to represent the norm.
        The 'left' environment for column ``j`` will be a contraction of all
        columns ``j - 1, j - 2, ...`` etc::
            ●<
            ┃
            ●<
            ┃
            ●<
            ┃
            ●<
        The 'right' environment for row ``j`` will be a contraction of all
        rows ``j + 1, j + 2, ...`` etc::
            >●
             ┃
            >●
             ┃
            >●
             ┃
            >●
        Such that
        ``envs['left', j] & self.select(self.col_tag(j)) & envs['right', j]``
        would look like::
               ╱o
            ●< o| >●
            ┃  |o  ┃
            ●< o| >●
            ┃  |o  ┃
            ●< o| >●
            ┃  |o  ┃
            ●< o╱ >●
        And be (an approximation of) the norm centered around column ``j``
        Parameters
        ----------
        dense : bool, optional
            If true, contract the boundary in as a single dense tensor.
        compress_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_left`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_right`
            .
        Returns
        -------
        col_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of column ``j`` will be stored
            in ``row_envs['left', j]`` and ``row_envs['right', j]``.
        """
        ### NOTE: should column identities be automatic?

        # self.fill_cols_with_identities_()

        ####

        col_envs = dict()

        grid_Lx = self.grid_Lx # = 2 * Lx - 1 
        grid_Ly = self.grid_Ly # = 2 * Ly - 1

        self._add_row_col_tags()

        # rightward pass #
        col_envs['left', 0] = qtn.TensorNetwork([])
        first_column_tag = self.col_tag(0)
        env_right = self.copy()
        
        if dense:
            env_right ^= first_column_tag
        
        col_envs['left', 1] = env_right.select(first_column_tag).copy()
        
        for y in range(2, grid_Ly):
            
            if dense:
                env_right ^= (self.col_tag(y-2), self.col_tag(y-1))
            else:
                env_right.contract_boundary_from_left_(
                    yrange=(y - 2, y - 1), **compress_opts)
            
            col_envs['left', y] = env_right.select(first_column_tag).copy()


        # leftward pass #
        
        col_envs['right', grid_Ly - 1] = qtn.TensorNetwork([])
        last_column_tag = self.col_tag(grid_Ly - 1)
        env_left = self.copy()

        if dense:
            env_left ^= last_column_tag
        
        col_envs['right', grid_Ly - 2] = env_left.select(last_column_tag).copy()

        for y in range(grid_Ly - 3, -1, -1):
            if dense:
                env_left ^= (self.col_tag(y + 1), self.col_tag(y + 2))
            else:
                env_left.contract_boundary_from_right_(
                    yrange=(y + 1, y + 2),
                    **compress_opts)
            
            col_envs['right', y] = env_left.select(last_column_tag).copy()

        return col_envs        


    def _add_row_col_tags(self, rows=True, cols=True):
        '''Add row and column tags like 'ROW{x}', 'COL{y}', to every 
        occupied supergrid location.
        '''
        for (x, y), tag_xy in self.gen_occupied_grid_tags(with_coo=True):
            if rows:
                self.add_tag(tag=self.row_tag(x), where=tag_xy)
            if cols:
                self.add_tag(tag=self.col_tag(y), where=tag_xy)
                
    
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
        return self.qlattice().get_edges(which)


    @property
    def num_sites(self):
        return self.num_verts + self.num_faces


    @property
    def num_verts(self):
        return (self.Lx * self.Ly)

    @property
    def num_faces(self):
        '''Number of occupied face qubits, assuming 
        lattice has an even number of faces!
        '''
        return int((self.Lx - 1) * (self.Ly - 1) / 2)

    @property
    def codespace_dims(self):
        '''List like [phys_dim] * num_verts,
        i.e. dimensions for vertex subspace
        '''
        # return self.qlattice().codespace_dims
        return [self.phys_dim] * self.num_verts

    @property
    def simspace_dims(self):
        '''List like [phys_dim] * num_sites,
        i.e. dimensions for full qubit space
        '''
        # return self.qlattice().simspace_dims
        return [self.phys_dim] * self.num_sites


    def bond(self, where1, where2):
        '''Get index (should only be one!) of bond connecting 
        the tensors at given sites. ``where`` can take qubit ints
        or tags, but *not* coos.
        '''
        bond, = self[where1].bonds(self[where2])
        return bond


    def find_bonds_between(self, tags1, tags2):
        '''List of bonds between tensors with given tags.
        Returns empty [] if tensors are not connected.
        '''
        return list(self[tags1].bonds(self[tags2]))


    def check_for_vertical_bond_across(self, x, y, layer_tag=None):
        '''Check if there is a vertical bond crossing supergrid
        coordinate ``x,y``, i.e. whether there are connected tensors 
        above and below. Can optionally give a ``layer_tag`` to 
        look only in 'BRA' or 'KET' layer.
             
                 y   y+1
                 :   :
          x-1 -> ●───●──       
                 │  
            x -> │          ==> True
                 │  
          x+1 -> ●───●──       


        Assumes:
        -------
        There is no bond at the supergrid edges
        Need only check above and below by 1 node
        '''
        if x == 0 or x == 2 * self.Lx - 2: 
            return False
        
        tags_above = self.grid_coo_tag(x - 1, y)
        tags_below = self.grid_coo_tag(x + 1, y)

        # if (tags_above is None) or (tags_below is None):
        #     return False
        
        if layer_tag is not None:
            # tags_above = (*tags_above, layer_tag)
            # tags_below = (*tags_below, layer_tag)
            tags_above = (tags_above, layer_tag)
            tags_below = (tags_below, layer_tag)
        
        # if there isn't a tensor matching either of the tags, return False
        if not all(map(self.check_for_matching_tags, (tags_above, tags_below))):
            return False

        return bool(self.find_bonds_between(tags1=tags_above,
                                            tags2=tags_below))

    
    def check_for_horizontal_bond_across(self, x, y, layer_tag=None):
        '''Check if there is a horizontal bond crossing supergrid
        coordinate `x,y`, i.e. whether there are connected tensors 
        to the right and left. Can optionally give a ``layer_tag``
        to specifically look only in 'BRA' or 'KET' layer.
        
        Assumes:
        -------
        There are no bonds at supergrid edges
        Need only check nearest right/left-neighbors
        '''
        if y == 0 or y == 2 * self.Ly - 2:
            return False
        
        left_tags = self.grid_coo_tag(x, y-1)
        right_tags = self.grid_coo_tag(x, y+1)

        
        if layer_tag is not None:
            left_tags = (left_tags, layer_tag)
            right_tags = (right_tags, layer_tag)


        if not all(map(self.check_for_matching_tags, (left_tags, right_tags))):
            return False

        return bool(self.find_bonds_between(tags1=left_tags,
                                            tags2=right_tags))




    def graph(
        self, 
        fix_lattice=True, 
        layer_tags=None, 
        with_gate=False, 
        auto_detect_layers=True,
        **graph_opts):
        '''
        TODO: DEBUG ``fix_tags`` PARAM
              GRAPH GATE LAYERS

        Overload ``TensorNetwork.graph`` for clean/aesthetic lattice-fixing,
        unless `fix_lattice` is set False. 

        NOTE: now graphs with (0,0)-coordinate origin at bottom left.

        `auto_detect_layers` will check if 'BRA', 'KET' are in `self.tags`,
        and if so will attempt to graph as a two-layer sandwich.
        Can also specify `layer_tags` specifically for TNs with other layers.
        '''
        
        graph_opts.setdefault('color', ['VERT','FACE','GATE'])
        graph_opts.setdefault('show_tags', False)
        graph_opts.setdefault('show_inds', True)

        if all((auto_detect_layers == True,
                'BRA' in self.tags,
                'KET' in self.tags)):
            layer_tags=('BRA', 'KET')


        def get_layer_fix_tags(layer_tag=None, offset=0.0):
            
            # if layer_tag == 'GATE':
            #     gate_tensors = self.select_tensors('GATE')
            #     for gt in gate_tensors:

            
            
            # layer_tag = tags_to_oset(layer_tag)
            
            # scale x and y differently if desired
            LATX, LATY = 3, 4
            fix_tags = dict()
            
            layer_tn = self if layer_tag is None else self.select_tensors(layer_tag)
            layer_tn = qtn.TensorNetwork(layer_tn).view_like_(self)

            nonempty_sites =  {(x,y): layer_tn.supergrid(x,y) for x,y in product(range(self.grid_Lx), range(self.grid_Ly))
                                if layer_tn.supergrid(x,y) is not None}
            
            for (x,y), tags in nonempty_sites.items():
                tags_xy = tags_to_oset(tags) | tags_to_oset(layer_tag)
                fix_tags.update({tuple(tags_xy): (LATY * (y + 0.5 * offset), 
                                                 LATX * (x + offset))})
        
            return fix_tags
        
        
        if fix_lattice == False:
            super().graph(**graph_opts)
        
        # elif 'GATE' in tuple(layer_tags):
        #     fix_tags = dict()
            
        #     for k, layer in enumerate(tags_to_oset(layer_tags) - tags_to_oset('GATE')):
        #         fix_tags.update(get_layer_fix_tags(layer_tag= layer,
        #                                             offset= 0.5 * k))
            

        #     grid_coo_regex = re.compile(r'S\d,\d')
        #     for t in self.select_tensors('GATE'):
        #         neighbors = self.select_neighbors(t.tags)
        #         for nt in neighbors:
        #             neighbor_coo = [x for x in nt.tags if grid_coo_regex.match(x)]



        else:
            # try:
            if layer_tags is None:
                fix_tags = get_layer_fix_tags()
        
            else:
                fix_tags = dict()
                for k, layer in enumerate(tags_to_oset(layer_tags)):
                    fix_tags.update(get_layer_fix_tags(layer_tag= layer,
                                                        offset= 0.5 * k))

            super().graph(fix=fix_tags, **graph_opts)
            

    graph_layers = functools.partialmethod(graph, 
                            layer_tags=('BRA','KET'))



    # def exact_projector_from_matrix(self, Udag_matrix):
    #     Nfermi, Nqubit = self.num_verts(), self.num_sites()

    #     if Udag_matrix.shape != (2**Nfermi, 2**Nqubit):
    #         raise ValueError('Wrong U* shape')
        
    #     sim_dims = self.simspace_dims()
    #     code_dims = self.codespace_dims()
        
    #     sim_inds = [f'q{i}' for i in range(Nqubit)]
        
    #     Udagger = Udag_matrix.reshape(code_dims+sim_dims)

    #     Udagger = qtn.Tensor()


    def flatten(self, inplace=False, fuse_multibonds=True):
        '''Contract all tensors at each grid location to one, thus 
        squishing together layers ('BRA', 'KET', etc) into a flat TN.
        '''
        tn = self if inplace else self.copy()

        # for each grid location, check if there are tensors there
        nonempty_coo_tags = filter(
                        lambda t: t in tn.tags, 
                        starmap(tn.grid_coo_tag, 
                        product(range(tn.grid_Lx), 
                                range(tn.grid_Ly)))
                                )
        
        # squish together the layers at each 'occupied' grid coo
        for tag_xy in nonempty_coo_tags:
            tn ^= tag_xy

        if fuse_multibonds:
            tn.fuse_multibonds_()
        
        return tn
        # return tn.view_as_(QubitEncodeNetFlat)

    flatten_ = functools.partialmethod(flatten, inplace=True)


    def multiply_each_qubit_tensor(self, x, inplace=False):
        """Multiply all qubit (physical) tensors by scalar ``x``,
        i.e. avoid multiplying the dummy identity tensors.

        Parameters
        ----------
        x : scalar
            The number that multiplies each *qubit* tensor in the tn
        inplace : bool, optional
            Whether to perform the multiplication inplace.
        """
        multiplied = self if inplace else self.copy()

        for t in multiplied.select_tensors('QUBIT'):
            t.modify(apply=lambda data: data * x)
        
        return multiplied


    # def absorb_face_left(self, face_coo, inplace=False, fuse_multibonds=True):
    #     '''NOTE: CURRENTLY ONLY FOR FLAT NETWORKS
    #     Need way to do one layer at a time for sandwiches.
    #     partition? 
    #     '''
    #     tn = self if inplace else self.copy()

    #     face_tag = tn.maybe_convert_face(face_coo)
        
    #     fi, fj = face_coo

    #     #tags for corner vertex sites
    #     ul_tag = tn.vert_coo_tag(fi, fj)
    #     ur_tag = tn.vert_coo_tag(fi, fj + 1)
    #     dl_tag = tn.vert_coo_tag(fi + 1, fj)
    #     dr_tag = tn.vert_coo_tag(fi + 1, fj + 1)

    #     #corner bonds
    #     ul_bond = tn.bond(ul_tag, face_tag)
    #     ur_bond = tn.bond(ur_tag, face_tag)
    #     dl_bond = tn.bond(dl_tag, face_tag)
    #     dr_bond = tn.bond(dr_tag, face_tag)


    #     face_tensor = tn[face_tag]
        
    #     #split face tensor into two, upper/lower tensors
    #     tensors = face_tensor.split(
    #                 left_inds=(ul_bond, ur_bond),
    #                 right_inds=(dl_bond, dr_bond),
    #                 get=None,
    #                 ltags=['UPPER','SPLIT'],
    #                 rtags=['LOWER','SPLIT'])

    #     tn.delete(face_tag)
    #     tn |= tensors

    #     # Absorb split-tensors into the vertices
    #     tn.contract_((ul_tag, 'UPPER'))
    #     tn.contract_((dl_tag, 'LOWER'))

    #     tn[ul_tag].drop_tags([face_tag, 'FACE', 'UPPER'])
    #     tn[dl_tag].drop_tags([face_tag, 'FACE', 'LOWER'])

    #     if fuse_multibonds:
    #         tn.fuse_multibonds_()

    #     return tn


    # absorb_face_left_ = functools.partialmethod(absorb_face_left,
    #                                             inplace=True)

    def rotate_face_qubits(self, layer_tag=None, inplace=False):
        """Find every face qubit in this TN (optionally, only those matching 
         ``layer_tag``) and 'rotate' the bonds  to make "X"-connectivity look
         like "+"-connectivity
        """
        tn = self if inplace else self.copy()
        
        for x, y in tn.gen_face_coos(including_empty=False):
            
            tn.rotate_face_qubit_bonds_(x, y, layer_tag)

        return tn
    
    rotate_face_qubits_ = functools.partialmethod(rotate_face_qubits,
                                                inplace=True)


    def rotate_face_qubit_bonds(
        self, 
        x, 
        y, 
        layer_tag=None,
        inplace=False, 
        fuse_multibonds=True
    ):
        tn = self if inplace else self.copy()

        face_tag = tn.face_coo_tag(x, y)

        corner_coos = ((x - 1, y - 1), 
                       (x - 1, y + 1),
                       (x + 1, y + 1),
                       (x + 1, y - 1),)
                       
        # corner_coos = tn.corner_coos_around_face(i, j)
        corner_tags = list(starmap(tn.vert_coo_tag, corner_coos))        

        for k, ctag in enumerate(corner_tags):
            add_tags = tags_to_oset(('AUX', f'TEMP{k}')) | tags_to_oset(layer_tag)

            tn.insert_identity_between_(where1=face_tag, where2=ctag, 
                    layer_tag=layer_tag, add_tags=add_tags)
            
            next_ctag = corner_tags[0] if k==3 else corner_tags[k+1]
            
            #insert I between this and 'next' corner in the square
            tn.insert_identity_between_(where1=ctag, where2=next_ctag, 
                    layer_tag=layer_tag, add_tags=add_tags)        


        for k, ccoo in enumerate(corner_coos):

            tn ^= f'TEMP{k}'

            corner_qnumber = tn.vertex_coo_map(*ccoo)

            next_ccoo = corner_coos[0] if k == 3 else corner_coos[k + 1]

            # get the grid coo between the two corners
            mid_coo = tn.coo_between(xy1=ccoo, xy2=next_ccoo)
            new_grid_tag = tn.grid_coo_tag(*mid_coo)

            # replace temporary tag with the proper grid coo tag
            tn.retag_({f'TEMP{k}': new_grid_tag})
        
            # also add the 'ADJ{k}' tag to know where to reabsorb if necessary
            tn.add_tag(tag=f"ADJ{corner_qnumber}", 
                where=tags_to_oset(new_grid_tag) | tags_to_oset(layer_tag),
                which='all',)

        # tn._supergrid = tn.calc_supergrid()
        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn

    rotate_face_qubit_bonds_ = functools.partialmethod(rotate_face_qubit_bonds, 
                                                    inplace=True)


    # def reabsorb_face_qubit_identities(self, layer_tag=None, inplace=False):

    #     tn = self if inplace else self.copy()

    #     adj_tags = {k: f'ADJ{k}' for k in range((tn.num_sites) if f'ADJ{k}' in tn.tags)}

    #     for k, tag_k in adj_tags.items():
    #         # absorb identity into a qubit site
    #         dummy_tags = tn[tag_k].tags
    #         tn.contract_tags(tags=(tn.qubit_tag(k), tag_k))
    #         tn.drop_tags(dummy_tags)
        
            
        


    def insert_identity_between(self, where1, where2, layer_tag=None, add_tags=None, inplace=False):
        '''Inserts an identity tensor at the bond between tags ``where1, where2``.

        Params
        ------
        where1, where2: str
            The tags specifying between which two tensors to insert an identity 
        layer_tag: str, optional
            If needed, can specify a layer tag (e.g. 'BRA' or 'KET') to be 
            added to the ``where`` tags.
        add_tags: sequence of str, optional
            The new identity tensor can be inserted with additional ``add_tags``.
        '''
        tn = self if inplace else self.copy()

        tto = qtn.tensor_core.tags_to_oset

        if layer_tag is not None:
            where1 = tto(where1) | tto(layer_tag)
            where2 = tto(where2) | tto(layer_tag)

        T1, T2 = tn[where1], tn[where2]
        
        I = insert_identity_between_tensors(T1, T2, add_tags=add_tags)
        tn |= I
        return tn
        

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


    # def supergrid_coo_between_verts(self, ij1, ij2):
    #     '''Returns the *supergrid* coordinates lying halfway between
    #     the two given *vertex* coordinates.
    #     '''
    #     i1, j1 = ij1 #correspond to supercoo (2i, 2j)
    #     i2, j2 = ij2

    #     x = i1 + i2
    #     y = j1 + j2
    #     #supercoo in-between is (2i1+2i2)/2, (2j1+2j2)/2
    #     return (x, y)

    def coo_between(self, xy1, xy2):
        '''Coordinate between two neighboring coos
        '''

        x1, y1 = xy1
        x2, y2 = xy2

        return (int((x1 + x2)/2), int((y1 + y2)/2))


    def _fill_column_with_identities(self, y, xrange=None, layer_tag=None):

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        grid_coo_tag = self.grid_coo_tag

        for x in range(min(xrange), max(xrange) + 1):
            
            floating_bond = self.check_for_vertical_bond_across(x, y, layer_tag)
            
            if floating_bond:

                dummy_tag = tags_to_oset(('AUX', grid_coo_tag(x, y))) 

                self.insert_identity_between_(where1=grid_coo_tag(x-1, y), 
                                              where2=grid_coo_tag(x+1, y),
                                              layer_tag=layer_tag,
                                              add_tags=dummy_tag | tags_to_oset(layer_tag))
                # self._update_supergrid(x, y, dummy_tag)  
                

    def _fill_row_with_identities(self, x, yrange=None, layer_tag=None):
        
        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)
        
        grid_coo_tag = self.grid_coo_tag
        
        for y in range(min(yrange), max(yrange) + 1):

            floating_bond = self.check_for_horizontal_bond_across(x, y, layer_tag)

            if floating_bond:

                dummy_tag = tags_to_oset(('AUX', grid_coo_tag(x, y)))

                self.insert_identity_between_(where1=grid_coo_tag(x, y-1),
                                              where2=grid_coo_tag(x, y+1),
                                              layer_tag=layer_tag,
                                              add_tags=dummy_tag | tags_to_oset(layer_tag))
                # self._update_supergrid(x, y, dummy_tag)                                              


    def fill_rows_with_identities(self, xrange=None, layer_tag=None, inplace=False):
        '''For each row in xrange (inclusive), fill the row with
        auxiliary tensors in the 'empty' supergrid locations. Can
        optionally specify a ``layer_tag`` (like 'BRA')
        
             │  │  │         │  │  │
             ●──●──●         ●──●──●    
             │     │         │     │    
         x   ●─────●   ==>   ●──i──●  x
             │     │         │     │
             ●──●──●         ●──●──●    
        
        xrange: tuple[int], optional.
            (first row, last row) to fill. Defaults to all rows.
        '''

        tn = self if inplace else self.copy()

        if xrange is None:
            xrange = (0, 2 * self.Lx - 2)

        for x in range(min(xrange), max(xrange) + 1):

            tn._fill_row_with_identities(x=x, layer_tag=layer_tag)

        return tn


    fill_rows_with_identities_ = functools.partialmethod(fill_rows_with_identities,
                                                        inplace=True)


    def fill_cols_with_identities(self, yrange=None, layer_tag=None, inplace=False):
        '''For each col in yrange (inclusive), fill the row with
        auxiliary tensors in the 'empty' supergrid locations.
        
             y                y 

        ──●──●──●        ──●──●──●
          │  │  │          │  │  │ 
        ──●  │  ●  ==>   ──●  i  ● 
          │  │  │          │  │  │ 
        ──●──●──●        ──●──●──● 

        yrange: optional, defaults to all columns.
            (first col, last col to fill)
        '''
        tn = self if inplace else self.copy()

        if yrange is None:
            yrange = (0, 2 * self.Ly - 2)
        
        for y in range(min(yrange), max(yrange) + 1):
            
            tn._fill_column_with_identities(y=y, layer_tag=layer_tag)

        return tn
        
    
    fill_cols_with_identities_ = functools.partialmethod(fill_cols_with_identities,
                                                        inplace=True)


    def _compute_plaquette_environments_row_first(
        self,
        x_bsz,
        y_bsz,
        second_dense=None,
        row_envs=None,
        **compute_environment_opts
    ):
        # needs_face_qubit = (x_bsz >= 3) or (y_bsz >= 3)


        if second_dense is None:
            second_dense = x_bsz < 2
        
        # first contract from either side to produce column environments
        if row_envs is None:
            row_envs = self.compute_row_environments(
                **compute_environment_opts)
        
        # now form strips and contract from sides, for each row
        col_envs = dict()
        for x in range(self.grid_Lx - x_bsz + 1):
        # for x in [2*i for i in range(self.Lx) if 2*i < self.grid_Lx - x_bsz + 1]:
            
            # if x % 2 == 1:
            #     continue
            
            #
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #     ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┬
            #     | | | | | | | | | | | | | | | | | | | |     ┊ x_bsz
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┴
            #     ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #

            # row_x = self.row_environment_sandwich(x0=x, x_bsz=x_bsz, 
            #                                     row_envs=row_envs,
            #                                     simple_test=True)
            row_x = qtn.TensorNetwork((
                        row_envs['above', x],
                        self.select_any([self.row_tag(x + dx) for dx in range(x_bsz)]),
                        row_envs['below', x + x_bsz - 1],
            ), check_collisions=False).view_as_(QubitEncodeNet, like=self)
                                                
            #
            #           y_bsz
            #           <-->               second_dense=True
            #       ●──      ──●
            #       │          │            ╭──     ──╮
            #       ●── .  . ──●            │╭─ . . ─╮│     ┬
            #       │          │     or     ●         ●     ┊ x_bsz
            #       ●── .  . ──●            │╰─ . . ─╯│     ┴
            #       │          │            ╰──     ──╯
            #       ●──      ──●
            #     'left'    'right'       'left'    'right'
            #
            col_envs[x] = row_x.compute_col_environments(
                xrange=(max(x - 1, 0), min(x + x_bsz, self.grid_Lx - 1)),
                dense=second_dense, **compute_environment_opts)
        
        # range through all possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()

        #only pick vertices as corners
        for x0, y0 in product(range(self.grid_Lx), range(self.grid_Ly)):
            
            if (x0 > self.grid_Lx - x_bsz) or (y0 > self.grid_Ly - y_bsz):
                continue
            
            # skip if the plaquette has no face qubit?
            # if needs_face_qubit:
            #     plaq_tags = self.plaquette_at(xy=(x0,y0), x_bsz=x_bsz, 
            #                                 y_bsz=y_bsz, get='tags')


            # if needs_face_qubit and self.face_coo_map(x0 + 1, y0 + 1) is None:
            #     continue


            # select bordering tensors from:
            #
            #       L──A──A──R    <- A from the row environments
            #       │  │  │  │
            #  x0   L──●──i──R
            #       │  │  │  │    
            #  x0+1 L──i──●──R    <- L, R from the column environments
            #       │  │  │  │
            #  x0+2 L──●──i──R
            #       │  │  │  │
            #       L──B──B──R    <- B from the row environments
            #
            #         y0  y0+1
            #

            left_coos = ((x0 + dx, y0 - 1) for dx in range(-1, x_bsz + 1))
            left_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, left_coos))
            )

            right_coos = ((x0 + dx, y0 + y_bsz) for dx in range(-1, x_bsz + 1))
            right_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, right_coos))
            )

            above_coos = ((x0 - 1, y0 + dy) for dy in range(y_bsz))
            above_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, above_coos))
            )

            below_coos = ((x0 + x_bsz, y0 + dy) for dy in range(y_bsz))
            below_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, below_coos))
            )

            env_xy = qtn.TensorNetwork((
                col_envs[x0]['left', y0].select_any(left_tags),
                col_envs[x0]['right', y0 + y_bsz - 1].select_any(right_tags),
                row_envs['above', x0].select_any(above_tags),
                row_envs['below', x0 + x_bsz - 1].select_any(below_tags),
            ), check_collisions=False)

            #absorb any rank-2 corner tensors
            env_xy.rank_simplify_()

            plaquette_envs[(x0, y0), (x_bsz, y_bsz)] = env_xy

        return plaquette_envs


    def _compute_plaquette_environments_col_first(
        self, 
        x_bsz, 
        y_bsz,
        second_dense=None,
        col_envs=None,
        **compute_environment_opts
    ):
        # needs_face_qubit = (x_bsz >= 3) or (y_bsz >= 3)

        if second_dense is None:
            second_dense = y_bsz < 2
        
        # first contract from either side, get column envs
        if col_envs is None:
            col_envs = self.compute_col_environments(
                **compute_environment_opts)
        
        # form vertical strips and contract from top + bottom, 
        # for each column
        row_envs = dict()
        for y in range(self.grid_Ly - y_bsz + 1):
            
            # skip the face-qubit columns?
            # if y % 2 == 1:
            #     continue
            
            #
            #        y_bsz
            #        <-->
            #
            #      ╭─╱o─╱o─╮
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #

            col_y = qtn.TensorNetwork((
                col_envs['left', y],
                self.select_any([self.col_tag(y + dy) for dy in range(y_bsz)]),
                col_envs['right', y + y_bsz - 1],
            ), check_collisions=False).view_as_(QubitEncodeNet, like=self)
            
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'above'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'below'
            #     ●──●──●──●      ╰──●──╯
            #
            row_envs[y] = col_y.compute_row_environments(
                yrange=(max(y - 1, 0), min(y + y_bsz, self.grid_Ly - 1)),
                dense=second_dense, **compute_environment_opts)
            
        # range through possible plaquettes, slecting correct boundary tensors
        # from either column or row envs
        plaquette_envs = dict()

        #only choose vertex qubits as plaquette corners
        for x0, y0 in self.gen_vertex_coos():

            if (x0 > self.grid_Lx - x_bsz) or (y0 > self.grid_Ly - y_bsz):
                continue
            
            # we want to select bordering tensors from:
            #
            #          A──A──A──A    <- A from the row environments
            #          │  │  │  │
            #     i0+1 L──●──●──R
            #          │  │  │  │    <- L, R from the column environments
            #     i0   L──●──●──R
            #          │  │  │  │
            #          B──B──B──B    <- B from the row environments
            #
            #            j0  j0+1
            #

            left_coos = ((x0 + dx, y0 - 1) for dx in range(-1, x_bsz + 1))
            left_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, left_coos))
            )

            right_coos = ((x0 + dx, y0 + y_bsz) for dx in range(-1, x_bsz + 1))
            right_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, right_coos))
            )

            above_coos = ((x0 - 1, y0 + dy) for dy in range(y_bsz))
            above_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, above_coos))
            )

            below_coos = ((x0 + x_bsz, y0 + dy) for dy in range(y_bsz))
            below_tags = tuple(
                starmap(self.grid_coo_tag, filter(self.valid_supercoo, below_coos)))
            
            
            right_tags = (t for t in right_tags if t in self.tags)
            left_tags = (t for t in left_tags if t in self.tags)
            above_tags = (t for t in above_tags if t in self.tags)
            below_tags = (t for t in below_tags if t in self.tags)


            env_xy = qtn.TensorNetwork((
                col_envs['left', y0].select_any(left_tags),
                col_envs['right', y0 + y_bsz - 1].select_any(right_tags),
                row_envs[y0]['below', x0 + x_bsz - 1].select_any(below_tags),
                row_envs[y0]['above', x0].select_any(above_tags),
            ), check_collisions=False)

            # absorb any rank-2 corner tensors
            env_xy.rank_simplify_()

            plaquette_envs[(x0, y0), (x_bsz, y_bsz)] = env_xy

        return plaquette_envs

    # NOTE: row-first working but still need to debug column-first plaq_envs
    def compute_plaquette_environments(
        self, 
        x_bsz=2,
        y_bsz=2,
        first_contract=None,
        second_dense=None,
        **compute_environment_opts,
    ):
        r"""Compute all environments like::
            second_dense=False   second_dense=True (& first_contract='columns')
              ●──●                  ╭───●───╮
             ╱│  │╲                 │  ╱ ╲  │
            ●─.  .─●    ┬           ●─ . . ─●    ┬
            │      │    ┊ x_bsz     │       │    ┊ x_bsz
            ●─.  .─●    ┴           ●─ . . ─●    ┴
             ╲│  │╱                 │  ╲ ╱  │
              ●──●                  ╰───●───╯
              <-->                    <->
             y_bsz                   y_bsz
        Use two boundary contractions sweeps.
        Parameters
        ----------
        x_bsz : int, optional
            The size of the plaquettes in the x-direction (number of rows).
        y_bsz : int, optional
            The size of the plaquettes in the y-direction (number of columns).
        first_contract : {None, 'rows', 'columns'}, optional
            The environments can either be generated with initial sweeps in
            the row or column direction. Generally it makes sense to perform
            this approximate step in whichever is smaller (the default).
        second_dense : None or bool, optional
            Whether to perform the second set of contraction sweeps (in the
            rotated direction from whichever ``first_contract`` is) using
            a dense tensor or boundary method.
        compute_environment_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_col_environments`
            or
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_row_environments`
            .
        """
        if first_contract is None:
            if x_bsz > y_bsz:
                first_contract = 'columns'
            elif y_bsz > x_bsz:
                first_contract = 'rows'
            elif self.Lx >= self.Ly:
                first_contract = 'rows'
            else:
                first_contract = 'columns'

        compute_env_fn = {
            'rows': self._compute_plaquette_environments_row_first,
            'columns': self._compute_plaquette_environments_col_first,
        }[first_contract]

        return compute_env_fn(
            x_bsz=x_bsz, y_bsz=y_bsz, second_dense=second_dense,
            **compute_environment_opts)



    def plaquette_at(self, xy, bsz, get='tn'):
        x0, y0 = xy
        x_bsz, y_bsz = bsz
        # plaq_coos = ((x0 + dx, y0 + dy) for dx, dy in product(range(x_bsz), range(y_bsz)))
        plaq_coos = self.plaquette_to_site_coos(plaq=(xy, (x_bsz, y_bsz)))
        
        plaq_tags = tuple(filter(
            lambda t: t in self.tags,
            starmap(self.grid_coo_tag, plaq_coos)
            ))

        if get == 'tags':
            return plaq_tags

        tensors = self.select_tensors(plaq_tags, which='any')
        
        return qtn.TensorNetwork(tensors, check_collisions=False).view_as_(QubitEncodeNet,
                                                                        like=self)

    def calc_qubit_plaquette_sizes(self, qubit_groups, autogroup=True):
        """Compute sequence of block size pairs (x_bsz, y_bsz) to cover
        all the qubit groupings in ``qubit_groups``.

        Args
        ----
        qubit_groups: sequence of tuple[int]
            The set of qubit groups; each group is a tuple of ints
            (could be length 2 or 3) labeling qubits that get acted 
            on together by a Ham term
        autogroup: bool, optional:
            Whether to return minimal sequence of blocksizes that will
            cover all the groups, or merge them into a single ``((x_bsz, y_bsz),)``.

        Returns
        -------
        block_sizes: tuple[tuple[int]]
            Pairs of block sizes like ((1,2), (2,1))
        """
        # maps a group of qubits to group of coos, e.g.
        # (0, 1, 9) --> ((0, 0), (0, 2), (1, 1))
        # (1, 2) --> ((0, 2), (0, 4))
        qs2coos = lambda qs: tuple(map(self.qubit_to_coo_map, qs))


        # get groups of grid coordinates
        # e.g.  ((0, 1, 9), (0, 9, 2)) --> coo_groups[0] = ((0, 0), (0, 2), (1, 1)) 
        #                                  coo_groups[1] = ((0, 0), (1, 1), (0, 4))
        coo_groups = map(qs2coos, qubit_groups)
        

        # get rectangular plaq sizes for each group
        # e.g.  ((0, 0), (0, 2), (1, 1)) --> (2, 3)
        # 
        #   ●───o───●──   ┬
        #   │   │   │     ┊  x_bsz = 2
        #   o───●───o──   ┴ 
        #   │   │   │     
        #   o───o───o── 
        # 
        #   <------->  y_bsz = 3
        # 
        block_sizes = set()
        for group in coo_groups:
            xs, ys = zip(*group)
            x_bsz = max(xs) - min(xs) + 1
            y_bsz = max(ys) - min(ys) + 1
            block_sizes.add((x_bsz, y_bsz))
        
        # block_sizes = {tuple(max(ds) - min(ds) + 1 for ds in zip(*group)) 
        #               for group in coo_groups}

        # remove block sizes that can be contained in another block size
        #     e.g. {(1, 2), (2, 1), (2, 2)} -> ((2, 2),)
        relevant_block_sizes = []

        for b in block_sizes:

            is_included = any(
                (b[0] <= b2[0]) and (b[1] <= b2[1]) 
                for b2 in block_sizes - {b}
                )

            if not is_included:
                relevant_block_sizes.append(b)
        
        bszs = tuple(sorted(relevant_block_sizes))

        # bszs = tuple(sorted(
        # b for b in block_sizes
        # if not any(
        #     (b[0] <= b2[0]) and (b[1] <= b2[1])
        #     for b2 in block_sizes - {b}
        #         )
        # ))

        # return all plaquette sizes separately
        if autogroup:
            return bszs


        # otherwise make a big blocksize to cover all terms
        #     e.g. ((1, 2), (3, 2)) -> ((3, 2),)
        #          ((1, 2), (2, 1)) -> ((2, 2),)
        return (tuple(map(max, zip(*bszs))),)


    def calc_plaquette_map(self, plaquettes, face_qubits=True):
        """Generate a dictionary of all the coordinate pairs in ``plaquettes``
        mapped to the 'best' (smallest) rectangular plaquette that contains them.
        
        Will optionally compute for 3-length combinations as well, to
        capture 3-local qubit interactions like (vertex, vertex, face)
        interactions.

        Args:
            plaquettes: sequence of tuple[tuple[int]]
                Sequence of plaquettes like ((x0, y0), (dx, dy))

            face_qubits: bool, optional
                Whether to include 3-local interactions as well 
                as 2-local (pairwise).

        NOTE: Inefficient bc most of the coo mappings are irrelevant; only
        *qubit* sites will be acted on (auxiliary grid coos don't matter). 
        """
        # sort in descending total plaquette size
        plqs = sorted(plaquettes, key=lambda p: (-p[1][0] * p[1][1], p))
        
        mapping = dict()

        
        for p in plqs:
            sites = self.plaquette_to_site_coos(p)
        
            # for pairwise (no face qubit) interactions
            # like ij_a, ij_b
            for coo_pair in combinations(sites, 2):

                if all(tuple(map(self.is_qubit_coo, coo_pair))):
                    mapping[coo_pair] = p


            if face_qubits:
                # include 3-local interactions
                # like ij_a, ij_b, ij_c
                for coo_triple in combinations(sites, 3):
                    
                    if all(tuple(map(self.is_qubit_coo, coo_triple))):
                        mapping[coo_triple] = p

        return mapping

            



    def plaquette_to_site_coos(self, plaq):
        """Turn a plaquette ``((x0, y0), (dx, dy))`` into the grid
        coordinates of the sites it contains.

        Examples
        --------
            >>> plaquette_to_site_coos([(3, 4), (2, 2)])
            ((3, 4), (3, 5), (4, 4), (4, 5))
        """
        (x0, y0), (dx, dy) = plaq
        return tuple((x, y) for x in range(x0, x0 + dx)
                            for y in range(y0, y0 + dy))

    # def plaquette_to_grid_tags(self, p):


##                       ##
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
        '_Lx',
        '_Ly',
        '_phys_dim',
        '_grid_tag_id',
        '_site_tag_id',
        '_aux_tag_id',
        '_phys_ind_id'
    )
    
            
    def __init__(
            self, 
            tn, 
            *, # qlattice, 
            Lx=None,
            Ly=None,
            phys_dim=2,
            grid_tag_id='S{},{}',
            site_tag_id='Q{}',
            phys_ind_id='q{}',
            aux_tag_id='IX{}Y{}',
            **tn_opts
        ):
        
        #shortcut for copying QEN vectors
        if isinstance(tn, QubitEncodeVector):
            super().__init__(tn)
            return


        self._Lx = Lx
        self._Ly = Ly
        self._phys_dim = phys_dim

        self._grid_tag_id = grid_tag_id
        self._site_tag_id = site_tag_id
        self._aux_tag_id = aux_tag_id

        self._phys_ind_id = phys_ind_id

        
        super().__init__(tn, **tn_opts)

    @classmethod
    def product_state_from_bitstring(cls, Lx, Ly, bitstring, phys_dim=2, 
        bond_dim=3, dtype='complex128',**tn_opts):
        '''Create a product state with local spins *specified* in the
        given `bitstring`. 
        '''
        
        all_opts = {'Lx': Lx,
                    'Ly': Ly,
                    'bitstring': bitstring,
                    'phys_dim': phys_dim,
                    'bond_dim': bond_dim,
                    'dtype': dtype,
                    **tn_opts}

        tn = make_product_state_net(**all_opts)
        
        return cls(tn=tn, Lx=Lx, Ly=Ly, 
            phys_dim=phys_dim, **tn_opts)


    @classmethod
    def rand_product_state(cls, Lx, Ly, phys_dim=2, 
        bond_dim=3, dtype='complex128',**tn_opts):
        '''Create a product state of local spins, generated from
        a random bitstring.
        '''

        #total sites is #vertices + #faces
        num_sites = Lx * Ly + int((Lx-1) * (Ly-1) / 2)
        
        random_bitstring = ''.join(
            str(randint(0,1)) for _ in range(num_sites))

        all_opts = {'Lx': Lx,
                    'Ly': Ly,
                    'bitstring': random_bitstring,
                    'phys_dim': phys_dim,
                    'bond_dim': bond_dim,
                    'dtype': dtype,
                    **tn_opts}

        tn = make_product_state_net(**all_opts)
        
        return cls(tn=tn, Lx=Lx, Ly=Ly, 
            phys_dim=phys_dim, **tn_opts)



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
        Lx, Ly = qlattice.lattice_shape
        phys_dim = qlattice.local_dim

        return cls.rand(Lx=Lx, Ly=Ly, phys_dim=phys_dim, 
                    bond_dim=bond_dim, **tn_opts)


    @classmethod
    def rand(cls, Lx, Ly, phys_dim=2, bond_dim=3, dtype='complex128',**tn_opts):
        
        rand_tn = make_random_net(Lx=Lx, Ly=Ly, 
                                  phys_dim=phys_dim,
                                  bond_dim=bond_dim,
                                  dtype=dtype,
                                  **tn_opts)
        
        return cls(tn=rand_tn, Lx=Lx, Ly=Ly, phys_dim=phys_dim, **tn_opts)                                


    @classmethod
    def rand_local_product_state(cls, Lx, Ly, phys_dim=2, bond_dim=3, dtype='complex128', **tn_opts):
        
        local_prod_state = local_product_state_net(Lx, Ly, phys_dim, bond_dim, dtype=dtype, **tn_opts)
        
        return cls(tn=local_prod_state, Lx=Lx, Ly=Ly, phys_dim=phys_dim, **tn_opts)

    @property
    def phys_ind_id(self):
        '''Format string for the physical indices
        '''
        return self._phys_ind_id
    
    def _vert_coo_ind(self,i,j):
        '''Physical index for vertex-qubit at coo (i,j)
        '''
        k = self.vertex_coo_map(i,j)
        return self.phys_ind_id.format(k)


    def _face_coo_ind(self, i, j):
        '''Physical index for face-qubit at coo (i,j)
        '''
        k = self.face_coo_map(i, j)
        return None if k is None else self.phys_ind_id.format(k)
        # if k is None:
        #     return None
        # return self.phys_ind_id.format(k)
        
        
    def vec_to_dense(self, normalize=True):
        '''Return this state as dense vector, i.e. a qarray with 
        shape (-1, 1), in the order assigned to the local sites.
        '''
        # use the order that the qubits are numbered in
        inds_seq = (self._phys_ind_id.format(i) 
                    for i in range(self.num_sites))

        psi_d = self.to_dense(inds_seq).reshape(-1,1)

        if normalize:
            return psi_d / np.linalg.norm(psi_d)
        
        return psi_d



    def make_norm(self, layer_tags=('KET','BRA'), return_all=False):
        '''<psi|psi> as an uncontracted ``QubitEncodeNet``.

        If ``return_all == True`` return (norm, bra, ket), otherwise
        just norm. 
        '''
        ket = self.copy()
        ket.add_tag(layer_tags[0])

        bra = ket.H.retag({layer_tags[0]: layer_tags[1]})

        norm  = bra | ket
        
        if return_all:
            return norm, bra, ket

        return norm
    


    def apply_gate(
        self,
        G, 
        where,
        contract=False,
        tags=['GATE'],
        inplace=False, 
        info=None,
        **compress_opts
    ):
        '''Apply dense array ``G`` acting at sites ``where``,
        preserving physical indices. Uses ``self._phys_ind_id``
        and the integer(s) ``where`` to apply ``G`` at the
        correct sites.

        By default, tags the new gate tensor with "GATE".

        NOTE: this method doesn't assume only 2-qubit gates, can 
        handle 3-qubit gates. But it DOES assume the qubits being 
        acted on are direct nearest neighbors -- skips long-range
        procedure.

        Params:
        ------            
        G : array
            Gate to apply, should be compatible with 
            shape ([physical_dim] * 2 * len(where))
        
        where: sequence of ints
            The qubits on which to act, using the (default) 
            custom numbering that labels both face and vertex
            sites.
        
        inplace: bool, optional
            If False (default), return copy of TN with gate applied
        
        contract: {False, True, 'split', 'reduce_split'}, optional
            
            -False: (default) leave all gates uncontracted
            -True: contract gates into one tensor in the lattice
            -'split': uses tensor_2d.gate_split method for two-site gates
            -'reduce_split': TODO

        '''

        # if isinstance(G, qtn.TensorNetwork):
        #     self.apply_mpo(G, where, inplace, contract)

        psi = self if inplace else self.copy()

        #G may be a one-site gate
        if isinstance(where, Integral): 
            where = (where,)

        numsites = len(where) #gate acts on `numsites`

        dp = self.phys_dim #local physical dimension
        tags = qtn.tensor_2d.tags_to_oset(tags)

        G = qtn.tensor_1d.maybe_factor_gate_into_tensor(G, dp, numsites, where)

        #new physical indices 'q{i}'
        site_inds = [self.phys_ind_id.format(i) for i in where] 

        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        
        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))

        TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=tags)
        
        if contract is False:
            #attach gates without contracting any bonds
            #
            #       │   │      <- site_inds
            #       GGGGG
            #       │╱  │╱     <- bond_inds
            #     ──●───●──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)
            psi |= TG
            return psi


        elif contract is True or numsites==1:
            #just contract the physical indices
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)
            
            #sites that used to have physical indices
            site_tids = psi._get_tids_from_inds(bond_inds, which='any')
           
            # pop the sites (inplace), contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]

            psi |= qtn.tensor_contract(*pts, TG)

            return psi
        

        #the original tensors at sites being acted on
        original_ts = [psi[q] for q in where]

        #list of (num_sites - 1) elements; the bonds connecting
        # each site in ``where`` to the next (open boundary)
        bonds_along = [next(iter(qtn.bonds(t1, t2)))
                    for t1, t2 in qu.utils.pairwise(original_ts)]
        
        if contract == 'split' and numsites==2:
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #     <SVD>
            # 
            gss_opts = {'TG' : TG,
                        'where' : where,
                        'string': where,
                        'original_ts' : original_ts,
                        'bonds_along' : bonds_along,
                        'reindex_map' : reindex_map,
                        'site_ix' : site_inds,
                        'info' : info,
                        **compress_opts}

            qtn.tensor_2d.gate_string_split_(**gss_opts)
            return psi                        


        elif contract == 'reduce_split' and numsites == 2:
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gsrs_opts = {'TG': TG,
                         'where': where,
                         'string': where,
                         'original_ts': original_ts,
                         'bonds_along': bonds_along,
                         'reindex_map': reindex_map,
                         'site_ix': site_inds,
                         'info': info,
                         **compress_opts}
            
            qtn.tensor_2d.gate_string_reduce_split_(**gsrs_opts)
            return psi

        else:
            raise ValueError('Unknown contraction')


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

            # psi.fuse_multibonds_()

            return psi

        else:
            raise NotImplementedError('Approx. contraction for MPO gates')
        

    apply_mpo_ = functools.partialmethod(apply_mpo, inplace=True)
    

    def apply_stabilizer(self, qubits, gates, inplace=False, contract=True):
        '''Apply each of the 1-site gates making up a stabilizer to
        the corresponding sites.

        qubits: tuple(int)
            The integer labels of the qubits to act on
        gates: tuple(gate/array)
            The 1-qubit gates to apply, in same order

        stab: dict-like
            Map qubit sites (ints) to 1-site Paulis, e.g.
            {1: pauli(z), 2: pauli(z), 6: pauli(x), 4: pauli(z),
             5: pauli(z)} for the stabilizer like

             Z---Z
         X---:   :
             Z---Z

        This is a convenience method to efficiently apply an 8-site
        stabilizer operator as 8 separate 1-site gates (or less than 
        8, if at the lattice edge) rather than one dense 8-leg tensor
        '''
        psi = self if inplace else self.copy()

        for where, gate in zip(qubits, gates):
            psi.apply_gate_(G=gate, where=where, contract=contract)
        
        return psi

    

    apply_stabilizer_ = functools.partialmethod(apply_stabilizer, inplace=True)

    #TODO: make efficient (compute norm separately?)
    #      check whether setup_bmps is necessary?
    def compute_stabilizer_expec(self, qubits, gates, setup_bmps=True, norm=None, **contract_opts):
        '''Returns <psi|S|psi> for a specified stabilizer. Does not change TN inplace. 

        qubits: tuple(int)
            Integer labels of the qubits to act on (in order)
        
        gates: tuple(array)
            1-qubit gates to apply on qubits (in order)
        
        setup_bmps: bool, optional (defaults True)
            Whether to rotate face-qubits. If state is
            already face-rotated, can specify False.
        
        norm: float, optional
            If given a norm, will return the normalized 
            expectation <psi|S|psi> / norm
        '''
        ket = self.copy()
        
        # Need to rotate face qubits before applying layer tags
        if setup_bmps:
            ket.setup_bmps_contraction_()
        
        ket.add_tag('KET')
        bra = ket.H.retag({'KET': 'BRA'})
        boundary_contract_opts = {'layer_tags': ('KET', 'BRA'),
                                  **contract_opts}

        S_ket = ket.apply_stabilizer(qubits, gates, contract=True)

        expectation = (bra | S_ket).contract_boundary(**boundary_contract_opts)

        if norm is None:
            return expectation 

        return expectation / norm




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
            
            where = self.vertex_coo_map(i,j)
            G_ket = self.apply_gate(G, where=(where,))

            nxy_array[i][j] = (bra | G_ket) ^ all
            

        if return_array: 
            return nxy_array

        return np.sum(nxy_array)            
    

    
    def compute_ham_expec(self, Ham, normalize=True):
        '''Return <psi|H|psi> (inefficiently computed)

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
            E += (bra | self.apply_mpo(mpo, where, contract=True))^all
        
        if not normalize:
            return E
        
        normsquared = self.make_norm()^all
        return E / normsquared


    def apply_trotter_gates_(self, Ham, tau, **contract_opts):

        for where, exp_gate in Ham.gen_trotter_gates(tau):
            self.apply_gate_(G=exp_gate, where=where, **contract_opts)



    #TODO: inds --> qubits
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


    def compute_local_expectation(
        self,
        qubit_terms,
        normalized=False,
        autogroup=True,
        contract_optimize='auto',
        return_all=False,
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):
        # TODO: make an equivalent method but for stabilizer gates, so that 
        # we can evaluate 8 1-site gates rather than a dense 8-site gates

        r"""Compute the sum of many local expecations by essentially forming
        the reduced density matrix of all required plaquettes.
        Parameters
        ----------
        qubit_terms : dict[tuple[int]: array or quimb.tensor.MPO]
            A dictionary mapping sets of *qubits* to raw operators, which will
            be supplied to
            :meth:`~QubitEncodeVector.apply_gate` or to
            :meth:`~QubitEncodeVector.apply_mpo`.
        
        normalized : bool, optional
            If True, normalize the value of each local expectation by the local
            norm: $\langle O_i \rangle = Tr[\rho_p O_i] / Tr[\rho_p]$.
        
        autogroup : bool, optional
            If ``True`` (the default), group terms into horizontal and vertical
            sets to be computed separately (usually more efficient) if
            possible.
        
        return_all : bool, optional
            Whether to the return all the values individually as a dictionary
            of coordinates to tuple[local_expectation, local_norm].
        
        plaquette_envs : None or dict, optional
            Supply precomputed plaquette environments.

        plaquette_map : None, dict, optional
            Supply the mapping of which plaquettes (denoted by
            ``((x0, y0), (dx, dy))``) to use for which coordinates, it will be
            calculated automatically otherwise.
        
        plaquette_env_options
            Supplied to
            :meth:`~QubitEncodeNet.compute_plaquette_environments`
            to generate the plaquette environments.

        Returns
        -------
        scalar or dict
        """
        
        norm, bra, ket = self.make_norm(return_all=True)

        # the groups of qubits that get acted on together, e.g.
        # (vertex, vertex, face): (0, 1, 9), (0, 2, 9), or
        # (vertex, vertex, ----): (1, 2), (2, 5)
        # for 2-body and 3-body terms
        qubit_groups = qubit_terms.keys()

        if plaquette_envs is None:
            # set some sensible defaults
            plaquette_env_options.setdefault('layer_tags', ('BRA','KET'))
            
            # until col-first environments are working properly
            plaquette_env_options.setdefault('first_contract', 'rows')

            plaquette_envs = dict()
            
            for x_bsz, y_bsz in self.calc_qubit_plaquette_sizes(qubit_groups, autogroup):
                plaquette_envs.update(norm.compute_plaquette_environments(
                    x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options))
            
        if plaquette_map is None:

            # max number of qubits acted on at a time
            max_local = max(tuple(map(len, qubit_groups)))

            with_face_qubits = max_local > 2

            # find what plaquette to use for each group of qubits
            plaquette_map = self.calc_plaquette_map(plaquette_envs, with_face_qubits)

        # map plaquettes to list[(qubit group, ham gate)] of tuples
        # that use that plaquette
        plaq2qubits = defaultdict(list)
        for where, G in qubit_terms.items():

            qubit_coos = tuple(sorted(map(self.qubit_to_coo_map, where)))
            p = plaquette_map[qubit_coos]
            plaq2qubits[p].append((where, G))
        

        expecs = dict()
        for p in plaq2qubits:
            # site tags in plaquette
            site_tags = starmap(self.grid_coo_tag, self.plaquette_to_site_coos(p))
            site_tags = tuple(filter(lambda t: t in self.tags, site_tags))

            # plaq_coos = self.plaquette_to_site_coos(p)
            # site_tags = tuple(filter(
            # lambda t: t in self.tags,
            # starmap(self.grid_coo_tag, plaq_coos)
            # ))
            # check which sites are in plaquette though?
            # some sites may be empty
            
            bra_and_env = bra.select_any(site_tags) | plaquette_envs[p]
            ket_local = ket.select_any(site_tags)
            ket_local.view_as_(QubitEncodeVector, like=self)

            with oe.shared_intermediates():
                # local norm estimate for this plaquette
                if normalized:
                    norm_x0y0 = (
                        ket_local | bra_and_env
                    ).contract(all, optimize=contract_optimize)
                else:
                    norm_x0y0 = None
                
                for where, G in plaq2qubits[p]:
                    expec_xy = (
                        ket_local.apply_gate(G, where, contract=False) | bra_and_env
                    ).contract(all, optimize=contract_optimize)

                    expecs[where] = expec_xy, norm_x0y0
            
        if return_all:
            return expecs
        
        if normalized:
            return functools.reduce(add, (e / n for e, n in expecs.values()))
        
        return functools.reduce(add, (e for e, _ in expecs.values()))
        
    
    def normalize(
        self,
        balance_bonds=False,
        equalize_norms=False,
        inplace=False,
        **boundary_contract_opts,
    ):
        """Normalize this ``QubitEncodeVector``.
        Warning: Needs to be setup for boundary contraction!
        (i.e. via setup_bmps_contraction())

        Params
        ------
        inplace : bool, optional
            Whether to perform the normalization inplace.
        balance_bonds : bool, optional
            Whether to balance the bonds after normalization, a form of
            conditioning.
        equalize_norms : bool, optional
            Whether to set all the tensor norms to the same value after
            normalization, another form of conditioning.
        boundary_contract_opts
            Supplied to
            :meth:`~QubitEncodeVector.contract_boundary`,
            by default, two layer contraction will be used.
        """
        norm = self.make_norm()

        # default to two layer contraction
        boundary_contract_opts.setdefault('layer_tags', ('KET', 'BRA'))

        nfact = norm.contract_boundary(**boundary_contract_opts)

        num_qubits = len(self.select_tensors('QUBIT'))
        
        # only modify the qubit sites, i.e. leave identities alone
        n_ket = self.multiply_each_qubit_tensor(
            nfact**(-1 / (2 * num_qubits)), inplace=inplace)

        if balance_bonds:
            n_ket.balance_bonds_()

        if equalize_norms:
            n_ket.equalize_norms_()

        return n_ket

    normalize_ = functools.partialmethod(normalize, inplace=True)


    #************Converting to quimb TensorNetwork2D*************#


    def convert_to_tensor_network_2d(
        self, 
        dummy_size=1,
        remap_coordinate_tags=True,
        transpose_tensor_shapes=True,
        relabel_physical_inds=False,
        insert_physical_inds=False,
        new_index_id='k{},{}'
    ):
        '''Given a (Lx, Ly) lattice `self`, returns a ``QubitEncodeVector`` 
        object on a (2Lx-1, 2Ly-1) lattice that's structured like a 
        ``qtn.tensor_2d.TensorNetwork2D``, i.e. it has a single tensor per 
        site. 
               
        dummy_size: int, optional
            The size of the dummy indices we insert, both physical 
            and internal.
        remap_coordinate_tags: bool, optional
            Whether to relabel the coordinate tags to fit Johnny's
            convention i.e. (0,0) at bottom left rather than top.
        transpose_tensor_shapes: bool, optional
            Whether to transpose all the tensors to have dimensions
            ordered like 'urdl' (or 'urdlp' with physical index).
        relabel_physical_inds: bool, optional
            Whether to reindex the qubits to match the PEPS 
            coordinate-style indexing scheme.
        insert_physical_inds: bool, optional
            Whether to add physical indices to the new face sites.
        new_index_id: str, optional
            If new indices are added, this is the labeling scheme
            to be used. e.g. "k1,4"
        
        '''
        psi = self.copy()

        # 'rotate' qubits at nonempty faces
        # (unless already rotated)
        # 
        #      ─●───────●─      ─●───●───●──
        #       │ \   / │        │   │   │ 
        #       │   ●   │  ==>   ●───●───● 
        #       │ /   \ │        │   │   │ 
        #      ─●───────●─      ─●───●───●──
        #       │       │        │       │  
        # 
        already_rotated = psi.check_if_bmps_setup()
        if not already_rotated:
            psi.setup_bmps_contraction_()
        
        # now fill in all the empty faces with identities

        for x,y in psi.gen_face_coos(including_empty=True):    
            
            #skip the non-empty faces 
            coo_tag = psi.grid_coo_tag(x, y)
            if coo_tag in psi.tags:
                continue
                
            # coos (up, right, down, left) of the face site
            coos_around = ((x-1, y),
                           (x, y+1),
                           (x+1, y),
                           (x, y-1))
            
            # tags around site, e.g. ('S0,1', 'S1,2', ...)
            tags_around = tuple(starmap(psi.grid_coo_tag, coos_around))
            
            new_tensor_tags = (coo_tag, 
                            psi.row_tag(x),
                            psi.col_tag(y),
                            'AUX', 'FACE')
        
            # connect `up` and `down` with a new identity tensor
            # 
            # x-1   ●───●───●        ●───●───●
            #       │       │        │   │   │ 
            # x     ●       ●  ==>   ●   i   ● 
            #       │       │        │   │   │ 
            # x+1   ●───●───●        ●───●───● 
            #
            tag_up, tag_down = (tags_around[i] for i in (0,2))
            qtn.new_bond(T1=psi[tag_up], T2=psi[tag_down], size=dummy_size)

            psi.insert_identity_between_(
                where1=tag_up, 
                where2=tag_down, 
                add_tags=new_tensor_tags)
            
            # connect new identity tensor to the sides
            #    ●───●───●        ●───●───●
            #    │   │   │        │   │   │ 
            #    ●   i   ●  ==>   ●───i───● 
            #    │   │   │        │   │   │ 
            #    ●───●───●        ●───●───● 
            #
            tag_right, tag_left = (tags_around[i] for i in (1,3))
            qtn.new_bond(T1=psi[new_tensor_tags], T2=psi[tag_right], size=dummy_size)
            qtn.new_bond(T1=psi[new_tensor_tags], T2=psi[tag_left], size=dummy_size)

            if insert_physical_inds:
                # add a dummy physical index of `dummy_size`
                # 
                #    ●───●───●        ●───●───●
                #    │   │   │        │   │/  │ 
                #    ●───i───●  ==>   ●───i───● 
                #    │   │   │        │   │   │ 
                #    ●───●───●        ●───●───● 
                #
                new_index_name = new_index_id.format(x, y)
                psi[new_tensor_tags].new_ind(name=new_index_name, size=dummy_size)

            
        if remap_coordinate_tags:
            psi.remap_coordinates_()
        
        if relabel_physical_inds:
            psi.relabel_qubit_indices_()
        
        if transpose_tensor_shapes:
            psi.transpose_tensors_to_shape(shape='urdl')
            
        return psi


    def remap_coordinates(self, new_coo_tag=None, inplace=False):
        '''Retag the coordinate tags of every tensor to match Johnny's
        convention of grid.

        new_coo_tag: string, optional
            New format for the tag id, if desired e.g. "I{},{}"
            If unspecified, will use self.grid_tag_id
        
        inplace: bool, optional
            Whether to retag this TN in place
        '''
        psi = self if inplace else self.copy()

        # Make sure the "ROWX" and "COLY" tags are in place
        psi._add_row_col_tags()

        # Choose current row tag "S{},{}" by default
        if new_coo_tag is None:
            new_coo_tag = psi.grid_tag_id

        retag_map = dict()
        for x_old, y in psi.gen_supergrid_coos():

            old_tag_xy = psi.grid_coo_tag(x_old, y)

            # skip empty sites
            if old_tag_xy not in psi.tags:
                continue
            
            # Relabel the x-coordinates to run "upwards"
            # 
            #  0    ●───●───●─        ●───●───●─  L-1
            #       │   │   │         │   │   │ 
            #  1    ●───●───●─  ==>   ●───●───●─  L-2
            #       │   │   │         │   │   │ 
            # ...   :   :   :         :   :   :   ...
            #       │   │   │         │   │   │   
            # L-1   ●───●───●─        ●───●───●─   0
            # 
            x_new = psi.grid_Lx - 1 - x_old

            #"S{xold},{y}" --> "S{xnew}{y}"

            retag_map.update({old_tag_xy: new_coo_tag.format(x_new, y),
                         psi.row_tag(x_old): psi.row_tag(x_new)})


        #Retag this tn, and reset internal attribute!
        psi.retag_(retag_map)
        psi._grid_tag_id = new_coo_tag
    
        return psi

    remap_coordinates_ = functools.partialmethod(remap_coordinates,
                                                inplace=True)

    
    def relabel_qubit_indices(self, inplace=False, new_index_id='k{},{}'):
        '''For the qubit indices like 'q0', 'q1', ..., etc, rename
        the indices as 'k{x},{y}'
        '''
        
        psi = self if inplace else self.copy()
        old_index_id = psi.phys_ind_id

        # use dict mapping ints to coordinates, 
        # e.g. {0: (0,0), 1: (0,2), ...}
        old_qubit_coos = psi.qubit_to_coo_map()
        reindex_map = dict()

        for q, (x_old, y) in old_qubit_coos.items():
            #switch to Johnny's convention
            x_new = psi.grid_Lx - 1 - x_old
            reindex_map.update({old_index_id.format(q): 
                                new_index_id.format(x_new, y)})
        
        psi.reindex_(reindex_map)
        return psi

    relabel_qubit_indices_ = functools.partialmethod(relabel_qubit_indices, 
                                                    inplace=True)

    def transpose_tensors_to_shape(self, shape='urdl'):
        '''(Inplace) transpose the indices of every tensor in this tn
        to match the order ``shape``. Automatically puts the physical
        index (if it exists) as the last dimension, e.g. 
        
        "urdl" --> "urdlp" for tensors with physical index 'p'.

                  u                     u p
                  │                     │/       (if there's a physical ind)
             l ───●─── r    or     l ───●─── r    
                  │                     │
                  d                     d
        
        NOTE: (>>> ?) Assumes convention with (0,0)-origin at bottom left.

        NOTE: Only works for square 2D lattices! 
        '''
        #iterate over every occupied lattice site
        for (x,y), tag_xy in self.gen_occupied_grid_tags(with_coo=True):
            
            array_order = shape

            if x == 0:
                array_order = array_order.replace('d', '')
            if x == self.grid_Lx - 1:
                array_order = array_order.replace('u', '')
            if y == 0:
                array_order = array_order.replace('l', '')
            if y == self.grid_Ly - 1:
                array_order = array_order.replace('r', '')

            tensor_xy, = [self[tag_xy]]
            if len(tensor_xy.shape) == len(array_order) + 1:
                has_physical_index = True

            elif len(tensor_xy.shape) == len(array_order):
                has_physical_index = False            
            
            else:
                raise ValueError(f"The tensor at ({x}, {y}) has weird indices")

            
            #get coordinate tags for sites around this tensor.
            coos_around = ((x-1, y),
                           (x, y+1),
                           (x+1, y),
                           (x, y-1))
            
            tags_around = tuple(starmap(self.grid_coo_tag, coos_around))

            dir_to_tag = dict(zip(('d','r','u','l'), tags_around))

            dir_to_bond = {}
            for direction in array_order:                
                #throws an error if there's more than 1 shared index
                bond = self.bond(where1=tag_xy, where2 = dir_to_tag[direction])
                dir_to_bond[direction] = bond

            #indices ordered according to original ``shape```
            ordered_bonds = [dir_to_bond[x] for x in array_order]

            #put physical index last
            if has_physical_index:
                phys_index, = set(tensor_xy.inds) - set(ordered_bonds)
                ordered_bonds.append(phys_index)
            
            tensor_xy.transpose_(*ordered_bonds)
        

####################################################

class ePEPS(#qtn.TensorNetwork2DFlat,
            qtn.TensorNetwork2D,
            qtn.TensorNetwork):
    
    '''
    Notice that we are going from the convention 
    
       {site_tag_id='Q{}',
        grid_tag_id='S{},{}',
        aux_tag_id='IX{}Y{}',
        phys_ind_id='q{}'} 
        |
        |
        |
        V  
       {site_tag_id='S{},{}'}
    
    '''
    _EXTRA_PROPS = (
        '_Lx',
        '_Ly',
        '_site_tag_id',
        '_site_ind_id',
        '_row_tag_id',
        '_col_tag_id',
        # '_phys_dim',
        # '_grid_tag_id',
        # '_aux_tag_id', ... aux_tag_id='IX{}Y{}',
        # '_phys_ind_id',
    )
        # _EXTRA_PROPS = (
        # '_site_tag_id',
        # '_row_tag_id',
        # '_col_tag_id',
        # '_Lx',
        # '_Ly',
        # '_site_ind_id',
    # )

            
    def __init__(self, tn, *, 
            Lx=None, Ly=None, 
            site_tag_id='S{},{}', 
            site_ind_id='k{},{}',
            row_tag_id='ROW{}',
            col_tag_id='COL{}',
            **tn_opts):
        
        if isinstance(tn, ePEPS):
            super().__init__(tn)
            return

        # try to infer lattice shape from given tn
        self._Lx = Lx
        self._Ly = Ly

        self._site_tag_id = site_tag_id
        self._site_ind_id = site_ind_id
        self._row_tag_id = row_tag_id
        self._col_tag_id = col_tag_id

        # self._phys_dim = phys_dim
        # self._phys_ind_id = phys_ind_id
        # self._grid_tag_id = grid_tag_id

        
        super().__init__(tn, **tn_opts)


#************* Hamiltonian Classes *****************#

class MasterHam():
    '''Commodity class to combine a simulator Ham `Hsim`
    and a stabilizer pseudo-Ham `Hstab` to generate each of 
    their gates in order.

    Attributes:
    ----------
    `gen_ham_terms()`: generate Hsim terms followed by Hstab terms

    `gen_trotter_gates(tau)`: trotter gates for Hsim followed by Hstab
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
    '''Pseudo-Hamiltonian of stabilizers,
        
            H_stab = multiplier * (S1 + S2 + ... + Sk), 
        
        i.e. sum of all the stabilizers multiplied by ``multiplier``.

        Stores 8-site gates corresponding to the loop
        stabilizer operators, intending to be added to a simulator
        Hamiltonian `H_sim` for TEBD.
        '''

    def __init__(self, Lx, Ly, multiplier = -1.0):
        
        
        self.qlattice = dense_qubits.QubitLattice(Lx, Ly, local_dim=0)

        self.multiplier = multiplier
        
        #map coos to `loopStabOperator` objects
        coo_stab_map = self.qlattice.make_coo_stabilizer_map()

        self._stab_gates = self.make_stab_gate_map(coo_stab_map, store='gate')
        self._exp_stab_gates = dict()
        self._stab_lists = self.make_stab_gate_map(coo_stab_map, store='tuple')



    def make_stab_gate_map(self, coo_stab_map, store='gate'):
        '''TODO: NOW MAPS coos to (where, gate) tuples.

        Return
        -------
        `gate_map`: dict[tuple : (tuple, qarray)] 
            Maps coordinates (x,y) in the *face* array (empty 
            faces!) to pairs (where, gate) that specify the 
            stabilizer gate and the sites to be acted on.

        Param:
        ------
        coo_stab_map: dict[tuple : dict]
            Maps coordinates (x,y) in the face array of the lattice
            to `loop_stab` dictionaries of the form
            {'inds' : (indices),   'opstring' : (string)}
        
        store: 'gate' or 'tuple', optional
            Whether to store a 'dense' 2**8 x 2**8 array or a tuple
            of 8 (ordered) 2 x 2 arrays
        '''
        gate_map = dict()

        for coo, loop_stab in coo_stab_map.items():
            #tuple e.g. (1,2,4,5,6)
            qubits = tuple(loop_stab['inds'])

            #string e.g. 'ZZZZX'
            opstring = loop_stab['opstring']
            
            if store == 'gate':            
                #qarray
                gate = qu.kron(*(qu.pauli(Q) for Q in opstring))
                gate *= self.multiplier
                gate_map[coo] = (qubits, gate)
            
            elif store == 'tuple':

                signs = [self.multiplier] + [1.0] * (len(opstring) - 1)
                gates = tuple(signs[k] * qu.pauli(Q) for k, Q in enumerate(opstring))
                gate_map[coo] = (qubits, gates)
            
        
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

    def gen_ham_stabilizer_lists(self):
        '''Generate ``(where, gates)`` pairs for each stabilizer term.
        where: tuple[int]
            The qubits to be acted on
        gates: tuple[array]
            The one-site qubit gates (2x2 arrays)
        '''
        for where, gatelist in self._stab_lists.values():
            yield (where, gatelist)
    
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
    
    def __init__(self, Lx, Ly, phys_dim, ham_terms):
        
        self._Lx = Lx
        self._Ly = Ly
        self._phys_dim = phys_dim

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

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly

    
    def ham_params(self):
        '''Relevant parameters. Override for
         each 'daughter' Hamiltonian.
        '''
        pass


    def gen_ham_terms(self):
        pass
    
    
    def gen_horizontal_ham_terms(self):
        pass
    
    def gen_vertical_ham_terms(self):
        pass


    def gen_trotter_gates(self, tau):
        pass


## ******************* ##
# Subclass Hamiltonians
## ******************* ##

class SpinlessSimHam(SimulatorHam):
    '''Qubit Hamiltonian: spinless fermion Hubbard Ham,
    encoded as a qubit simulator Ham.

    H =   t  * hopping
        + V  * repulsion
        - mu * occupation
    '''

    def __init__(self, Lx, Ly, t=1.0, V=1.0, mu=0.5):
        '''
        Lx: number of (vertex) qubit rows
        Ly: number of vertex qubit columns
        
        t: hopping parameter
        V: nearest-neighbor repulsion
        mu: single-site chemical potential
        '''
        # Hubbard parameters
        self._t = t
        self._V = V
        self._mu = mu

        # to handle the fermion-to-qubit encoding & lattice geometry
        self.qlattice = dense_qubits.QubitLattice(Lx=Lx, Ly=Ly, local_dim=0)
        
        terms = self._make_ham_terms()

        super().__init__(Lx=Lx, Ly=Ly, phys_dim=0, ham_terms=terms)

    
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
        '''Generate ``(where, gate)`` pairs for every group 
        of qubits (i.e. every graph edge ``where``) to be acted
        on with a Ham term.
        '''
        for (i,j,f), gate in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, gate


    def gen_horizontal_ham_terms(self):
        '''Only those terms in the Hamiltonian acting on horizontal
        graph edges, i.e. the ``where`` qubit sites must correspond
        to a *horizontal* (vertex, vertex, face or None).
        '''

        for (i, j, f) in self.get_edges(which='horizontal'):
            gate = self.get_term_at(i, j, f)
            where = (i, j) if f is None else (i, j, f)
            yield where, gate
    
    def gen_vertical_ham_terms(self):
    
        for (i, j, f) in self.get_edges(which='vertical'):
            gate = self.get_term_at(i, j, f)
            where = (i, j) if f is None else (i, j, f)
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


    def _make_ham_terms(self):
        '''Get all terms in Ham as two/three-site gates, 
        in a dict() mapping edges to qarrays.
        
        ``terms``:  dict[edge (i,j,f) : gate [qarray] ]

        If `f` is None, the corresponding gate will be two-site 
        (vertices only). Otherwise, gate acts on three sites.
        '''
        t, V, mu = self.ham_params()
        qlattice = self.qlattice

        terms = dict()

        #vertical edges
        for direction, sign in (('down', 1), ('up', -1)):

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

            #should appear in at least two edge terms!
            if not (num_edges > 1 or qlattice.num_faces == 0):
                raise ValueError("Something's wrong")


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





def compare_bmps_norm_error():
    import matplotlib.pyplot as plt
    
    init_bdim = 6
    
    ntest = QubitEncodeNet.random_flat(Lx=3, Ly=3, bond_dim=init_bdim)
    ntest.rotate_face_qubit_bonds_(1,1)
    ntest.rotate_face_qubit_bonds_(3,3)
    ntest.fill_cols_with_identities_()
    ntest.fill_rows_with_identities_()
    
    nex = ntest ^ all
    
    chis = np.linspace(4, 40, num=15, endpoint=True)

    directions = ('t', 'b', 'r', 'l')
    
    norms = {k: [] for k in directions}

    for chi in chis:
        for k in directions:
            nk = ntest.contract_boundary(sequence=k, max_bond=chi)
            norms[k].append(nk)        
    
    plotnorms = {k: np.abs(np.array(norms[k]) - nex) for k in directions}

    for k in directions:
        # plt.plot(chis, plnorms[k], label=k)
        plt.semilogy(chis, plotnorms[k], label=k, marker='.')
    plt.legend()
    plt.title(f'Initial bond $D={init_bdim}$')
    plt.xlabel(r"Max bond $D'$")
    plt.ylabel(r'Norm err')
    plt.show()

    # for k in directions:
    #     plt.plot(chis, plotnorms[k], label=k, marker='.')
    #     # plt.semilogy(chis, plotnorms[k], label=k, marker='.')
    # plt.legend()
    # plt.title(f'Initial bond $D={init_bdim}$')
    # plt.xlabel(r"Max bond $D'$")
    # plt.ylabel(r'Norm err')
    # plt.show()





def bmps_norm_test():
    net = beeky.QubitEncodeVector.rand(Lx=3,Ly=3, add_tags=['KET'])
    norm = net.make_norm()
    norm.flatten_()
    norm.rotate_face_qubit_bonds_(0,0)
    norm.rotate_face_qubit_bonds_(1,1)

    nex = norm ^ all
    

def main_debug():
    nflat = QubitEncodeNet.random_flat(3,3)
    nflat.rotate_face_qubit_bonds_(0,0)
    nflat.rotate_face_qubit_bonds_(1,1)
    nflat.fill_cols_with_identities_()
    nflat.fill_rows_with_identities_()
    row_envs = nflat.compute_row_environments()

    row2 = nflat.row_environment_sandwich(x0=2, x_bsz=2, row_envs=row_envs)

    row2.compute_col_environments()



if __name__ == '__main__':
    
    Hstab = HamStab(Lx=3, Ly=3)


    # qvec = QubitEncodeVector.rand(3, 3, bond_dim=4)
    # norm = qvec.make_norm().setup_bmps_contraction_(layer_tags=('BRA','KET'))
    # phi = norm.flatten().contract_boundary_from_left(xrange=(0,4), yrange=(0,2), max_bond=5)
    # phi._compress_supergrid_column(y=2, sweep='down', xrange=(0,4), max_bond=5)
    
    # HubHam = SpinlessSimHam(Lx=3, Ly=3)
    # qubit_terms = dict(HubHam.gen_ham_terms())

    # qvec.compute_local_expectation(qubit_terms, return_all=True)