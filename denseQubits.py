import numpy as np
from itertools import product
import quimb as qu
from operator import add
import functools
import itertools
import operator
import stabilizers

class QubitLattice():
    def __init__(self, Lx, Ly, local_dim):

        if (Lx-1)*(Ly-1) % 2 == 1:
            raise NotImplementedError('Need even number of faces!')

        self._Lx = Lx
        self._Ly = Ly
        self._lat_shape = (Lx, Ly)

        #generate ordering for qbit lattice
        verts, faces = gen_lattice_sites(Lx,Ly)

        #vertex/face indices in np.ndarrays
        self._verts = verts
        self._faces = faces

        #map strings {'u','r','d', ...} to edge 3-tuples
        self._edge_map = make_edge_map(verts,faces)
        
        #map face coos to `loopStabOperator` objects
        # self._coo_stab_map = self.make_coo_stabilizer_map()

        #number lattice sites (IGNORES faces w/out qubits)
        self._Nsites = verts.size + faces[faces!=None].size
        self._sim_dims = [local_dim]*(self._Nsites)
        # self._local_dim = local_dim

        #number of vertex qubits, i.e. fermionic sites
        self._Nfermi = verts.size


        #TODO: not true for odd num. faces
        #codespace dimensions = dim(Fock)
        self._encoded_dims = [local_dim]*(self._Nfermi)

        self._local_dim = local_dim


    def active_sites_from_edge(self, edge):
        '''Return the valid sites in `edge`, i.e. delete
        the face-site if it's None.
        '''
        if len(edge) == 2:
            return edge
        
        elif len(edge) == 3:
            i,j,f = edge
            active_sites = (i,j) if f is None else edge
            return active_sites
        
        else:
            raise ValueError('Edge can only have 2-3 sites!')

    
    # def edge_from_verts(self, va, vb):
    #     '''Return the edge (i,j,f), appropriately
    #     ordered, that has vertices `va, vb` as nodes.
    #     '''




    def get_edges(self, which):
        '''TODO IMPLEMENT
        '''
        if which in ['horizontal', 'right+left']:
            return self._edge_map['r'] + self._edge_map['l']
        

        if which == 'all':
            return (self._edge_map['r']+
                    self._edge_map['l']+
                    self._edge_map['u']+
                    self._edge_map['d'] )


        key =  {'d' : 'd',
                'down':'d',
                'u' : 'u',
                'up': 'u',
                'left':'l',
                'l' : 'l',
                'right':'r',
                'r':'r',
                'he':'he',
                'ho':'ho',
                've':'ve',
                'vo':'vo'}[which]
        
        return self._edge_map[key]
    
    
    def vertex_sites(self):
        '''
        Indices for vertex qubits,
        equivalent to range(Lx*Ly)
        '''
        return list(self._verts.flatten())

    def face_sites(self):
        '''
        Indices of face-qubits 
        '''
        F = self._faces.flatten()
        return list(F[F!=None])
    
    def all_sites(self):
        '''
        All qubit indices (vertex and face)
        '''
        # return self.vertex_sites() + self.face_sites()
        return list(range(self._Nsites))

    @property
    def lattice_shape(self):
        return (self._Lx, self._Ly)


    @property
    def vert_array(self):
        '''ndarray of vertex site numbers
        '''
        return self._verts.copy()
    

    @property
    def face_array(self):
        '''ndarray of face site numbers
        '''
        return self._faces.copy()


    def num_verts(self):
        return self._verts.size
    

    def num_faces(self):
        return self._faces[self._faces!=None].size


    def num_sites(self):
        return self.num_faces()+self.num_verts()
    

    @property
    def codespace_dims(self):
        return self._encoded_dims
    

    @property
    def simspace_dims(self):
        '''
        '''
        return self._sim_dims


    @property
    def local_site_dim(self):
        return self._local_dim


    def gen_vert_coos(self):
        '''Generate *vertex* coordinates (i,j)
        '''
        return product( range(self._Lx),
                        range(self._Ly))



    def make_coo_stabilizer_map(self):
        Lx, Ly = self._Lx, self._Ly
        faces = self._faces
        stab_map = dict()

        for x,y in product(range(Lx-1),range(Ly-1)):
            if faces[x,y] is None:
                stab_map[(x,y)] = self.loop_stabilizer_data(x,y)
        
        return stab_map

    
    def loop_stabilizer_data(self, i, j, verbose=False):
        '''
        Returns `loop_op_data`, corresponding to face
        at location (i,j) in face array, which maps
        
        {'inds' : (indices),
         'opstring' : (string)}

              u
           1-----2
         l |(i,j)| r   ==>  S = Z1*Z2*Z3*Z4*Yu*Xr*Yd*Xl
           4-----3
              d

        Note clockwise = ccwise, i.e.
        (E12)(E23)(E34)(E41) = (E14)(E43)(E32)(E21)
        
        TODO: test!
        '''

        # X, Y, Z = (qu.pauli(mu) for mu in ['x','y','z'])
        
        verts = self.vert_array
        faces = self.face_array
        
        assert faces[i,j] is None

        #corner vertex sites, 1=upper-left --> clockwise
        v1 = verts[i, j]
        v2 = verts[i, j+1]
        v3 = verts[i+1, j+1]
        v4 = verts[i+1, j]

        
        u_face = find_face_up_down(row=i, cols=(j,j+1), Vs=verts, Fs=faces)
        d_face = find_face_up_down(row=i+1, cols=(j,j+1), Vs=verts, Fs=faces)
        l_face = find_face_right_left(rows=(i,i+1), col=j, Vs=verts, Fs=faces)
        r_face = find_face_right_left(rows=(i,i+1), col=j+1, Vs=verts, Fs=faces)


        if verbose:
            print('Stabilizer:')
            print('{}-----{}\n|     |\n{}-----{}'.format(v1,v2,v4,v3))
            print(f'u: {u_face}, d: {d_face}, l: {l_face}, r: {r_face}', end='\n\n')


        # vert_ops = [Z, Z, Z, Z]
        vert_inds = [v1, v2, v3, v4]

        face_inds, face_ops = [], ''
        if u_face is not None:
            face_inds.append(u_face)
            face_ops += 'Y'
        
        if d_face is not None:
            face_inds.append(d_face)
            face_ops += 'Y'

        if r_face is not None:
            face_inds.append(r_face)
            face_ops += 'X'
        
        if l_face is not None:
            face_inds.append(l_face)
            face_ops += 'X'
        
        assert len(face_ops) >= 1
        
        # ops = vert_ops + face_ops
        # inds = vert_inds+face_inds
        # loop_op = qu.ikron(ops=ops, dims=self._sim_dims, inds=inds)
        # if qu.isreal(loop_op): 
        #     loop_op = loop_op.real

        # face_ops = ''.join(face_ops)
        
        # loop_op = stabilizers.loopStabOperator(vert_inds=vert_inds,
        #                                     face_inds=face_inds,
        #                                     face_op_str=face_ops)
        
        loop_op_data = { 'inds':  tuple(vert_inds + face_inds),
                         'opstring': 'ZZZZ' + face_ops}
        
        return loop_op_data


class SpinlessHub(QubitLattice):

    def __init__(self, Lx=2, Ly=3):
        '''
        Spinless fermion model is mapped to 
        a lattice (Lx, Ly) of qubits, following
        the low-weight fermionic encoding of
        Derby and Klassen (2020):

            arXiv:2003.06939 [quant-ph]
        '''
        #Simulator Hamiltonian in full qubit space
        self._HamSim = None

        #Codespace Hamiltonian
        self._HamCode = None
        self._eigens, self._eigstates = None, None
        
        super().__init__(Lx, Ly, local_dim=2)



    def H_nn_int(self, i, j):
        '''
        Nearest neighbor repulsive 
        interaction, for vertex qbits 
        i, j.

        Return (n_i)(n_j)
        '''
        return qu.ikron(ops=[self.number_op(), self.number_op()],
                        dims=self._sim_dims,
                        inds=[i,j])


    def H_hop(self, i, j, f, dir):
        '''
        TODO: pkron vs ikron performance?

        i, j: (directed) edge qubits indices
        f: face qubit index
        dir: {'vertical', 'horizontal'}

        Returns hopping term acting on qbits ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of is Xf, Yf or Identity depending on ijf
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        Of = {'vertical': X, 'horizontal':Y} [dir]
        
        #if no face qbit
        if f==None:  
            # print('{}--{}-->{}   (None)'.format(i,dir[0],j))       
            # XXO=qu.ikron(ops=[X,X], dims=self._sim_dims, inds=[i,j])
            # YYO=qu.ikron(ops=[Y,Y], dims=self._sim_dims, inds=[i,j])
            XXO=qu.pkron(op=X&X, dims=self._sim_dims, inds=[i,j])
            YYO=qu.pkron(op=Y&Y, dims=self._sim_dims, inds=[i,j])
        
        #if there's a face qubit: Of acts on index f
        else:
            # print('{}--{}-->{},  face {}'.format(i,dir[0],j,f))
            # XXO=qu.ikron(ops=[X,X,Of], dims=self._sim_dims, inds=[i,j,f])
            # YYO=qu.ikron(ops=[Y,Y,Of], dims=self._sim_dims, inds=[i,j,f])
            XXO = qu.pkron(op=X&X&Of, dims=self._sim_dims, inds=(i,j,f))
            YYO = qu.pkron(op=Y&Y&Of, dims=self._sim_dims, inds=(i,j,f))

        return 0.5*(XXO+YYO) 


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
    

    def make_simulator_ham(self, t=1.0, V=0.0, mu=0.0):
        '''
        Compute the qubit simulator Hamiltonian (H_sim)
        corresponding to the spinless Fermi-Hubbard Hamiltonian
        
        H =   t <pairwise hopping>
            + V <nearest-neighbor repulsion> 
            - mu <onsite occupation>
        
        Store qubit Hamiltonian in --> self._HamSim
        '''
        self._coupling_const = (t, V, mu)

        def hops(): #fermion hopping terms
            
            for (i,j,f) in self.get_edges('right'):
                yield t * self.H_hop(i, j, f, 'horizontal')
            
            for (i,j,f) in self.get_edges('left'):
                yield t * self.H_hop(i, j, f, 'horizontal')

            for (i,j,f) in self.get_edges('up'):
                #minus sign! (See Derby & Klassen)
                yield -t * self.H_hop(i, j, f, 'vertical')
            
            for (i,j,f) in self.get_edges('down'):
                yield  t * self.H_hop(i, j, f, 'vertical')

        def interactions(): #pairwise neighbor repulsion terms
           
            # allEdges = self._edgesR + self._edgesU + self._edgesL + self._edgesD
            
            for (i,j,f) in self.get_edges('all'): 
                #doesn't require face qbit!
                yield V * self.H_nn_int(i,j)

        def occs(): #occupation at each vertex
            #only counts vertex qbits, ignores faces!
            for i in self.vertex_sites():
                yield -mu * qu.ikron(self.number_op(), 
                                    dims=self._sim_dims, 
                                    inds=i)

        hopping_terms, int_terms, occ_terms = 0, 0, 0

        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if V != 0.0: int_terms = functools.reduce(operator.add, interactions())
        if mu != 0.0: occ_terms = functools.reduce(operator.add, occs())

        H = hopping_terms + int_terms + occ_terms

        if qu.isreal(H): 
            H = H.real    

        self._HamSim = H
        


    def one_qubit_dm(self, qstate, i):
        '''
        Returns subsystem density
        matrix for qubit i by tracing out
        all other qubits in `qstate`.

        `qstate` needs to be in *full* qubit space!
        '''
        assert qstate.size == qu.prod(self._sim_dims)
        return qu.ptr(qstate, dims=self._sim_dims, keep=i)


    def lift_cstate(self, cstate):
        '''
        Lift `cstate` from codespace to full qubit space

        Args:
            cstate [qarray, dim 64]: codespace state,
            vector written in stabilizer eigenbasis

        Returns:
            dim-128 vector in standard qubit basis
        '''
        return self._Uplus @ cstate
        

    def solve_ED(self):
        '''
        Solves full spectrum of simulator Hamiltonian, 
        stores eigensystem. 
        
        Assumes self._HamSim exists.
        '''
        self._eigens, self._eigstates = qu.eigh(self._HamCode)
    

    def eigspectrum(self):
        '''
        Returns energy eigenvalues and eigenstates of
        self._HamCode, obtained through ED
        '''        
        if (self._eigens is None) or (self._eigstates is None):
            self.solve_ED() 

        return self._eigens.copy(), self._eigstates.copy()


    def site_number_ops(self, sites, fermi=False):
        '''
        TODO: remove fermi?

        Returns: list of qarrays, local number 
        operators acting on specified sites.
        
        param `sites`: list of ints, indices of sites
        
        param `fermi`: bool, indicates whether we are acting
        in the large qubit space basis (False) or in the restricted
        stabilizer eigenbasis (True)
        '''
        ds = {False : self._sim_dims,
              True : self._fermi_dims} [fermi]

        return [qu.ikron(self.number_op(), ds, [site]) 
                        for site in sites]#.reshape(self._lat_shape)


    def state_local_occs(self, qstate=None, k=0, faces=False):
        '''
        TODO: fix face shapes?

        Expectation values <k|local fermi occupation|k>
        in the kth excited eigenstate, at vertex
        sites (and faces if specified).

        param qstate: vector in full qubit basis
        
        param k: if `state` isn't specified, will compute for
        the kth excited energy eigenstate (defaults to ground state)

        param faces: if True, return occupations at face qubits as well

        Returns: local occupations `nocc_v`, (`nocc_f`)

            nocc_v (ndarray, shape (Lx,Ly))
            nocc_f (ndarray, shape (Lx-1,Ly-1))
        '''

        if qstate is None:
            qstate = self._eigstates[:,k] 
        
        #local number ops acting on each site j
        # Nj = self.site_number_ops(sites=self.all_sites() , fermi=False)
        Nj = [qu.ikron(self.number_op(), self._sim_dims, [site]) 
                        for site in self.all_sites()]

        #expectation <N> for each vertex
        nocc_v = np.array([qu.expec(Nj[v], qstate)
                        for v in self.vertex_sites()]).reshape(self._lat_shape)

        if not faces:
            return nocc_v

        #expectation <Nj> for each face
        nocc_f = np.array([qu.expec(Nj[f], qstate) for f in self.face_sites()])
        
        return nocc_v, nocc_f
            

    def make_stabilizer(self):
        '''
        DEPREC -- DON'T USE 
        
        TODO: 
        * add a projector onto codespace (see method 2)
        * generalize indices, rotation, reshape, etc
        * always need to round? check always real
        '''
        
        X, Z = (qu.pauli(mu) for mu in ['x','z'])
        
        ## TODO: make general
        # ops = [Z,Z,Z,Z,X]
        # inds = [1,2,4,5,6]
        # stabilizer = qu.ikron(ops=ops, dims=self._sim_dims, inds=inds)
        
        _, Ux = qu.eigh(qu.pauli('x')) 
        
        stab_data = self.loop_stabilizer_data(0,1)
        oplist = [qu.pauli(Q) for Q in stab_data['opstring']]
        
        stabilizer = qu.ikron(  ops=oplist,
                                dims=self._sim_dims,
                                inds=stab_data['inds'] )
        #TODO: change to general rather than inds=6, Ux
        U = qu.ikron(Ux.copy(), dims=self._sim_dims, inds=[6])
        
        #TODO: can this be done without rounding()/real part?
        Stilde = (U.H @ stabilizer @ U).real.round()
        
        assert is_diagonal(Stilde)

        U_plus = U[:, np.where(np.diag(Stilde)==1.0)[0]]
        
        HamCode = U_plus.H @ self.ham_sim() @ U_plus

        self._stabilizer = stabilizer
        self._Uplus = U_plus #+1 eigenstates written in full qubit basis
        self._HamCode = HamCode # in +1 stabilizer eigenbasis

    def stabilizer_at_face(self, fi, fj):
        stab_data = self.loop_stabilizer_data(fi,fj)
        oplist = [qu.pauli(Q) for Q in stab_data['opstring']]
        
        return qu.ikron(ops=oplist,
                        dims=self._sim_dims,
                        inds=stab_data['inds'] )
        
    def operator_to_codespace(self, operator):
        '''
        Return operator in codespace basis,
        O_ij = <vi|O|vj>,
        where v's are +1 stabilizer eigenstates.
        '''
        U_plus = self._Uplus#(128,64)
        return U_plus.H @ op @ U_plus #(64,64)


    def stabilizer(self):
        return self._stabilizer.copy()


    def ham_sim(self):
        '''
        Simulator Hamiltonian acting
        on full qubit space.
        '''
        return self._HamSim.copy()


    def ham_code(self):
        '''
        Projected Hamiltonian acting only on
        +1 stabilizer eigenspace.
        '''       
        return self._HamCode.copy()


    # def t_make_stabilizers(self):
    #     '''
    #     TODO: test! Could be completely wrong

    #     To be used in general case when lattice has 
    #     multiple stabilizer operators.
    #     '''
        
    #     self._stabilizers = {}
        
    #     for (i,j) in np.argwhere(self._F_ind==None):

    #         loop_op_ij = self.loop_stabilizer(i,j)
            
    #         self._stabilizers[(i,j)] = loop_op_ij




    # # TODO: delete?
    # def projected_ham_2(self):
    #     '''
    #     Alternative projected Hamiltonian.

    #     Obtained by numerically diagonalizing stabilizer
    #     rather than "manually", so pHam should be equivalent 
    #     to regular HamCode under a basis rotation.
    #     '''

    #     evals, V = qu.eigh(self._stabilizer)
        
    #     assert np.array_equal(evals,np.array([-1.0]*64 + [1.0]*64))

    #     if qu.isreal(V): V=V.real

    #     projector = V @ np.diag( [0]*64 + [1]*64 ) @ V.H #projector onto +1 eigenspace

    #     self._stab_projector = projector

    #     Vp = V[:, np.where(evals==1.0)].reshape(128,64)
    #     #shape=(128,64)

    #     pHam = Vp.H @ self._HamSim @ Vp 
    #     return pHam
        #(64,64)
        
    #TODO: delete?
    def projected_ham_3(self):
        '''
        Should be equivalent to self.ham_code()
        '''
        #X because stabilizer acts with X on 7th qubit
        _, Ux = qu.eigh(qu.pauli('x')) 
        
        U = qu.ikron(Ux.copy(), dims=self._sim_dims, inds=[6])
        
        Stilde = (U.H @ self.stabilizer() @ U).real.round()
        #Stilde should be diagonal!

        U_plus = U[:, np.where(np.diag(Stilde)==1.0)[0]]
        
        HamProj = U_plus.H @ self.ham_sim() @ U_plus
        
        return HamProj #in +1 stabilizer eigenbasis
        




### ************************ ###
### Spin-1/2 Hubbard (dense) ###
### ************************ ###




class SpinhalfHub(QubitLattice):

    def __init__(self, Lx=2, Ly=3):
        '''
        Each site has local physical dim = 4:
        effectively two qubits per site, for spin up/down 
        sectors of fermionic Fock space.

        N sites => N qdits (2N qbits) counting both vertex and
        face qbits, residing in pairs on the N lattice
        sites.

        Lattice indices: i = {0, 1, ... N-1}

        '''
        #Simulator Hamiltonian in full qubit space
        self._HamSim = None


        #Codespace Hamiltonian
        self._HamCode = None
        self._eigens, self._eigstates = None, None

        super().__init__(Lx, Ly, local_dim=4)



    def H_onsite(self, i):
        '''
        Spin-spin repulsion at site i (should only
        be called for vertex sites, not face sites)

        (n^up_i)(n^down_i) --> (1-Vi^up)(1-Vi^down)/4

        Projects onto [down (x) down] for qbits at index i
        '''
        return qu.ikron(ops=[self.q_number_op() & self.q_number_op()],
                        dims=self._sim_dims,
                        inds=[i])


    def H_hop(self, i, j, f, dir, spin):
        '''
        Returns "hopping" term acting on sites i,j,f:
            
            (XiXjOf + YiYjOf)/2
                
        on vertex qdits i,j, face qdit f, spin sector `spin`,
        where Of is {Xf, Yf, I} depending on edge.

        dir: {'vertical', 'horizontal'}
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

        X_sigma = {'u': X & I,
                   'd': I & X}[spin]

        Y_sigma = {'u': Y & I,
                   'd': I & Y}[spin]


        Of = {'vertical': X_sigma, 'horizontal':Y_sigma}[dir]


        #no associated face qbit
        if f==None:         
            XXO=qu.ikron(ops=[X_sigma,X_sigma], dims=self._sim_dims, inds=[i,j])
            YYO=qu.ikron(ops=[Y_sigma,Y_sigma], dims=self._sim_dims, inds=[i,j])
        
        #otherwise, let Of act on index f
        else:
            XXO=qu.ikron(ops=[X_sigma,X_sigma,Of], dims=self._sim_dims, inds=[i,j,f])
            YYO=qu.ikron(ops=[Y_sigma,Y_sigma,Of], dims=self._sim_dims, inds=[i,j,f])

        return 0.5*(XXO+YYO) 
    


    def q_number_op(self):
        '''
        Single-qubit operator,
        equiv. to fermionic number
        on a single spin sector.
        '''
        return qu.qu([[0, 0], [0, 1]])


    #TODO: delete?
    def spin_numberop(self, spin):
        '''
        TODO: why can't use ikron()?

        Fermionic number operator
        acting on `spin` sector.
        '''
        #which fermion-spin sector to act on?
        #i.e. act on qbit 0 or qbit 1?
        opmap = {0 : lambda: qu.qu([[0, 0], [0, 1]]) & qu.eye(2),

                 1 : lambda: qu.eye(2) & qu.qu([[0, 0], [0, 1]])}
        
        op = opmap[spin]() 

        qu.core.make_immutable(op)
        return op
        

        
    def make_simulator_ham(self, t, U):
        '''Makes spin-1/2 SIMULATOR Hamiltonian, stores in self._HamSim.

        Hubbard H = t <pairwise hopping>
                  + U <on-site spin-spin repulsion> 
        '''

        def hops(): #edgewise hopping terms 

            for (i,j,f) in self.get_edges('horizontal'):
                yield t * self.H_hop(i, j, f, 'horizontal',spin=0)
                yield t * self.H_hop(i, j, f, 'horizontal',spin=1)

            for direction, sign in [('down',1), ('up',-1)]:
                for (i,j,f) in self.get_edges(direction):
                    yield sign * t * self.H_hop(i, j, f, 'vertical', spin=0)
                    yield sign * t * self.H_hop(i, j, f, 'vertical', spin=1)


        def onsites(): #onsite spin-spin interaction
            for i in self.vertex_sites():
                yield U * self.H_onsite(i)
        

        hopping_terms, onsite_terms = 0, 0

        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if U != 0.0: onsite_terms = functools.reduce(operator.add, onsites())
            
        H = hopping_terms + onsite_terms
        
        if qu.isreal(H):
            H = H.real
        
        self._HamSim = H


    def ham_sim(self):
        return self._HamSim.copy()


    def solveED(self):
        '''Diagonalize codespace Ham, 
        store eigensystem.
        '''
        self._eigens, self._eigstates = qu.eigh(self._HamCode)


    def eigspectrum(self):
        '''Returns energy eigenvalues and eigenstates of
        self._Ham, obtained through ED
        '''
        if None in [self._eigens, self._eigstates]:
            self.solveED()
        
        return np.copy(self._eigens), np.copy(self._eigstates)
    

### end class ###



def gen_lattice_sites(Lx, Ly):

    '''
    Generate sites for lattice of shape (Lx,Ly)

    Returns (Vs, Fs) tuple of arrays
    (Vertex indices, face indices)

    Qubit sites ordered thus (4x4 e.g.)

    0--<-1--<-2--<-3
    ^ 16 v    ^ 17 v
    4->--5->--6->--7
    ^    | 18 |    v
    8--<-9--<-10-<-11
    ^ 19 |    | 20 v
    12->-13->-14->-15

    Vs contains the vertex indices,
    (0, 1, ..., Lx*Ly -1)

    Vs.shape = (Lx,Ly)

    0----1----2----3
    |    |    |    |
    4----5----6----7
    |    |    |    |
    8----9----10---11
    |    |    |    |
    12---13---14---15

    and Fs contains face sites
    (Lx*Ly, Lx*Ly+1, ..., N-1)

     ---- ---- ----
    | 16 |    | 17 |
     ---- ---- ----
    |    | 18 |    |
     ---- ---- ----
    | 19 |    | 20 |
     ---- ---- ----

    where N is the total qbits.

    **Note that Fs only has indices for faces
    that contain a qbit, i.e. odd faces. 
    For even faces Fs stores `None`.
    '''
    Vs = np.arange(Lx*Ly).reshape(Lx,Ly)
    
    Fs = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    
    N_vert = Lx*Ly

    k = 0
    for i in range(Lx-1):
        for j in range(Ly-1):
            if i%2==j%2: 
                Fs[i,j] = k + N_vert
                k+=1
            else: pass
    
    return Vs, Fs


def make_right_edges(V_ind, F_ind):
    '''
    |  U? |
    i-->--j
    |  D? |

    For horizontally connected vertices i, j: 
    At most ONE of faces U (up) and D (down) 
    will have a qubit on it. 

    *If U has a qubit: (i, j, U) added to edgesR.
    *If D has a qubit: (i, j, D) added to edgesR.
    *If neither face exists/has a qubit: (i, j, None)

    '''
    Lx, Ly = V_ind.shape
    assert F_ind.shape==(Lx-1,Ly-1)

    edgesR=[] #rightward edges, (a, b, f(a,b))

    # for row in range(0,Lx,2):
    for row in range(1,Lx,2): 
        for col in range(Ly-1):
                i, j = V_ind[row,col], V_ind[row,col+1]
                f = find_face_up_down(row=row, cols=(col,col+1), Vs=V_ind, Fs=F_ind)
                edgesR.append((i, j, f))

    return edgesR

'''
TODO: COMMENT
'''
def make_left_edges(V_ind, F_ind):
    '''List of left-edges for given arrays
    of vertices and faces
    '''
    Lx, Ly = V_ind.shape
    assert F_ind.shape==(Lx-1,Ly-1)

    edgesL=[]

    # for row in range(1,Lx,2):
    for row in range(0,Lx,2):
        for col in range(Ly-1,0,-1):

            i,j = V_ind[row, col], V_ind[row, col-1]
            f = find_face_up_down(row=row, cols=(col-1,col), Vs=V_ind, Fs=F_ind)
            edgesL.append((i, j, f))

    return edgesL

'''
TODO: COMMENT
'''
def make_up_edges(V_ind, F_ind):
    '''List of up-edges for given arrays
    of vertices and faces
    '''
    Lx, Ly = V_ind.shape
    edgesU=[]

    for row in range(Lx-1,0,-1):
        for col in range(0, Ly, 2):
            i, j = V_ind[row, col], V_ind[row-1, col]
            f = find_face_right_left(rows=(row-1,row), col=col, Vs=V_ind, Fs=F_ind)
            edgesU.append((i,j,f))

    return edgesU

'''
TODO: COMMENT
'''
def make_down_edges(V_ind, F_ind):
    '''List of down-edges for given arrays
    of vertices and faces  
    '''
    Lx, Ly = V_ind.shape
    assert F_ind.shape == (Lx-1,Ly-1)
    
    edgesD=[]

    for row in range(0,Lx-1):
        for col in range(1, Ly, 2):
            i, j = V_ind[row, col], V_ind[row+1, col]
            f = find_face_right_left(rows=(row,row+1), col=col, Vs=V_ind, Fs=F_ind)
            edgesD.append((i,j,f))

    return edgesD


def inverse_coo_map(Vs):
    '''Given array/map of coordinates to vertices,
    return a map taking vertex number to (x,y)
    '''
    if not isinstance(Vs, dict):
        Vs = dict(np.ndenumerate(Vs))
    
    inv_map = {vert : coo for coo, vert in Vs.items()}
    return inv_map



def make_edge_map(Vs, Fs):
    '''Map to edges of the graph
    in various directions/groupings.
    
    Returns:
    edge_map: dict[string : list(tuple(int))]

        keys: {'u','d','r','l','he','ho','ve','vo'}
        
        values: lists of edges, i.e. lists of
        tuples (i,j,f) 
    '''
    Lx,Ly = Vs.shape
    
    edge_map = {}
    
    edge_map['r'] = make_right_edges(Vs, Fs)
    edge_map['l'] = make_left_edges(Vs, Fs)
    edge_map['u'] = make_up_edges(Vs, Fs)
    edge_map['d'] = make_down_edges(Vs, Fs)


    horizontals = edge_map['r'] + edge_map['l']
    verticals = edge_map['u'] + edge_map['d']

    inv_coo_map = inverse_coo_map(Vs)

    hor_even, hor_odd = [], []
    for (i,j,f) in horizontals:
        xi, yi = inv_coo_map[i]
        xj, yj = inv_coo_map[j]
        assert xi==xj and abs(yi-yj)==1

        if min([yi,yj])%2 == 0:
            hor_even.append(tuple([i,j,f]))
        else:
            hor_odd.append(tuple([i,j,f]))

    ver_even, ver_odd = [], []
    for (i,j,f) in verticals:
        xi, yi = inv_coo_map[i]
        xj, yj = inv_coo_map[j]
        assert yi==yj and abs(xi-xj)==1

        if min([xi,xj])%2 == 0:
            ver_even.append(tuple([i,j,f]))
        else:
            ver_odd.append(tuple([i,j,f]))
        
    edge_map['he'] = hor_even
    edge_map['ho'] = hor_odd
    edge_map['ve'] = ver_even
    edge_map['vo'] = ver_odd

    return edge_map


def find_face_up_down(row, cols, Vs, Fs):
    '''
    Get face-site that's above *or* below the edge
    corresponding to (x,y)->(x,y+1)

    row: int
    cols: tuple[int] -- (y, y+1)
    Vs: (Lx, Ly) ndarray of vertex indices
    Fs: (Lx-1, Ly-1) ndarray of face indices
    

    |  U? |
    i-->--j
    |  D? |

    Returns None if neither of U, D have
    valid face indices. 

    Returns U or D if valid index (not None)

    Throws error if both U, D have valid indices
    (are not None)
    '''
    assert cols[1] == cols[0]+1
    
    Lx, Ly = Vs.shape

    assert Fs.shape == (Lx-1,Ly-1)

    #face indices above/below
    U, D = None, None

    if row < Lx-1: #there is a face site below
        D = Fs[row, cols[0]]

    if row > 0: #there is a face site above
        U = Fs[row - 1, cols[0]]

    #at MOST one face should have a valid index
    assert (D is None) or (U is None) 

    #if one of U/D is not None, return it.
    #Otherwise return None
    if D != None: 
        return D
    else: 
        return U

def find_face_right_left(rows, col, Vs, Fs):
    '''
    `rows`: tuple (k, k+1)
    `col`: int
    `Vs`: (Lx, Ly) ndarray of vertex indices
    `Fs`: (Lx-1, Ly-1) ndarray of face indices
    
    -----i-----
    | L? | R? |
    -----j-----

    Returns None if neither of L, R have
    valid face qubits. 

    Returns L or R if valid qubit site (not None)

    Throws error if both L, R aren't None,
    since they shouldn't both should have qbits.
    '''    

    assert rows[1]==rows[0]+1

    Lx, Ly = Vs.shape

    assert Fs.shape == (Lx-1, Ly-1)

    L, R = None, None

    if col < Ly-1: #there is a face site on right
        R = Fs[rows[0], col]

    if col > 0: #there is a face site on left
        L = Fs[rows[0], col-1]

    #at MOST one face should have a qubit
    assert L==None or R==None 

    #if one of L/R is not None, return it.
    #Otherwise return None
    if L != None: 
        return L
    else: 
        return R

    
def edges_commute(edges):
    '''Check that no edges share a vertex
    '''
    vertices=[]
    for (i,j,_) in edges:
        vertices.extend([i,j])

    return len(vertices)==len(set(vertices))


def is_diagonal(a):
    return np.allclose(a, np.diag(np.diag(a)))

