import numpy as np
import quimb as qu
from operator import add
import functools
import itertools
import operator
import spinlessQubit

class DenseSpinhalfEncoder(spinlessQubit.QubitCodeLattice):

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
        
        # if (Lx-1)*(Ly-1) % 2 == 1:
        #     raise NotImplementedError('Need even number of faces!')

        # self._Lx = Lx
        # self._Ly = Ly
        # self._lat_shape = (Lx, Ly)

        # #generate indices for qbit lattice
        # verts, faces = spinlessQubit.gen_lattice_sites(Lx,Ly)
        
        # #vertex/face indices in np.ndarrays
        # self._verts = verts
        # self._faces = faces


        # self._edge_map = spinlessQubit.make_edge_map(verts,faces)


        # #number lattice sites (IGNORES faces w/out qudits)
        # self._Nsites = verts.size + faces[faces!=None].size
        
        # self._d_physical = 4
        # self._sim_dims = [4]*(self._Nsites)
        

        # #number of vertices, i.e. fermionic sites
        # self._Nfermi = verts.size


        # #TODO: not true for odd num. faces
        # #codespace dimensions = dim(Fock)
        # self._encoded_dims = [4]*(self._Nfermi)

        #Simulator Hamiltonian in full qubit space
        self._HamSim = None


        #Codespace Hamiltonian
        self._HamCode = None
        self._eigens, self._eigstates = None, None

        super().__init__(Lx, Ly, local_dim=4)

    

    # def get_edges(self, which):

    #     if which == 'horizontal':
    #         return self._edge_map['r'] + self._edge_map['l']
        

    #     if which == 'all':
    #         return (self._edge_map['r']+
    #                 self._edge_map['l']+
    #                 self._edge_map['u']+
    #                 self._edge_map['d'] )


    #     key =  {'d' : 'd',
    #             'down':'d',
    #             'u' : 'u',
    #             'up': 'u',
    #             'left':'l',
    #             'l' : 'l',
    #             'right':'r',
    #             'r':'r',
    #             'he':'he',
    #             'ho':'ho',
    #             've':'ve',
    #             'vo':'vo'}[which]
        
    #     return self._edge_map[key]



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

    
    # def vertex_sites(self):
    #     '''Array of indices for vertex qubits,
    #     equivalent to range(Lx*Ly)
    #     '''
    #     return list(self._verts.flatten().copy())


    # def face_sites(self):
    #     '''Indices of face-qubits 
    #     '''
    #     F = self._faces.flatten()
    #     return list(F[F!=None])


    # def all_sites(self):
    #     '''All 'qudit' indices (vertex and face)
    #     '''
    #     return self.vertex_sites() + self.face_sites()

    
    # def vert_array(self):
    #     '''ndarray of vertex site numbers
    #     '''
    #     return self._verts.copy()
    
    # def face_array(self):
    #     '''ndarray of face site numbers
    #     '''
    #     return self._faces.copy()

    # def num_verts(self):
    #     return self._verts.size
    
    # def num_faces(self):
    #     return self._faces[self._faces!=None].size

    # def num_sites(self):
    #     return self.num_faces()+self.num_verts()


    # def second_lat(self, V1, F1):
    #     '''
    #     Given Lattice 1, return a lattice with
    #     all valid indices increased by N 
    #     (N = # of qubits in the first lattice)

    #     `V1`: vertex indices of Lattice 1

    #     `F1`: face indices of Lattice 1 (includes
    #     None entries for faces without qbits!)
    #     '''
    #     Lx,Ly = V1.shape
    #     assert F1.shape==(Lx-1,Ly-1)

    #     V2 = np.copy(V1) + self._Nsites
        
    #     F2 = np.copy(F1) #
    #     for i in range(Lx-1):
    #         for j in range(Ly-1):
    #             if F2[i,j]!=None: F2[i,j] += self._Nsites
        
    #     return V2, F2
    
