import numpy as np
import quimb as qu
from operator import add
import functools
import itertools
import operator

class DoubleQubitLattice():

    def __init__(self, Lx=2, Ly=3, spin='half'):
        '''
        
        vertex-site indices --> self._V_ind
        face-site indices   --> self._F_ind

        ***

        Make an (Lx,Ly) (quasi)lattice, where 
        each site has local physical dim=4:
        effectively two qubits per site, for 
        spin up/down sectors of fermionic Fock space.

        => We thus have 2N qbits (counting both vertex and
        face qbits) residing in pairs on the N lattice
        sites.

        Lattice indices: i = {0, 1, ... N-1}

        '''
        self._Lx = Lx
        self._Ly = Ly
        self._shape = (Lx, Ly)

        #generate indices for ONE qbit lattice
        V_ind, F_ind = VF_inds(Lx,Ly)

        #vertex/face indices (for first lattice)
        self._V_ind = V_ind
        self._F_ind = F_ind
        
        
        #number of "2-qubit" sites on one lattice
        self._Nsites = V_ind.size + F_ind[F_ind!=None].size
        self._dims = [4] * self._Nsites

        #number of vertex qubits only, aka
        #number of fermionic sites.
        self._Nfermi = V_ind.size
        self._dims_code = [2]*(self._Nfermi)


        #lists of tuples (i, j, f)
        self._edgesR = get_R_edges(V_ind, F_ind)
        self._edgesU = get_U_edges(V_ind, F_ind)
        self._edgesD = get_D_edges(V_ind, F_ind)
        self._edgesL = get_L_edges(V_ind, F_ind)
        
        self._HamSim = None
        self._HamCode = None
        self._eigens, self._eigstates = None, None



    def H_onsite(self, i):
        '''
        Spin-spin repulsion at site i (should only
        be called for vertex sites, not face sites)

        (n^up_i)(n^down_i) --> (1-Vi^up)(1-Vi^down)/4

        Projects onto [down (x) down] for qbits at index i
        '''
        return qu.ikron(ops=[self.q_number_op() & self.q_number_op()],
                        dims=self._dims,
                        inds=[i])


    def H_hop(self, i, j, f, dir, spin):
        '''
        For fermionic `spin` sector:

        Edge i->j qbits
        Face f qbit

        ``dir`` is 'vertical' or 'horizontal'

        Returns "hopping" term acting on sites ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of = {Xf, Yf, I} depending on edge
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
            XXO=qu.ikron(ops=[X_sigma,X_sigma], dims=self._dims, inds=[i,j])
            YYO=qu.ikron(ops=[Y_sigma,Y_sigma], dims=self._dims, inds=[i,j])
        
        #otherwise, let Of act on index f
        else:
            XXO=qu.ikron(ops=[X_sigma,X_sigma,Of], dims=self._dims, inds=[i,j,f])
            YYO=qu.ikron(ops=[Y_sigma,Y_sigma,Of], dims=self._dims, inds=[i,j,f])

        return 0.5*(XXO+YYO) 
    


    def q_number_op(self):
        '''
        Single-qubit operator,
        equiv. to fermionic number
        op on a single spin sector.
        '''
        return qu.qu([[0, 0], [0, 1]])


    #TODO: delete?
    def numberop(self, sigma):
        '''
        TODO: why can't use ikron()?

        Fermionic number operator
        acting on spin=sigma sector.
        
        As qubit operator, projects 
        to spin-down (sigma determines 
        which sector in local d=4
        to act on.)

        n_j --> (1-Vj)/2 
                = (1-Zj)/2 
                = |down><down|
        '''
        #which fermion-spin sector to act on?
        #i.e. act on qbit 0 or qbit 1?
        opmap = {0 : lambda: qu.qu([[0, 0], [0, 1]]) & qu.eye(2),

                 1 : lambda: qu.eye(2) & qu.qu([[0, 0], [0, 1]])}
        
        op = opmap[sigma]() 

        qu.core.make_immutable(op)
        return op
        

        
    def make_simulator_ham(self, t=1.0, U=1.0):
        '''
        Makes spin-1/2 Hubbard Hamiltonian, stores in self._Ham.

        Hubbard H = t <pairwise hopping>
                  + U <on-site spin-spin repulsion> 
        '''

        def hops(): #edgewise hopping terms 
            for spin in ['u', 'd']:
                for (i,j,f) in self._edgesR + self._edgesL:
                    yield t * self.H_hop(i, j, f, 'horizontal',spin)

                for (i,j,f) in self._edgesU:
                    yield -t * self.H_hop(i, j, f, 'vertical',spin)
                
                for (i,j,f) in self._edgesD:
                    yield  t * self.H_hop(i, j, f, 'vertical',spin)


        def interactions(): #onsite spin-spin interaction
            
            for i in self._V_ind.flatten():
                yield U * self.H_spin_int(i)
        

        onsite_terms, hopping_terms = 0, 0

        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if U != 0.0: onsite_terms = functools.reduce(operator.add, interactions())
            
        H = onsite_terms + hopping_terms
        
        if qu.isreal(H):
            H = H.real
        
        self._Ham = H


    def solveED(self):
        '''
        Diagonalize codespace Ham, 
        store eigensystem.
        '''
        self._eigens, self._eigstates = qu.eigh(self._HamCode)


    def eigspectrum(self):
        '''
        Returns energy eigenvalues and eigenstates of
        self._Ham, obtained through ED
        '''
        if None in [self._eigens, self._eigstates]:
            self.solveED()
        
        return np.copy(self._eigens), np.copy(self._eigstates)

    
    def vertexInds(self):
        '''
        Array of indices for vertex qubits,
        equivalent to np.arange(Lx*Ly)
        '''
        return np.copy(self._V_ind.flatten())


    def faceInds(self):
        '''
        Indices of face-qubits 
        '''
        F = np.copy(self._F_ind).flatten()
        return F[F!=None]


    def allSiteInds(self):
        '''
        All qubit indices (vertex and face)
        '''
        return np.concatenate([self.vertexInds(),
                              self.faceInds()])


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
    

    
###
#  END OF CLASS QubitLattice() 
###


