import quimb as qu
from operator import add
import functools
import itertools
import operator
import spinlessQubit as sqb
import numpy as np

DEBUG=0

class HubbardSpinless():
    def __init__(self, Lx=2, Ly=3):
        V_ind, F_ind = sqb.gen_lattice_sites(Lx,Ly)

        self._V_ind = V_ind
        
        self._edgesR = [(i,j) for (i,j,f) in 
                        sqb.get_R_edges(V_ind, F_ind)]
        self._edgesL = [(i,j) for (i,j,f) in 
                        sqb.get_L_edges(V_ind, F_ind)]
        self._edgesU = [(i,j) for (i,j,f) in 
                        sqb.get_U_edges(V_ind, F_ind)]
        self._edgesD = [(i,j) for (i,j,f) in 
                        sqb.get_D_edges(V_ind, F_ind)]

        self._allEdges = self._edgesU+self._edgesR+self._edgesD+self._edgesL

        self._N = V_ind.size
        self._dims = [2] * self._N
        self._shape = (Lx,Ly)


    
    def H_hop_boson(self, i, j):
        '''
        Hopping term *without* anti-symmetrization
        '''
        c, cdag = annihil_op(), creation_op()
        ccd = qu.ikron(ops=[c, cdag], dims=self._dims, inds=[i, j])
        cdc = qu.ikron(ops=[c, cdag], dims=self._dims, inds=[j, i])
        return ccd+cdc


    def H_hop_fermion(self, i, j):
        '''
        Qubit operator for fermion-hopping 
        on sites i, j
        '''
        ci, cj = self.jw_annihil_op(i), self.jw_annihil_op(j)
        cidag, cjdag = self.jw_creation_op(i), self.jw_creation_op(j)
        return cidag@cj + cjdag@ci


    def H_occ(self, i):
        '''
        Occupation/number operator at site i
        '''
        return qu.ikron(ops=number_op(), dims=self._dims, inds=[i])


    def H_nn(self, i, j):
        '''
        Nearest-neighbor repulsion
        (n_i)x(n_j)
        '''
        nop = number_op()
        return qu.ikron(ops=[number_op(),number_op()],
                        dims=self._dims, inds=[i,j])


    def build_spinless_ham(self, t=1.0, V=0.0, mu=0.0):

        def hops():
            for (i,j) in self._allEdges:
                # print('Edge {},{}'.format(i,j))
                yield t * self.H_hop_fermion(i,j)

        def interactions():
            for (i,j) in self._allEdges:
                yield V * self.H_nn(i,j)

        def occs():
            for i in self._V_ind.flatten():
                yield -mu * self.H_occ(i)

        hopping_terms, int_terms, occ_terms = 0, 0, 0

        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if V != 0.0: int_terms = functools.reduce(operator.add, interactions())
        if mu != 0.0: occ_terms = functools.reduce(operator.add, occs())

        H = hopping_terms + int_terms + occ_terms

        if qu.isreal(H): 
            H = H.real    

        self._Ham = H

    
    def state_occs(self, state):
        Nk = [qu.ikron(number_op(), self._dims, [site]) 
            for site in self._V_ind.flatten()]

        n_ij = np.real(np.array([qu.expec(Nk[k], state)
                for k in range(self._N)])).reshape(self._shape)
        
        return n_ij

    def parity(self, state):
        '''
        +1 if even, -1 if odd 
        '''
        if isinstance(state, int):
            state = qu.basis_vec(i=state, dim=np.prod(self._dims))
        
        occ = self.state_occs(state)
        assert np.logical_or(occ==0.0, occ==1.0).all()
        par = np.sum(occ)%2
        return {0.0: 1, 
                1.0:-1}[par]

    ###
    # JORDAN WIGNER
    ###
    
    def jw_annihil_op(self, k):
        '''
        Annihilation Jordan-Wigner qubit operator.

        (Z_0)x(Z_1)x ...x(Z_k-1)x |0><1|_k
        '''
        Z = qu.pauli('z')
        s_minus = qu.qu([[0,1],[0,0]]) #|0><1|

        ind_list = [i for i in range(k+1)]
        op_list = [Z for i in range(k)] + [s_minus]

        return qu.ikron(ops=op_list, dims=self._dims, 
                        inds=ind_list)
    
    def jw_creation_op(self, k):
        '''
        Creation Jordan-Wigner qubit operator.

        (Z_0)x(Z_1)x ...x(Z_k-1)x |1><0|_k
        '''
        Z = qu.pauli('z')
        s_plus = qu.qu([[0,0],[1,0]]) #|1><0|

        ind_list = [i for i in range(k+1)]
        op_list = [Z for i in range(k)] + [s_plus]

        return qu.ikron(ops=op_list, dims=self._dims, 
                        inds=ind_list)
    

    ###
    # End Jordan-Wigner methods
    ###


def creation_op():
    return qu.qu([[0,0],[1,0]])

def annihil_op():
    return qu.qu([[0,1],[0,0]])

def number_op():
    return qu.qu([[0,0],[0,1]])


# ***************************************** #
