import quimb as qu
from operator import add
import functools
import itertools
import operator
import spinlessQubit as sqb
import numpy as np

class ElectronHubbard():
    
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

        self._allEdges = list(itertools.chain(self._edgesU,
                                                self._edgesR,
                                                self._edgesD,
                                                self._edgesL))

        self._Nsites = V_ind.size
        self._dims = [4] * self._Nsites
        self._shape = (Lx,Ly)



    def H_hop(self, i, j, spin):
        '''
        Qubit operator for fermion-hopping 
        on sites i, j
        '''
        ci, cj = self.jw_annihil_op(i, spin), self.jw_annihil_op(j, spin)
        cidag, cjdag = self.jw_creation_op(i,spin), self.jw_creation_op(j,spin)
        return cidag@cj + cjdag@ci



    def H_onsite(self, i):
        '''
        On-site spin-spin repulsion
        '''
        n_op = fermi_number_op
        return qu.pkron(n_op&n_op, dims=self._dims, inds=i)


    def build_hubbard_ham(self, t=1.0, U=0.0):
        def hops():
            for (i,j) in self._allEdges:
                yield t * self.H_hop(i, j, 'up')
                yield t * self.H_hop(i, j, 'down')
        
        def onsites():
            for i in self._V_ind.flatten():
                yield U * self.H_onsite(i)

        hopping_terms, onsite_terms = 0, 0
        
        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if U != 0.0: onsite_terms = functools.reduce(operator.add, onsites())

        H = hopping_terms + onsite_terms

        if qu.isreal(H): 
            H = H.real    

        self._Ham = H

        
    def jw_annihil_op(self, k, spin):
        '''
        Annihilation Jordan-Wigner qubit operator,
        acting on site k and `spin` sector.

        (Z_0)x(Z_1)x ...x(Z_k-1)x |0><1|_k
        '''
        spin = {0: 0,
                1: 1,
                'up': 0,
                'down': 1,
                'u': 0,
                'd': 1
                }[spin]

        Z, I = qu.pauli('z'), qu.eye(2)
        s_minus = qu.qu([[0,1],[0,0]]) #|0><1|
        N = self._Nsites
        
        Z_sig = {0: Z & I, 
                 1: I & Z}[spin]

        s_minus_sig = { 0: s_minus & I,
                        1: I & s_minus
                     }[spin]
                    
       
        op_list = [Z_sig for i in range(k)] + [s_minus_sig]
        ind_list = [i for i in range(k+1)]

        return qu.ikron(ops=op_list, dims=self._dims, 
                        inds=ind_list)
    

    def jw_creation_op(self, k, spin):
        '''
        Jordan-Wigner transformed creation operator,
        for site k and `spin` sector

        (Z_0)x(Z_1)x ...x(Z_k-1)x |1><0|_k
        '''
        spin = {0: 0,
                1: 1,
                'up': 0,
                'down': 1,
                'u': 0,
                'd': 1
                }[spin]

        Z, I = qu.pauli('z'), qu.eye(2)
        s_plus = qu.qu([[0,0],[1,0]])
        N = self._Nsites

        Z_sig = {0: Z & I, 
                 1: I & Z}[spin]

        s_plus_sig = {  0: s_plus & I,
                        1: I & s_plus
                     }[spin]
                    
       
        op_list = [Z_sig for i in range(k)] + [s_plus_sig]
        ind_list = [i for i in range(k+1)]

        return qu.ikron(ops=op_list, dims=self._dims, 
                        inds=ind_list)
    



# *************************

def fermi_number_op():
    return qu.qu([[0,0],[0,1]])
