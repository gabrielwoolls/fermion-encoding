import quimb as qu
from operator import add
import functools
import itertools
import operator
import spinlessQubit as sqb

class FermiHubbardSpinless():
    def __init__(self, Lx=2, Ly=3):
        V_ind, F_ind = sqb.VF_inds(Lx,Ly)

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


    def H_hop(self, i, j):
        c, cdag = annihil_op(), creation_op()
        ccd = qu.ikron(ops=[c, cdag], dims=self._dims, inds=[i, j])
        cdc = qu.ikron(ops=[c, cdag], dims=self._dims, inds=[j, i])
        return ccd+cdc


    def H_occ(self, i):
        return qu.ikron(ops=number_op(), dims=self._dims, inds=[i])


    def H_nn(self, i, j):
        nop = number_op()
        return qu.ikron(ops=[number_op(),number_op()],
                        dims=self._dims, inds=[i,j])


    def build_spinless_ham(self, t=1.0, V=0.0, mu=0.0):
        def hops():
            for (i,j) in self._allEdges:
                yield t * self.H_hop(i,j)

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


def creation_op():
    return qu.qu([[0,0],[1,0]])

def annihil_op():
    return qu.qu([[0,1],[0,0]])

def number_op():
    return qu.qu([[0,0],[0,1]])