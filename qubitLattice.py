import numpy as np
import quimb as qu
from operator import add
import functools
import itertools
import operator

class QubitLattice():

    def __init__(self, Lx=2, Ly=3, spin='half'):
        '''
        
        vertex-site indices --> self._V_ind
        face-site indices   --> self._F_ind

        ***

        Make an (Lx,Ly) (quasi)lattice, where 
        each site has local physical dim=4:
        effectively two qubits per site, for 
        spin-up/down sectors of fermionic Fock space.

        => We thus have 2N qbits (counting both vertex and
        face qbits), residing in pairs on the N lattice
        sites.

        Lattice indices: i = {0, 1, ... N-1}

        '''
        self._Lx = Lx
        self._Ly = Ly
        self._spin = spin

        #generate indices for ONE qbit lattice
        V_ind, F_ind = VF_inds(Lx,Ly)

        #vertex/face indices (for first lattice)
        self._V_ind = V_ind
        self._F_ind = F_ind
        
        
        #number of "2-qubit" sites on one lattice
        self._Nsites = V_ind.size + F_ind[F_ind!=None].size
        
        self._dims = [4] * self._Nsites

        #lists of tuples (i, j, f)
        self._edgesR = get_R_edges(V_ind, F_ind)
        self._edgesU = get_U_edges(V_ind, F_ind)
        self._edgesD = get_D_edges(V_ind, F_ind)
        self._edgesL = get_L_edges(V_ind, F_ind)
        
        self._HamSim = None
        self._HamCode = None
        self._eigens, self._eigstates = None, None

    def H_spin_int(self, i):
        '''
        Spin-spin repulsion at site i (should only
        be called for vertex sites, not face sites)

        (n^up_i)(n^down_i) --> (1-Vi^up)(1-Vi^down)/4

        Projects onto [down (x) down] for qbits at index i
        '''
        return qu.ikron(ops=[self.q_number_op() & self.q_number_op()],
                        dims=self._dims,
                        inds=[i])


    def H_hop_sup(self, i, j, f, dir):
        '''
        For fermionic spin-UP sector:

        Edge i->j qbits
        Face f qbit

        ``dir`` is 'vertical' or 'horizontal'

        Returns "hopping" term acting on sites ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of = {Xf, Yf, I} depending on edge
        '''
        
        #acts trivially on "second" qbit, i.e. on fermionic spin-down sector
        XI, YI = (qu.pauli(mu) & qu.eye(2) for mu in ['x','y'])
        Of = {'vertical': XI, 'horizontal':YI}[dir]


        #if no face qbit, Of = II identity
        if f==None:         
            XXO=qu.ikron(ops=[XI,XI], dims=self._dims, inds=[i,j])
            YYO=qu.ikron(ops=[YI,YI], dims=self._dims, inds=[i,j])
        
        #otherwise, let Of act on index f
        else:
            XXO=qu.ikron(ops=[XI,XI,Of], dims=self._dims, inds=[i,j,f])
            YYO=qu.ikron(ops=[YI,YI,Of], dims=self._dims, inds=[i,j,f])

        return 0.5*(XXO+YYO) 
    

    def H_hop_sdown(self, i, j, f, dir):
        '''
        For fermionic spin-DOWN sector:

        Edge i->j qbits
        Face f qbit

        ``dir`` is 'vertical' or 'horizontal'

        Returns "hopping" term acting on sites ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of = {Xf, Yf, I} depending on edge
        '''
        
        #acts trivially on "first" qbit, i.e. on fermionic spin-up sector
        IX, IY = (qu.eye(2) & qu.pauli(mu)  for mu in ['x','y'])
        Of = {'vertical': IX, 'horizontal':IY}[dir]


        #if no face qbit, Of = II identity
        if f==None:         
            XXO=qu.ikron(ops=[IX,IX], dims=self._dims, inds=[i,j])
            YYO=qu.ikron(ops=[IY,IY], dims=self._dims, inds=[i,j])
        
        #otherwise, let Of act on index f
        else:
            XXO=qu.ikron(ops=[IX,IX,Of], dims=self._dims, inds=[i,j,f])
            YYO=qu.ikron(ops=[IY,IY,Of], dims=self._dims, inds=[i,j,f])

        return 0.5*(XXO+YYO) 


    def q_number_op(self):
        '''
        Single-qubit operator: projects
        to spin-down in a 2-dim space.

        Equivalent to fermionic number op
        acting on ONE spin sector.
        '''
        return qu.qu([[0, 0], [0, 1]])


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
        

        
    def build_spin_Hubbard(self, t=1.0, U=1.0):
        '''
        Makes spin-1/2 Hubbard Hamiltonian, stores in self._Ham.

        Hubbard H = t <pairwise hopping>
                  + U <on-site spin-spin repulsion> 
        '''

        def hops(): #edgewise hopping terms 

            for (i,j,f) in self._edgesR + self._edgesL:
                #first qbit (fermion spin-up)
                yield t * self.H_hop_sup(i, j, f, 'horizontal')
                #second qbit (fermion spin-down)
                yield t * self.H_hop_sdown(i, j, f, 'horizontal') 

            for (i,j,f) in self._edgesU:
                yield -t * self.H_hop_sup(i, j, f, 'vertical')
                yield -t * self.H_hop_sdown(i, j, f, 'vertical')
            
            for (i,j,f) in self._edgesD:
                yield  t * self.H_hop_sup(i, j, f, 'vertical')
                yield  t * self.H_hop_sdown(i, j, f, 'vertical')


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
        Solves full spectrum of Hamiltonian, 
        stores eigensystem.

        Assumes self._Ham exists!
        '''
        self._eigens, self._eigstates = qu.eigh(self._Ham)


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


    def second_lat(self, V1, F1):
        '''
        Given Lattice 1, return a lattice with
        all valid indices increased by N 
        (N = # of qubits in the first lattice)

        `V1`: vertex indices of Lattice 1

        `F1`: face indices of Lattice 1 (includes
        None entries for faces without qbits!)
        '''
        Lx,Ly = V1.shape
        assert F1.shape==(Lx-1,Ly-1)

        V2 = np.copy(V1) + self._Nsites
        
        F2 = np.copy(F1) #
        for i in range(Lx-1):
            for j in range(Ly-1):
                if F2[i,j]!=None: F2[i,j] += self._Nsites
        
        return V2, F2
    
###
#  END OF CLASS QubitLattice() 
###



def VF_inds(Lx,Ly):

    '''
    Returns: V_ind, F_ind

    

    Orders the lattice sites:

    0-->-1-->-2-->-3
    | 16 |    | 17 |
    4-<--5-<--6-<--7
    |    | 18 |    |
    8-->-9-->-10->-11
    | 19 |    | 20 |
    12-<-13-<-14-<-15


    V_ind.shape = (Lx,Ly)
    V_ind contains the vertex indices,
    
    (0, 1, ..., Lx*Ly -1)

    0----1----2----3
    |    |    |    |
    4----5----6----7
    |    |    |    |
    8----9----10---11
    |    |    |    |
    12---13---14---15

    and F_ind contains face sites
    (Lx*Ly, Lx*Ly+1, ..., N-1)

     ---- ---- ----
    | 16 |    | 17 |
     ---- ---- ----
    |    | 18 |    |
     ---- ---- ----
    | 19 |    | 20 |
     ---- ---- ----

    where N is the total qbits per "lattice".

    **Note that F_ind only contains indices for faces
    that contain a qbit, i.e. skips the even faces!
    '''
    V_ind = np.arange(Lx*Ly).reshape(Lx,Ly)
    
    F_ind = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    
    N_vert = Lx*Ly

    f, k = 0, 0
    for i in range(Lx-1):
        for j in range(Ly-1):
            
            if f%2==0: 
                F_ind[i,j] = k + N_vert
                k+=1
            else: pass
            
            f+=1
    
    return V_ind, F_ind

def get_R_edges(V_ind, F_ind):
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

    for row in range(0,Lx,2): #only even rows
        for col in range(Ly-1):
                i, j = V_ind[row,col], V_ind[row,col+1]
                f = findFaceUD(row=row, cols=(col,col+1), Vs=V_ind, Fs=F_ind)
                edgesR.append((i, j, f))

    return edgesR

'''
TODO: COMMENT
'''
def get_L_edges(V_ind, F_ind):
    '''
    See `get_R_edges()`.

    Same method, but returns leftward edges
    rather than rightward.
    '''
    Lx, Ly = V_ind.shape
    assert F_ind.shape==(Lx-1,Ly-1)

    edgesL=[]

    for row in range(1,Lx,2):
        for col in range(Ly-1,0,-1):
            i,j = V_ind[row, col], V_ind[row, col-1]
            f = findFaceUD(row=row, cols=(col-1,col), Vs=V_ind, Fs=F_ind)
            edgesL.append((i, j, f))

    return edgesL

'''
TODO: COMMENT
'''
def get_U_edges(V_ind, F_ind):
    Lx, Ly = V_ind.shape
    edgesU=[]

    for row in range(Lx-1,0,-1):
        for col in range(0, Ly, 2):
            i, j = V_ind[row, col], V_ind[row-1, col]
            f = findFaceLR(rows=(row-1,row), col=col, Vs=V_ind, Fs=F_ind)
            edgesU.append((i,j,f))

    return edgesU

'''
TODO: COMMENT
'''
def get_D_edges(V_ind, F_ind):
    '''
    aaaah
    '''
    Lx, Ly = V_ind.shape
    assert F_ind.shape == (Lx-1,Ly-1)
    
    edgesD=[]

    for row in range(0,Lx-1):
        for col in range(1, Ly, 2):
            i, j = V_ind[row, col], V_ind[row+1, col]
            f = findFaceLR(rows=(row,row+1), col=col, Vs=V_ind, Fs=F_ind)
            edgesD.append((i,j,f))

    return edgesD


def findFaceUD(row, cols, Vs, Fs):
    '''
    `row`: int
    `cols`: tuple (k,k+1)
    `Vs`: (Lx, Ly) ndarray of vertex indices
    `Fs`: (Lx-1, Ly-1) ndarray of face indices
    

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
    assert D==None or U==None 

    #if one of U/D is not None, return it.
    #Otherwise return None
    if D != None: 
        return D
    else: 
        return U

def findFaceLR(rows, col, Vs, Fs):
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

    Throws error if both L, R are valid sites
    (are not None)
    '''    

    assert rows[1]==rows[0]+1

    Lx, Ly = Vs.shape

    assert Fs.shape == (Lx-1, Ly-1)

    L, R = None, None

    if col < Ly-1: #there is a face site on right
        R = Fs[rows[0], col]

    if col > 0: #there is a face site on left
        L = Fs[rows[0], col-1]

    #at MOST one face should have a valid index
    assert L==None or R==None 

    #if one of L/R is not None, return it.
    #Otherwise return None
    if L != None: 
        return L
    else: 
        return R
    
