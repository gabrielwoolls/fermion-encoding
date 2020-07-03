import numpy as np
import quimb as qu
from operator import add
import functools
import itertools
import operator

class SpinlessQubitLattice():

    def __init__(self, Lx=2, Ly=3):
        '''
        TODO: odd # of faces

        Spinless fermion model is mapped to 
        a lattice (Lx, Ly) of qubits, following
        the low-weight fermionic encoding of
        Derby and Klassen (2020):

        	arXiv:2003.06939 [quant-ph]

        Vertex qubit indices --> self._V_ind
        Face qubit indices   --> self._F_ind

        Lattice site dims: [2 for i in (0, 1, ... N-1)]
        '''

        if (Lx-1)*(Ly-1)%2==1:
            raise ValueError('Odd # faces!')

        self._Lx = Lx
        self._Ly = Ly
        self._shape = (Lx, Ly)

        #generate indices for qbit lattice
        V_ind, F_ind = VF_inds(Lx,Ly)

        #vertex/face indices in np.ndarrays
        self._V_ind = V_ind
        self._F_ind = F_ind

        #number of qbits on lattice
        #IGNORES faces w/out qubits!
        self._Nsites = V_ind.size + F_ind[F_ind!=None].size
        self._dims = [2]*(self._Nsites)

        #number of vertex sites only, corresponding
        #to the physical fermionic sites.
        self._Nfermi = V_ind.size

        #Obtained from qubit dims by tracing out
        #the auxiliary face qubit dimensions
        self._fermi_dims = [2]*(self._Nfermi)

        #lists of edge tuples (i, j, f(i,j)) so that
        #  (i, j, r) 
        #corresponds to edge i-->j and face qbit r
        self._edgesR = get_R_edges(V_ind, F_ind)
        # self._edgesL = [(5,4,None), (4,3,6)]
        # self._edgesU = [(3,0,6), (5,2,None)]
        # self._edgesD = [(1,4,6)]
        self._edgesU = get_U_edges(V_ind, F_ind)
        self._edgesD = get_D_edges(V_ind, F_ind)
        self._edgesL = get_L_edges(V_ind, F_ind)

        #Qubit Hamiltonian    
        self._HamSim = None
        self._eigens, self._eigstates = None, None


    def H_nn_int(self, i, j):
        '''
        Nearest neighbor repulsive 
        interaction, for vertex qbits 
        i, j.

        Return (n_i)(n_j)
        '''
        return qu.ikron(ops=[self.number_op(), self.number_op()],
                        dims=self._dims,
                        inds=[i,j])


    def H_hop(self, i, j, f, dir):
        '''
        Edge i->j qbits
        Face f qbit

        ``dir`` : 'vertical' or 'horizontal'

        Returns hopping term acting on qbits ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of is Xf, Yf or Identity depending on edge
        '''
        print('{}--->{},  face {}'.format(i,j,f))
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        Of = {'vertical': X, 'horizontal':Y} [dir]
        
        #if no face qbit, Of = Identity
        if f==None:         
            XXO=qu.ikron(ops=[X,X], dims=self._dims, inds=[i,j])
            YYO=qu.ikron(ops=[Y,Y], dims=self._dims, inds=[i,j])
        
        #otherwise, let Of act on index f
        else:
            XXO=qu.ikron(ops=[X,X,Of], dims=self._dims, inds=[i,j,f])
            YYO=qu.ikron(ops=[Y,Y,Of], dims=self._dims, inds=[i,j,f])

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
    

    def make_spinless_Hubbard(self, t=1.0, V=0.0, mu=0.0):
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
            #generate term for each graph edge
            for (i,j,f) in self._edgesR + self._edgesL:
                yield t * self.H_hop(i, j, f, 'horizontal')

            for (i,j,f) in self._edgesU:
                #minus sign! (See Derby & Klassen)
                yield -t * self.H_hop(i, j, f, 'vertical')
            
            for (i,j,f) in self._edgesD:
                yield  t * self.H_hop(i, j, f, 'vertical')

        def interactions(): #pairwise neighbor repulsion terms
           
            allEdges = self._edgesR + self._edgesU + self._edgesL + self._edgesD
            
            for (i,j,f) in allEdges: 
                #doesn't require face qbit!
                yield V * self.H_nn_int(i,j)

        def occs(): #occupation at each vertex
            #only counts vertex qbits, ignores faces!
            for i in self._V_ind.flatten():
                yield -mu * qu.ikron(self.number_op(), 
                                    dims=self._dims, 
                                    inds=i)

        hopping_terms, int_terms, occ_terms = 0, 0, 0

        if t != 0.0: hopping_terms = functools.reduce(operator.add, hops())
        if V != 0.0: int_terms = functools.reduce(operator.add, interactions())
        if mu != 0.0: occ_terms = functools.reduce(operator.add, occs())

        H = hopping_terms + int_terms + occ_terms

        if qu.isreal(H): 
            H = H.real    

        self._HamSim = H
        
    

    def solveED(self):
        '''
        Solves full spectrum of simulator Hamiltonian, 
        stores eigensystem. 
        
        Assumes self._HamSim exists.
        '''
        self._eigens, self._eigstates = qu.eigh(self._HamSim)

    
    def eigspectrum(self):
        '''
        Returns energy eigenvalues and eigenstates of
        self._HamSim, obtained through ED
        '''
        if None in [self._eigens, self._eigstates]:
            self.solveED()
        
        return np.copy(self._eigens), np.copy(self._eigstates)


    def retrieve_tgt(self, debug=1):
        '''
        ! TODO: this is wrong

        Traces out the face-qubit degrees of freedom, 
        gets Hamiltonian restricted to the vertex
        (fermionic) degrees of freedom.
        '''

        if debug==1:
            return qu.partial_trace(self._HamSim, dims=self._dims,
                                      keep=self.vertexInds().tolist())
        # elif debug==2:
        #     H = self._HamSim.reshape(self._dims*2)
        #     H = qu.itrace(H, axes=(6,13))
        #     H = H.reshape((2**6,2**6))
        #     return {False:H, True:H.real}[qu.isreal(H)]
        
        # elif debug==3:
        #     H = np.reshape(self._HamSim, self._dims*2)
        #     H = np.einsum('ijklmnoIJKLMNO->ijklmnIJKLMN',H)
        #     return H.reshape((2**6,2**6))

    def siteNumberOps(self, sites, fermi=False):
        '''
        Returns: np.array (1D, shape = len(sites))
        containing local number operators acting 
        on specified sites.

        `fermi`: bool, indicates whether we are acting
        in the " simulator"qubit codespace (False) or in
        the fermionic "target" Fock space (True) which is
        the subspace obtained by tracing out the auxiliary
        face-qubits (up to isometry)
        '''
        ds = {False : self._dims,
              True : self._fermi_dims} [fermi]

        return np.array([qu.ikron(self.number_op(), ds, [site]) 
                        for site in sites],dtype=object)#.reshape(self._shape)

#changeeeed
    def stateLocalOccs(self, k=0, state=None):
        '''
        TODO: fix shapes. Also, number operators 
        as qubit spin-down projectors need not be 
        meaningful in the face qubits!

        Returns: nocc_v, nocc_f

        nocc_v (ndarray, shape (Lx,Ly))
        nocc_f (ndarray, shape (Lx-1,Ly-1))

        Expectation values <k|local fermi occupation|k>
        in the kth excited eigenstate, at all vertex
        sites v and faces f.

        Defaults to ground state, k=0
        '''
        #kth excited state unless given full state
        if state==None:
            state = self._eigstates[:,k] 
        
        #local number ops acting on each site j
        Nj = self.siteNumberOps(sites=self.allSiteInds() , fermi=False)
        
        #expectation <N> for each vertex
        nocc_v = np.array([qu.expec(Nj[v], state)
                        for v in self.vertexInds()]).reshape(self._shape)
        #expectation <Nj> for each face
        nocc_f = np.array([qu.expec(Nj[f], state)
                        for f in self.faceInds()])

        
        return np.real(nocc_v), np.real(nocc_f)


    def vertexInds(self):
        '''
        Indices for vertex qubits,
        equivalent to range(Lx*Ly)
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


def VF_inds(Lx,Ly):

    '''
    V_ind.shape = (Lx,Ly)

    Orders the vertex qbits like

    0-->-1-->-2-->-3
    ^ 16 v    ^ 17 v
    4-<--5-<--6-<--7
    ^    | 18 |    v
    8-->-9-->-10->-11
    ^ 19 |    | 20 v
    12-<-13-<-14-<-15

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
    



