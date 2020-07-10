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
        V_ind, F_ind = gen_lattice_sites(Lx,Ly)

        #vertex/face indices in np.ndarrays
        self._V_ind = V_ind
        self._F_ind = F_ind

        #number of qbits on lattice
        #IGNORES faces w/out qubits
        self._Nsites = V_ind.size + F_ind[F_ind!=None].size
        self._dims = [2]*(self._Nsites)

        #number of vertex qubits only, aka
        #number of fermionic sites.
        self._Nfermi = V_ind.size

        #TODO: this is fake
        #Obtained from qubit dims by tracing out
        #the auxiliary face qubit dimensions
        self._fermi_dims = [2]*(self._Nfermi)

        #lists of edge tuples (i, j, f(i,j)) so that
        #  (i, j, r) 
        #corresponds to edge i-->j and face qbit r
        self._edgesR = get_R_edges(V_ind, F_ind)
        self._edgesL = get_L_edges(V_ind, F_ind)
        self._edgesU = get_U_edges(V_ind, F_ind)
        self._edgesD = get_D_edges(V_ind, F_ind)
        
        #Simulator Hamiltonian in full qubit space
        self._HamSim = None
        
        #Codespace Hamiltonian
        #(written in stabilizer eigenbasis)
        self._HamCode = None
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
        TODO: pkron vs ikron performance?

        i, j: (directed) edge qubits indices
        f: face qubit index
        dir: 'vertical' or 'horizontal'

        Returns hopping term acting on qbits ijf,
            
            (XiXjOf + YiYjOf)/2
        
        where Of is Xf, Yf or Identity depending on ijf
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        Of = {'vertical': X, 'horizontal':Y} [dir]
        
        #if no face qbit
        if f==None:  
            # print('{}--{}-->{}   (None)'.format(i,dir[0],j))       
            # XXO=qu.ikron(ops=[X,X], dims=self._dims, inds=[i,j])
            # YYO=qu.ikron(ops=[Y,Y], dims=self._dims, inds=[i,j])
            XXO=qu.pkron(op=X&X, dims=self._dims, inds=[i,j])
            YYO=qu.pkron(op=Y&Y, dims=self._dims, inds=[i,j])
        
        #if there's a face qubit: Of acts on index f
        else:
            # print('{}--{}-->{},  face {}'.format(i,dir[0],j,f))
            # XXO=qu.ikron(ops=[X,X,Of], dims=self._dims, inds=[i,j,f])
            # YYO=qu.ikron(ops=[Y,Y,Of], dims=self._dims, inds=[i,j,f])
            XXO = qu.pkron(op=X&X&Of, dims=self._dims, inds=(i,j,f))
            YYO = qu.pkron(op=Y&Y&Of, dims=self._dims, inds=(i,j,f))

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
            
            for (i,j,f) in self._edgesR:
                yield t * self.H_hop(i, j, f, 'horizontal')
            
            for (i,j,f) in self._edgesL:
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
        


    def single_qubit_dm(self, qstate, i):
        '''
        Returns subsystem density
        matrix for qubit i by tracing out
        all other qubits in `qstate`.

        `qstate` needs to be in *full* qubit space!
        '''
        assert qstate.size == qu.prod(self._dims)
        return qu.ptr(state, dims=self._dims, keep=i)


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
        ds = {False : self._dims,
              True : self._fermi_dims} [fermi]

        return [qu.ikron(self.number_op(), ds, [site]) 
                        for site in sites]#.reshape(self._shape)


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
        # Nj = self.site_number_ops(sites=self.allSiteInds() , fermi=False)
        Nj = [qu.ikron(self.number_op(), self._dims, [site]) 
                        for site in self.allSiteInds()]

        #expectation <N> for each vertex
        nocc_v = np.array([qu.expec(Nj[v], qstate)
                        for v in self.vertexInds()]).reshape(self._shape)

        if not faces:
            return nocc_v

        #expectation <Nj> for each face
        nocc_f = np.array([qu.expec(Nj[f], qstate) for f in self.faceInds()])
        
        return nocc_v, nocc_f
            

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


    def make_stabilizer(self):
        '''
        TODO: 
        * generalize indices, rotation, reshape, etc
        * always need to round? check always real
        * strange fix: if remove the "copy()" from Ux
        in ikron, quimb raises ownership error
        '''
        
        X, Z = (qu.pauli(mu) for mu in ['x','z'])
        
        ## TODO: make general
        # ops = [Z,Z,Z,Z,X]
        # inds = [1,2,4,5,6]
        # stabilizer = qu.ikron(ops=ops, dims=self._dims, inds=inds)
        
        _, Ux = qu.eigh(qu.pauli('x')) 
        
        stabilizer = self.loop_stabilizer(0,1)

        #TODO: change to general rather than inds=6, Ux
        U = qu.ikron(Ux.copy(), dims=self._dims, inds=[6])
        
        #TODO: can this be done without rounding()/real part?
        Stilde = (U.H @ stabilizer @ U).real.round()
        assert is_diagonal(Stilde)

        U_plus = U[:, np.where(np.diag(Stilde)==1.0)[0]]
        
        HamCode = U_plus.H @ self.ham_sim() @ U_plus

        self._stabilizer = stabilizer
        self._Uplus = U_plus #+1 eigenstates written in full qubit basis
        self._HamCode = HamCode # in +1 stabilizer eigenbasis


        
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


    #COMMENT
    def t_make_stabilizers(self):
        '''
        TODO: test! Could be completely wrong

        To be used in general case when lattice has 
        multiple stabilizer operators.
        '''
        
        self._stabilizers = {}
        
        for (i,j) in np.argwhere(self._F_ind==None):

            loop_op_ij = self.loop_stabilizer(i,j)
            
            self._stabilizers[(i,j)] = loop_op_ij

    def loop_stabilizer(self, i, j):
        '''
        Returns loop operator corresponding to face
        at location (i,j) in face array (self._F_ind)

        TODO: 
        *check signs, edge directions
        *consider ZZZZ case?
        *
        '''
        X, Y, Z = (qu.pauli(mu) for mu in ['x','y','z'])
        
        Vs, Fs = self._V_ind, self._F_ind
        
        if Fs[i,j] != None: 
            raise ValueError('Face at ({},{}) has a qubit!'.format(i,j))

        #corner vertex sites
        #start upper-left --> clockwise
        v1 = Vs[i, j]
        v2 = Vs[i, j+1]
        v3 = Vs[i+1, j+1]
        v4 = Vs[i+1, j]

        print('Stabilizer:')
        print('{}-----{}\n|     |\n{}-----{}'.format(v1,v2,v4,v3))

        u_face = findFaceUD(row=i, cols=(j,j+1), Vs=Vs, Fs=Fs)
        d_face = findFaceUD(row=i+1, cols=(j,j+1), Vs=Vs, Fs=Fs)
        l_face = findFaceLR(rows=(i,i+1), col=j, Vs=Vs, Fs=Fs)
        r_face = findFaceLR(rows=(i,i+1), col=j+1, Vs=Vs, Fs=Fs)

        print(u_face, d_face, l_face, r_face, end='\n\n')

        ops = [Z, Z, Z, Z]
        inds = [v1, v2, v3, v4]

        if u_face != None:
            inds.append(u_face)
            ops.append(Y)
        
        if d_face != None:
            inds.append(d_face)
            ops.append(Y)

        if r_face != None:
            inds.append(r_face)
            ops.append(X)
        
        if l_face != None:
            inds.append(l_face)
            ops.append(X)
        
        loop_op = qu.ikron(ops=ops, dims=self._dims, inds=inds)
        
        if qu.isreal(loop_op): 
            loop_op = loop_op.real

        return loop_op

    # TODO: delete?
    def projected_ham_2(self):
        '''
        Alternative projected Hamiltonian.

        Obtained by numerically diagonalizing stabilizer
        rather than "manually", so pHam should be equivalent 
        to regular HamCode under a basis rotation.
        '''

        evals, V = qu.eigh(self._stabilizer)
        
        assert np.array_equal(evals,np.array([-1.0]*64 + [1.0]*64))

        if qu.isreal(V): V=V.real

        projector = V @ np.diag( [0]*64 + [1]*64 ) @ V.H #projector onto +1 eigenspace

        self._stab_projector = projector

        Vp = V[:, np.where(evals==1.0)].reshape(128,64)
        #shape=(128,64)

        pHam = Vp.H @ self._HamSim @ Vp 
        return pHam
        #(64,64)
        
    #TODO: delete?
    def projected_ham_3(self):
        '''
        Should be equivalent to self.ham_code()
        '''
        #X because stabilizer acts with X on 7th qubit
        _, Ux = qu.eigh(qu.pauli('x')) 
        
        U = qu.ikron(Ux.copy(), dims=self._dims, inds=[6])
        
        Stilde = (U.H @ self.stabilizer() @ U).real.round()
        #Stilde should be diagonal!

        U_plus = U[:, np.where(np.diag(Stilde)==1.0)[0]]
        
        HamProj = U_plus.H @ self.ham_sim() @ U_plus
        
        return HamProj #in +1 stabilizer eigenbasis
        

def gen_lattice_sites(Lx,Ly):

    '''
    Generate sites for lattice of shape (Lx,Ly)

    Returns (Vs, Fs) tuple of arrays
    (Vertex indices, face indices)

    Qubit sites ordered thus:

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

    f, k = 0, 0
    for i in range(Lx-1):
        for j in range(Ly-1):
            if f%2==0: 
                Fs[i,j] = k + N_vert
                k+=1
            else: pass
            f+=1
    
    return Vs, Fs


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

    # for row in range(0,Lx,2):
    for row in range(1,Lx,2): 
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

    # for row in range(1,Lx,2):
    for row in range(0,Lx,2):
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
    

def is_diagonal(a):
    return np.allclose(a, np.diag(np.diag(a)))