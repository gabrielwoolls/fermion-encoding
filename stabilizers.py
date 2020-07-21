import numpy as np
import quimb as qu

class qubitStabilizer():
    def __init__(self, vert_inds, face_op, face_inds):
        '''
        Params:

        vert_inds [list/tuple of ints]: vertex qubit indices
        
        vert_op [string]: 'ZZZZ'

        face_inds: face qubit indices

        face_op [string]: action on face qubits, e.g. 'YXIX'
        '''
        assert len(face_inds)==len(face_op)
        assert len(vert_inds)==4

        self.vert_inds = vert_inds 
        self.vert_op = 'ZZZZ'
        
        self.face_op = face_op 
        self.face_inds = face_inds 
        
        
    def string_to_op(string_op):
        X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
        opmap = {'X': X, 'Y':Y, 'Z':Z, 'I':I}
        return qu.kron(*[opmap[Q] for Q in strng_op])



def two_qubit_eigsectors(strA='XY', strB='YX'):
    '''

    Params: strA, strB [strings]
        Action of stabilizers on face-qubit subspace.

    Return: `eigsectors` [ndarray(shape=(2,2), dtype=object)]
        Each element of eigsector contains the unique vector
        in the face-qubit subspace that diagonalizes strA, strB
        with respective eigenvalues +/-1.

        eigsectors[0,0]: +1, +1
        eigsectors[0,1]: +1, -1
        eigsectors[1,0]: -1, +1
        eigsectors[1,1]: -1, -1


    '''
    sign_dic = {0: 1.0,  1: -1.0}
    X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
    opmap = {'X': X, 'Y':Y, 'Z':Z, 'I':I}

    face_dims = [2]*2 #dimensions of face-qubit subspace

    # ###
    # strA = 'XY'
    # strB = 'YX'
    # ###

    SA = qu.kron(*[opmap[Q] for Q in strA])
    SB = qu.kron(*[opmap[Q] for Q in strB])
    # SA, SB = X&Y, Y&X

    eigsectors = np.ndarray(shape=face_dims, dtype=object)


    eva, Ua = qu.eigh(SA)
    for indA, signA in sign_dic.items():
        
        #pick evectors in (signA) sector
        Ua_sgn = Ua[:, np.isclose(eva, signA)] #4x2
        
        #Stabilizer B on (signA) eigspace of SA
        Qb = Ua_sgn.H @ SB @ Ua_sgn
        
        evb, Ub = qu.eigh(Qb)

        for indB, signB in sign_dic.items():

            #pick evector in signB sector
            Ub_sgn = Ub[:,np.isclose(evb, signB)] #2x1
            
            face_state =  Ua_sgn @ Ub_sgn #4x1

            assert np.allclose(SA@face_state, signA*face_state)
            assert np.allclose(SB@face_state, signB*face_state)

            eigsectors[indA,indB] = face_state

    return eigsectors



def two_qubit_codespace(qLattice):
    '''
    TODO: generalize :/

    Find the joint +1 stabilizer eigenbasis.

    Return: Uplus [qarray, shape = (2**Nqubit, 2**Nfermi)]

            rectangular matrix, contains stabilizer eigenstates
            written in the basis of the full qubit space
    
    Params: qLattice [spinlessQubitLattice object]
    '''
    
    X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
    sector = {1.0: 0,  -1.0: 1}


    #ndarray of joint stabilizer eigenstates 
    #(in 2-face-qubit subspace)
    eigenfaces = two_qubit_eigsectors()
    
    #########################
    qindices_A = [1,4,5,2]
    qindices_B = [3,6,7,4]
    SA, SB = X&Y, Y&X
    #########################

    Nfermi, Nqubit = qLattice._Nfermi, qLattice._Nsites
    code_dims = [2]*Nfermi
    qub_dims = [2]*Nqubit


    #parts of stabilizers acting on VERTEX qubits
    SA_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_A,
                    dims=code_dims)
    SB_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_B,
                        dims=code_dims)

    Uplus = np.ndarray(shape=(2**Nqubit, 2**Nfermi))


    for k in range(2**Nfermi):
        vertex_state = qu.basis_vec(i=k, dim=2**Nfermi)

        sgnA = qu.expec(SA_vert, vertex_state)
        sgnB = qu.expec(SB_vert, vertex_state)

        assert np.isclose(np.abs(sgnA), 1)
        assert np.isclose(np.abs(sgnB), 1)

        # face_state = eigfaces[sector[sgnA], sector[sgnB]]
        face_state = eigenfaces[sector[sgnA], sector[sgnB]]

        full_state = vertex_state & face_state
        Uplus[:,k] = full_state.flatten()

        #can i use kron like this?
        assert np.allclose((SA_vert&SA)@full_state, full_state)
        assert np.allclose((SB_vert&SB)@full_state, full_state)
    

    return qu.qu(Uplus)


## ***************************************** ##

def three_qubit_stabilizer(qLattice, qstabs=None):
    '''

    TODO: 
    *define SX_vert (vertices acted on by Z-stabilizer)

    '''
    sectors = {1.0: 0,  -1.0: 1}
    X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
    opmap = {'X': X, 'Y':Y, 'Z':Z, 'I':I}

    face_dims = [2]*3 #dimensions of face-qubit subspace


    if qstabs==None:

        qstab_A = qubitStabilizer(  vert_inds=[1,2,5,4],
                                    face_op='XYI',
                                    face_inds=[12,13,14])  

        qstab_B = qubitStabilizer(  vert_inds=[3,4,7,6],
                                    face_op='YXY',
                                    face_inds=[12,13,14])  
        
        qstab_C = qubitStabilizer(  vert_inds=[7,8,11,10],
                                    face_op='IYX',
                                    face_inds=[12,13,14])                                                                 
    

    SA = qu.kron(*[opmap[Q] for Q in qstab_A.face_op])
    SB = qu.kron(*[opmap[Q] for Q in qstab_B.face_op])
    SC = qu.kron(*[opmap[Q] for Q in qstab_C.face_op])
    # SA, SB, SC = X&Y&I, Y&X&Y, I&Y&X 

    eigfaces = np.ndarray(shape=face_dims, dtype=object)


    for signA, indA in sectors.items():

        eva, Ua = qu.eigh(SA)
        Ua_sector = Ua[:,eva==signA] #8x4

        Qb = Ua_sector.H @ SB @ Ua_sector
        # Qc = Ua_sector.H @ SC @ Ua_sector
        
        evb, Ub = qu.eigh(Qb)

        for signB, indB in sectors.items():

            Ub_sector = Ub[:, np.isclose(evb,signB)] #4x2

            #SC on b-eigenspace
            Qc = Ub_sector.H @ Ua_sector.H @ SC @ Ua_sector @ Ub_sector #2x2
            
            evc, Uc = qu.eigh(Qc)

            for signC, indC in sectors.items():
                
                #should be in a 1-dimensional subspace now
                vec = Uc[:,np.isclose(evc,signC)]

                #vector in 8-dim face qubit space
                face_vec = Ua_sector @ Ub_sector @ vec

                assert np.allclose(SA@face_vec, signA*face_vec)
                assert np.allclose(SB@face_vec, signB*face_vec)
                assert np.allclose(SC@face_vec, signC*face_vec)
                
                eigfaces[indA,indB,indC] = face_vec
    
    
    ###For testing!!
    return eigfaces

    Nfermi, Nqubit = qLattice._Nfermi, qLattice._Nsites
    code_dims = [2]*Nfermi #vertices 
    qub_dims = [2]*Nqubit #vertices&faces


    #######
    qindices_A = qstab_A.vert_inds
    qindices_B = qstab_B.vert_inds
    qindices_C = qstab_C.vert_inds
    #######

    #parts of stabilizers acting only on vertex qubits
    SA_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_A,
                    dims=code_dims)

    SB_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_B,
                        dims=code_dims)
    
    SC_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_C,
                        dims=code_dims)
                    
    Uplus = np.ndarray(shape=(2**Nqubit, 2**Nfermi))

    print('here')
    for k in range(2**Nfermi):
        print(k%10)
        #state of vertex qubits, all local z-eigenstates
        vertex_state = qu.basis_vec(i=k, dim=2**Nfermi)

        secA = sectors[qu.expec(SA_vert, vertex_state)]
        secB = sectors[qu.expec(SB_vert, vertex_state)]
        secC = sectors[qu.expec(SC_vert, vertex_state)]

        face_state = eigfaces[secA, secB, secC] 

        full_state = vertex_state&face_state

        #"stable" eigenstate written in qubit basis
        Uplus[:,k] = full_state.flatten()

        #full_state should be +1 eigenstate of all stabilizers
        assert np.allclose((SA_vert&SA) @ full_state, full_state)
        assert np.allclose((SB_vert&SB) @ full_state, full_state)
        assert np.allclose((SC_vert&SC) @ full_state, full_state)

    return qu.qu(Uplus)


