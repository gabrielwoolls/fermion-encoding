import numpy as np
import quimb as qu

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
            
            face_state =  Ua_sgn @ Ub_sgn

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


    #ndarray of face-qubit stabilizers (XY, YX)
    #joint eigenstates (in 2-qubit subspace)
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

def three_qubit_stabilizer():
    '''
    TODO: 
    *define SX_vert (vertices acted on by Z-stabilizer)

    '''
    sectors = {1.0: 0,  -1.0: 1}
    X, Y, Z, I = (qu.pauli(mu) for mu in ['x','y','z','i'])
    opmap = {'X': X, 'Y':Y, 'Z':Z, 'I':I}

    face_dims = [2]*3 #dimensions of face-qubit subspace

    # ###
    # strA = 'XYI'
    # strB = 'YXY'
    # strC = 'IYX'
    # ###

    SA = qu.kron(*[opmap[Q] for Q in strA])
    SB = qu.kron(*[opmap[Q] for Q in strB])
    SC = qu.kron(*[opmap[Q] for Q in strC])
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
    
    
    Nfermion, Nqubit = qLattice._Nfermi, qLattice._Nsites
    code_dims = [2]*Nfermion #vertices 
    qub_dims = [2]*Nqubit #vertices&faces

    #parts of stabilizers acting only on vertex qubits
    SA_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_A,
                    dims=code_dims)

    SB_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_B,
                        dims=code_dims)
    
    SC_vert = qu.ikron(ops=[Z,Z,Z,Z], inds=qindices_C,
                        dims=code_dims)
                    
    Uplus = qu.qarray(shape=(2**Nqubit, 2**Nfermi))

    for k in range(2**Nvertex):
        
        #state of vertex qubits, all local z-eigenstates
        vertex_state = qu.basis_vec(i=k, dim=2**Nvertex)

        secA = sectors[qu.expec(SA_vert, vertex_state)]
        secB = sectors[qu.expec(SB_vert, vertex_state)]
        secC = sectors[qu.expec(SC_vert, vertex_state)]

        face_state = eigfaces[secA, secB, secC] 

        full_state = vertex_state&face_state

        #"stable" eigenstate written in qubit basis
        Uplus[:,k] = full_state.flatten()

        #full_state should be +1 eigenstate of all stabilizers
        assert np.allclose(SA_full@full_state, full_state)
        assert np.allclose(SB_full@full_state, full_state)
        assert np.allclose(SC_full@full_state, full_state)




