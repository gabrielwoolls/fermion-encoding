import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import qubit_networks as my_qns
import quimb as qu
import pytest
import numpy as np
from random import randint
from itertools import starmap, product

class TestThreeBodyOps:
    '''Test methods for absorbing 3-body gates into a tn.
    '''
    @pytest.mark.parametrize('Lx,Ly', [(3,3), (3,4), (4,3)])
    @pytest.mark.parametrize('method', ['svd', 'qr', 'lq'])
    def test_qev_triangle_absorb_method(self, Lx, Ly, method):
        '''Test 'triangle_absorb' method, which works for 
        `QubitEncodeVector`-type geometry without rotating the 
        face tensors (i.e. non-rectangular lattice geometry).
        '''
        # opts for splitting 'blob' in `triangle_absorb` method
        compress_opts = {'method': method}

        # without rotating face tensors!
        psi = my_qns.QubitEncodeVector.rand(Lx, Ly, bond_dim=2)
        bra = psi.H
        # norm = (bra & psi) ^ all


        # compile all 3-tuples (vertex, vertex, face) qubits to act on
        LatticeHam = my_qns.SpinlessSimHam(Lx, Ly)
        for where, _ in LatticeHam.gen_ham_terms():
            
            # skip 2-body terms
            if len(where) == 2: 
                continue
            
            rand_gate = qu.rand_matrix(8)         

            # apply gate both 0) without contracting, and 1) with 'triangle-absorb'
            G_ket_0 = psi.apply_gate(G=rand_gate, where=where, contract=False)
            G_ket_1 = psi.apply_gate(G=rand_gate, where=where, contract='triangle_absorb', **compress_opts)

            expec_0 = (bra & G_ket_0) ^ all
            expec_1 = (bra & G_ket_1) ^ all
            assert expec_1 == pytest.approx(expec_0, rel=1e-2)
            
            ## also try with vertex qubits switched!
            where = (where[1], where[0], where[2])
            G_ket_0 = psi.apply_gate(G=rand_gate, where=where, contract=False)
            G_ket_2 = psi.apply_gate(G=rand_gate, where=where, contract='triangle_absorb', **compress_opts)
            
            expec_0 = (bra & G_ket_0) ^ all
            expec_2 = (bra & G_ket_2) ^ all
            assert expec_2 == pytest.approx(expec_0, rel=1e-2)
            
            
    @pytest.mark.parametrize('Lx,Ly', [(3,3), (3,4), (4,3)])
    @pytest.mark.parametrize('method', ['svd', 'qr', 'lq'])
    def test_epeps_absorb_3body_gate(self, Lx, Ly, method):
        '''Test absorbing a 3-body dense, random gate into the tn.
        '''
        
        # opts for splitting 'blob' in `triangle_absorb` method
        compress_opts = {'method': method}
        
        epeps = my_qns.QubitEncodeVector.rand(Lx, Ly).convert_to_ePEPS(dummy_size=1) 
        bra = epeps.H
        # norm = (bra & epeps) ^ all

        # use Ham to get (vertex, vertex, face) 'target' qubits
        LatticeCooHam = my_qns.SpinlessSimHam(Lx, Ly).\
            convert_to_coordinate_ham(lambda q: epeps.qubit_to_coo_map[q])
        
        for coos, _ in LatticeCooHam.gen_ham_terms():
            
            # skip 2-body interactions
            if len(coos) == 2:
                continue
            
            rand_gate = qu.rand_matrix(8) #use a random 3-body gate

            G_ket_0 = epeps.gate(G=rand_gate, coos=coos, contract=False)
            G_ket_1 = epeps.absorb_three_body_gate(G=rand_gate, coos=coos, 
                                            **compress_opts)

            expec_0 = (bra & G_ket_0) ^ all # `contract=False` reference
            expec_1 = (bra & G_ket_1) ^ all # `triangle_absorb` reference
            
            assert expec_1 == pytest.approx(expec_0, rel=1e-2)

            # now try with the vertex qubits swapped
            coos = (coos[1], coos[0], coos[2])
            G_ket_0 = epeps.gate(G=rand_gate, coos=coos, contract=False)
            G_ket_2 = epeps.absorb_three_body_gate(G=rand_gate, coos=coos, 
                                            **compress_opts)
            
            expec_0 = (bra & G_ket_0) ^ all 
            expec_2 = (bra & G_ket_2) ^ all 
            assert expec_2 == pytest.approx(expec_0, rel=1e-2)                                            


    @pytest.mark.parametrize('Lx,Ly', [(3,3), (3,4), (4,3)])
    @pytest.mark.parametrize('method', ['svd', 'qr', 'lq'])
    @pytest.mark.parametrize('tn_format', ['epeps', 'epeps_vec'])
    def test_epeps_vs_qev_absorb_gate(self, Lx, Ly, method, tn_format):
        '''Test that we get same result (for 3-body interactions)
        whether we evaluate with a ``QubitEncodeVector`` or the 
        2D-regular-lattice version ``ePEPS``.

        Methods tested:
        ---------------
            QubitEncodeVector.apply_gate(..., contract='triangle_absorb)
            ePEPS.absorb_three_body_gate(...)
        '''
        # opts for splitting 'blob' in `triangle_absorb` method
        compress_opts = {'method': method}

        # without rotating face tensors!
        psi = my_qns.QubitEncodeVector.rand(Lx, Ly, bond_dim=2)
        bra = psi.H

        # make into 'regular' 2D-square-lattice TN    
        if tn_format == 'epeps':
            psi_2d = psi.convert_to_ePEPS(dummy_size=1)
            
        elif tn_format == 'epeps_vec':
            psi_2d = psi.convert_to_ePEPS_vector(dummy_size=1)
        
        bra_2d = psi_2d.H
         
        Ham = my_qns.SpinlessSimHam(Lx, Ly)
        
        # LatticeCooHam = Ham.convert_to_coordinate_ham().\
        #     convert_to_coordinate_ham(psi_2d.qubit_to_coo_map)
        
        for where, _ in Ham.gen_ham_terms():

            # skip 2-body interactions
            if len(where) == 2:
                continue

            coos = [psi_2d.qubit_to_coo_map[q] for q in where]

            rand_gate = qu.rand_matrix(8) #use a random 3-body gate

            G_psi = psi.apply_gate(G=rand_gate, where=where, contract='triangle_absorb', **compress_opts)
            expec_QEV = (bra & G_psi) ^ all

            G_psi_2d = psi_2d.absorb_three_body_gate(G=rand_gate, coos=coos, **compress_opts)
            expec_ePEPS = (bra_2d & G_psi_2d) ^ all

            assert expec_ePEPS == pytest.approx(expec_QEV, rel=1e-2)

            # now try with vertex qubits swapped
            where = (where[1], where[0], where[2])
            coos = [psi_2d.qubit_to_coo_map[q] for q in where]

            # do the same test
            G_psi = psi.apply_gate(G=rand_gate, where=where, contract='triangle_absorb', **compress_opts)
            expec_QEV = (bra & G_psi) ^ all
            G_psi_2d = psi_2d.absorb_three_body_gate(G=rand_gate, coos=coos, **compress_opts)
            expec_ePEPS = (bra_2d & G_psi_2d) ^ all
            assert expec_ePEPS == pytest.approx(expec_QEV, rel=1e-2)

    

    @pytest.mark.parametrize('Lx,Ly', [(3,3), (3,4), (4,3)])
    @pytest.mark.parametrize('method', ['svd', 'qr', 'lq'])
    def test_epeps_absorb_3body_tensor(self, Lx, Ly, method):
        '''Currently only tests 3-body ops!

        contract_methods = {0: False,  1: 'triangle_absorb'}
        '''
        
        # opts for splitting 'blob' in `triangle_absorb` method
        compress_opts = {'method': method}
        
        epeps = my_qns.QubitEncodeVector.rand(Lx, Ly).convert_to_ePEPS(dummy_size=1) 
        bra = epeps.H


        # compile (vertex, vertex, face) 'target' qubits
        LatticeCooHam = my_qns.SpinlessSimHam(Lx, Ly).\
            convert_to_coordinate_ham(lambda q: epeps.qubit_to_coo_map[q])        

        for coos, _ in LatticeCooHam.gen_ham_terms():
            
            # SKIP 2-body interactions
            if len(coos) == 2:
                continue
            
            rand_gate = qu.rand_matrix(8) #use a random 3-body gate

            G_ket_0 = epeps.gate(G=rand_gate, coos=coos, contract=False)
            G_ket_1 = epeps.gate(G=rand_gate, coos=coos, contract='triangle_absorb', 
                                **compress_opts)

            expec_0 = (bra & G_ket_0) ^ all # `contract=False` reference
            expec_1 = (bra & G_ket_1) ^ all # `triangle_absorb` reference
            
            assert expec_1 == pytest.approx(expec_0, rel=1e-2)

            coos = (coos[1], coos[0], coos[2]) # now try w/ vertex qubits swapped

            G_ket_0 = epeps.gate(G=rand_gate, coos=coos, contract=False)
            G_ket_2 = epeps.gate(G=rand_gate, coos=coos, contract='triangle_absorb', 
                                    **compress_opts)
            
            expec_0 = (bra & G_ket_0) ^ all 
            expec_2 = (bra & G_ket_2) ^ all 
            assert expec_2 == pytest.approx(expec_0, rel=1e-2)  

    
    @pytest.mark.parametrize('Lx,Ly', [(2,3), (3,3), (3,4), (4,3)])
    @pytest.mark.parametrize('contract', ['reduce_split', 'reduce_split_lr'])
    def test_epeps_2body_reduce_split(self, Lx, Ly, contract):
        
        epeps = my_qns.QubitEncodeVector.rand(Lx, Ly).convert_to_ePEPS(dummy_size=1) 
        bra = epeps.H

        # compile (vertex, vertex) 'target' qubit pairs
        LatticeCooHam = my_qns.SpinlessSimHam(Lx, Ly).\
            convert_to_coordinate_ham(lambda q: epeps.qubit_to_coo_map[q])  

        for coos, _ in LatticeCooHam.gen_ham_terms():
            
            # skip 3-body terms here
            if len(coos) == 3:
                continue

            rand_gate = qu.rand_matrix(4)

            G_ket_0 = epeps.gate(G=rand_gate, coos=coos, contract=False)
            G_ket_1 = epeps.gate(G=rand_gate, coos=coos, contract=contract)

            assert (bra & G_ket_1) ^ all == pytest.approx(
                   (bra & G_ket_0) ^ all, rel=1e-2)


class TestEnergyContraction:

    @pytest.mark.parametrize('normalized', [True, False])
    @pytest.mark.parametrize('Lx,Ly', [(3,3), (4,3), (3,4)])
    def test_compute_local_expectation(self, Lx, Ly, normalized):
        
        # 'qubit' Fermi-Hubbard with default parameters
        t, V, mu = np.random.rand(3)
        H = my_qns.SpinlessSimHam(Lx, Ly, t, V, mu)
        
        psi_qev = my_qns.QubitEncodeVector.rand(Lx, Ly)\
            .setup_bmps_contraction_()
        norm, bra, ket = psi_qev.make_norm(return_all=True)
        norm ^= all        
        
        Exact = sum((
            (bra | ket.apply_gate(gate, where))^all
            for where, gate in H.gen_ham_terms()
        ))        

        if normalized:
            Exact /= norm
        
        # Now compute with `ePEPSvector.compute_local_expectation`

        epeps = psi_qev.convert_to_ePEPS_vector()

        q2coo = lambda q: epeps.qubit_to_coo_map[q]
        CooHam = H.convert_to_coordinate_ham(q2coo)
        terms = CooHam._coo_ham_terms

        envs, plaqmap = epeps.calc_plaquette_envs_and_map(terms)
        opts = dict(cutoff=2e-3, max_bond=9, 
            contract_optimize='random-greedy')

        e = epeps.compute_local_expectation(
            terms, normalized=normalized, autogroup=False, 
            plaquette_envs=envs, plaquette_map=plaqmap,
            **opts)

        assert e == pytest.approx(Exact, rel=1e-2)


    # Only tests small enough to do with dense vectors?
    @pytest.mark.parametrize('Lx,Ly', [(1,3), (3,1)])
    @pytest.mark.parametrize('normalized', [True, False])
    def test_compute_local_expectation_vs_dense(self, Lx, Ly, normalized):

        # (2Lx-1) * (2Ly-1) lattice ePEPSvector
        epeps = my_qns.QubitEncodeVector.rand(Lx, Ly)\
            .convert_to_ePEPS_vector()

        # qubits + dummies
        n_sites = (2*Lx-1) * (2*Ly-1)
        # vertices + occupied faces in 'original' Lx*Ly lattice
        n_qubits = (Lx * Ly) + int((Lx - 1) * (Ly - 1) / 2) 

        # 'qubit' Fermi-Hubbard with random parameters
        t, V, mu = np.random.rand(3)
        H = my_qns.SpinlessSimHam(Lx, Ly, t, V, mu)
        
        # separate indices of qubits from 'aux' tensors
        qubit_inds = tuple(starmap(epeps.site_ind, 
            (epeps.qubit_to_coo_map[q] for q in range(n_qubits))))
        non_qubit_inds = set(epeps.site_inds) - set(qubit_inds)

        # 'densify' into vector with qubit dimensions first
        psi_dense = epeps.to_dense((*qubit_inds, *non_qubit_inds))
        if normalized:
            qu.normalize(psi_dense)
        
        dense_terms = [qu.pkron(term, dims=[2]*n_sites, inds=where)
            for where, term in H.gen_ham_terms()]

        exact = sum((qu.expec(h, psi_dense) for h in dense_terms))
        
        # now compute energy with `ePEPSvector.compute_local_expectation`
        q2coo = lambda q: epeps.qubit_to_coo_map[q]
        CooHam = H.convert_to_coordinate_ham(q2coo)
        terms = CooHam._coo_ham_terms

        envs, plaqmap = epeps.calc_plaquette_envs_and_map(terms)
        opts = dict(cutoff=2e-3, max_bond=9, 
            contract_optimize='random-greedy')

        e = epeps.compute_local_expectation(
            terms, normalized=normalized, autogroup=False, 
            plaquette_envs=envs, plaquette_map=plaqmap,
            **opts)

        assert e == pytest.approx(exact, rel=1e-2)


class TestStabilizerEval:
    '''Remember for the even DK encoding to work we must have
    one of Lx, Ly odd (so Lx, Ly = (2,4) e.g. won't work)
    '''
    @pytest.mark.parametrize('Lx,Ly', [(1,3), (3,1), (2,3), (3,2)])
    def test_compute_stabilizer_expec(self, Lx, Ly):
        '''Use `compare_stabilizer_expecs` for testing
        '''
        #make the stabilizers
        Hstab = my_qns.HamStab(Lx=Lx, Ly=Ly)
        
        # random trial wavefunction (highly entangled)
        psi = my_qns.QubitEncodeVector.rand(Lx, Ly)

        # test stabilizer expecs working correctly
        compare_stabilizer_expecs(psi=psi, Hstab=Hstab)

        # NOTE: for lattices larger than 2x3, it seems expectations for a 
        # random state are so close to zero they are indistinguishable.
        # Need to test with non-random states
    

    @pytest.mark.parametrize('Lx,Ly', [(1,3), (2,3), (3,2), (3,3)])
    def test_stabilizer_expec_product_states(self, Lx, Ly):
        '''Use `compare_stabilizer_expecs` for testing
        '''
        # get the number qubits (vertices + faces)
        num_vertices = Lx * Ly
        num_faces = int((Lx-1) * (Ly-1) / 2)
        num_sites = num_vertices + num_faces

        # random product state configurations
        rand_bin_string = lambda: ''.join(
            str(randint(0,1)) for _ in range(num_sites))

        #make the stabilizers
        Hstab = my_qns.HamStab(Lx=Lx, Ly=Ly)
        
        #make the product state wavefunction
        psi = my_qns.QubitEncodeVector.product_state_from_bitstring(
            Lx=Lx, Ly=Ly, bitstring=rand_bin_string())

        compare_stabilizer_expecs(psi=psi, Hstab=Hstab)
        


def compare_stabilizer_expecs(psi, Hstab, get_expecs=False):
    '''Commodity function.
    Test ``psi.compute_stabilizer_expec`` method by 
    comparing to an "explicit" computation of the expectation
    values <psi|S|psi> for each stabilizer S.
    '''

    #get norm of state, exactly contracted
    norm = psi.make_norm() ^ all

    #(1) compute with method 'compute_stabilizer_expec'
    expecs_1 = []
    for S in Hstab.gen_ham_stabilizer_lists():
        x = psi.compute_stabilizer_expec(
            qubits=S[0], gates=S[1], 
            setup_bmps=True, norm=norm)

        expecs_1.append(x)
    

    ### check expectations are real ###
    assert np.allclose(np.imag(expecs_1), np.zeros(len(expecs_1)))


    #(2) compute explicitly
    expecs_2 = []

    psi.setup_bmps_contraction_()
    norm = psi.make_norm()^all

    bra = psi.copy().H
    bra.add_tag('BRA')

    boundary_contract_opts = {'layer_tags': ('KET', 'BRA')}

    for S in Hstab.gen_ham_stabilizer_lists():
        S_ket = psi.apply_stabilizer(qubits=S[0], gates=S[1],
                                inplace=False, contract=True)
        S_ket.add_tag('KET')
        x = (bra|S_ket).contract_boundary(**boundary_contract_opts)
        x_normalized = x / norm
        
        expecs_2.append(x_normalized)
    
    
    ### check expectations are correct ###
    assert np.allclose(np.real(expecs_1), np.real(expecs_2))

    if get_expecs:
        return np.real(expecs_1)


class TestConversionTo2D():

    @pytest.mark.parametrize('Lx,Ly', [(3,2), (2,3), (3,3), (3,4)])
    def test_transpose_tensors(self, Lx, Ly):
        psi = my_qns.QubitEncodeVector.rand_product_state(Lx, Ly)
        psi = psi.convert_to_tensor_network_2d() #relabel_physical_inds=False)
        
        shape = 'urdl'
        psi.transpose_tensors_to_shape(shape=shape)

        for (x,y), tag_xy in psi.gen_occupied_grid_tags(with_coo=True):
            
            tensor_xy, = [psi[tag_xy]]
            txy_inds = tensor_xy.inds

            array_order = shape
            if x == 0:
                array_order = array_order.replace('d', '')
            if x == psi.grid_Lx - 1:
                array_order = array_order.replace('u', '')
            if y == 0:
                array_order = array_order.replace('l', '')
            if y == psi.grid_Ly - 1:
                array_order = array_order.replace('r', '')

            num_physical_inds = len(tensor_xy.shape) - len(array_order)            
            assert num_physical_inds in (0, 1)

            coos_around = ((x-1, y),
                           (x, y+1),
                           (x+1, y),
                           (x, y-1))            

            tags_around = tuple(starmap(psi.grid_coo_tag, coos_around))
            dir_to_tag = dict(zip(('d','r','u','l'), tags_around))
            
            # check bonds are in correct order                
            for k, direction in enumerate(array_order):
                bond = psi.bond(where1=tag_xy, where2 = dir_to_tag[direction])
                assert bond == tensor_xy.inds[k]

    
    # def test_epeps_aux_tensors(self, Lx, Ly):
    #     '''Note ``Lx, Ly`` refers to the "underlying" lattice 
    #     geometry, not the ePEPS lattice (which has dimensions
    #     2*Lx-1, 2*Ly-1).
    #     '''
    #     epeps = my_qns.QubitEncodeVector.rand(Lx, Ly).convert_to_ePEPS(dummy_size=2) 

    #     for x, y in product(range(2 * Lx - 1), range(2 * Ly -1)):
    #         # tag_xy = epeps.site_tag_id.format(x,y)
    #         # tensor_xy = epeps[tag_xy]
    #         tensor_xy = epeps[x,y]
            
    #         if 'AUX' not in tensor_xy.tags:
    #             continue
            



class TestCoordinateHamiltonians():
    
    @pytest.mark.parametrize('Lx,Ly', [(3,2), (2,3), (3,3), (3,4)])
    def test_converting_hubbard_ham(self, Lx, Ly):
        '''Evaluate energy of spinless Hubbard model
        on a random state, using 'qubit numbers' and 
        qubit coordinates to specify target sites. 
        Check energies are equal.
        '''
        psi_trial = my_qns.QubitEncodeVector.rand(Lx, Ly, bond_dim=2)
        psi_trial.setup_bmps_contraction_()
        norm, bra, ket = psi_trial.make_norm(return_all=True)
        norm ^= all

        # encoded Fermi-Hubbard with default parameters
        HubbardHam = my_qns.SpinlessSimHam(Lx, Ly)
        
        # terms specify qubit coos rather than qubit nums
        CooHam = HubbardHam.convert_to_coordinate_ham(
            qubit_to_coo_map=psi_trial.qubit_to_coo_map)
        
        # energy computed with HubbardHam
        E1 = 0
        for where, gate in HubbardHam.gen_ham_terms():
            E1 += (bra | ket.apply_gate(gate, where, 
                    keys='qnumbers', contract=True))^all
        
        # energy computed with CooHam
        E2 = 0
        for coos, gate in CooHam.gen_ham_terms():
            E2 += (bra | ket.apply_gate(gate, where=coos,
                    keys='coos', contract=True))^all
        
        E1 = E1 / norm
        E2 = E2 / norm

        assert np.allclose(E1, E2)








