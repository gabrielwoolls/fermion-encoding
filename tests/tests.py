import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import qubit_networks as my_qns
import pytest
import numpy as np

class TestStability:
    '''Remember for the even DK encoding to work we must have
    one of Lx, Ly odd (so Lx, Ly = (2,4) e.g. won't work)
    '''
    @pytest.mark.parametrize('Lx', [2,])
    @pytest.mark.parametrize('Ly', [3,])
    def test_compute_stabilizer_expec(self, Lx, Ly):
        
        #make the stabilizers
        Hstab = my_qns.HamStab(Lx=Lx, Ly=Ly)
        
        #take a random trial state
        psi = my_qns.QubitEncodeVector.rand(Lx, Ly)

        #get norm of state, 'exactly' contracted
        norm = psi.make_norm()^all

        #(1) compute with method 'compute_stabilizer_expec'
        expecs_1 = []
        for S in Hstab.gen_ham_stabilizer_lists():
            x = psi.compute_stabilizer_expec(
                qubits=S[0], gates=S[1], 
                setup_bmps=True, norm=norm)

            expecs_1.append(x)
        

        # check expectations are real
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
        
        
        # check expectations are correct
        assert np.allclose(np.real(expecs_1), np.real(expecs_2))

        # NOTE: for lattices larger than 2x3, it seems expectations for a 
        # random state are so close to zero they are indistinguishable.
        # Need to test with non-random states


        





