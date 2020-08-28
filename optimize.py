import qubitNetworks as beeky

import quimb as qu
import quimb.tensor as qtn
from itertools import product, chain, starmap, cycle, combinations
import denseQubits
import functools
from quimb.tensor.tensor_core import tags_to_oset
from quimb.utils import pairwise, check_opt, oset
import opt_einsum as oe
from operator import add

from quimb.tensor.optimize import TNOptimizer

LX, LY = 3, 3
HubSimHam = beeky.SpinlessSimHam(Lx=LX, Ly=LY)

horizontal_terms = dict(HubSimHam.gen_horizontal_ham_terms())
vertical_terms = dict(HubSimHam.gen_vertical_ham_terms())

compute_expec_opts = dict(
                cutoff=2e-3, 
                max_bond=9, 
                contract_optimize='random-greedy'
                )

def state_energy(
    psi: beeky.QubitEncodeVector,
    hterms: dict,
    vterms: dict,
    **opts):
    '''Energy <psi|H|psi>, summing contribns from
    horiz/vertical terms in Hamiltonian.
    '''

    # TODO: compute row/col envs first?
    he = psi.compute_local_expectation(
        hterms, normalized=False, **opts)

    ve = psi.compute_local_expectation(
        vterms, normalized=False, **opts)        
    
    return he + ve

def normalize_state(psi: beeky.QubitEncodeVector):
    '''Set <psi|psi> to unity, only changing the 
    qubit tensors (leave identities alone).
    '''
    # inplace normalize
    return psi.normalize_()


def main():

    # random initial guess
    psi0 = beeky.QubitEncodeVector.rand(Lx=LX, Ly=LY, bond_dim=3)

    # important so that boundary contraction works!
    psi0.setup_bmps_contraction_()

    optmzr = TNOptimizer(
        psi0, # initial state guess
        loss_fn=state_energy,
        norm_fn=normalize_state,
        constant_tags=('AUX',),
        loss_constants={'hterms': horizontal_terms,
                    'vterms': vertical_terms,
                    'opts': compute_expec_opts},
        autodiff_backend='tensorflow',
    )

    tn_opt = optmzr.optimize(100)
    return tn_opt

if __name__ == '__main__':
    main()