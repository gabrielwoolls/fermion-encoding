import autoray

def tf_qr(x):
    U, s, VH = autoray.do('linalg.svd', x)
    
    dtype = autoray.get_dtype_name(U)
    if 'complex' in dtype:
        s = autoray.astype(s, dtype)
    
    Q = U
    R = autoray.reshape(s, (-1, 1)) * VH
    
    return Q, R


autoray.register_function('tensorflow', 'linalg.qr', tf_qr)


import qubit_networks as beeky
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer

LX, LY = 3, 2
DTYPE = 'complex128'

autodiff_backend = 'tensorflow'
autodiff_backend_opts = {'experimental_compile': True}
# autodiff_backend_opts = dict()



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
        hterms, normalized=True, **opts)

    ve = psi.compute_local_expectation(
        vterms, normalized=True, **opts)        
    
    return autoray.do('real', (he + ve))


def normalize_state(psi: beeky.QubitEncodeVector):
    '''Set <psi|psi> to unity, only changing the 
    qubit tensors (leave identities alone).
    '''
    return psi.normalize_()


def main():

    # random initial guess
    psi0 = beeky.QubitEncodeVector.rand(Lx=LX, Ly=LY, bond_dim=2, dtype=DTYPE)

    # important so that boundary contraction works!
    psi0.setup_bmps_contraction_()

        
    HubSimHam = beeky.SpinlessSimHam(Lx=LX, Ly=LY)

    horizontal_terms = dict(HubSimHam.gen_horizontal_ham_terms())
    vertical_terms = dict(HubSimHam.gen_vertical_ham_terms())

    compute_expec_opts = dict(
                    cutoff=0.0, 
                    max_bond=9, 
                    contract_optimize='random-greedy')


    optmzr = TNOptimizer(
        psi0, # initial state guess
        loss_fn=state_energy,
        constant_tags=['AUX',],
        loss_constants={'hterms': horizontal_terms,
                        'vterms': vertical_terms},
        loss_kwargs=compute_expec_opts,
        autodiff_backend=autodiff_backend,
        **autodiff_backend_opts)

    tn_opt = optmzr.optimize(100)
    return tn_opt

if __name__ == '__main__':
    main()