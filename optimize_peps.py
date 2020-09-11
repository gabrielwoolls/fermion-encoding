
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


import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer

LX, LY = 2, 2
DTYPE = 'complex128'

autodiff_backend = 'tensorflow'
autodiff_backend_opts = {'experimental_compile': True}


def state_energy(psi, hterms, vterms, **opts):
    
    he = psi.compute_local_expectation(
        hterms, normalized=True, **opts)

    ve = psi.compute_local_expectation(
        vterms, normalized=True, **opts)        
    
    return autoray.do('real', (he + ve))


def normalize_state(psi):
    return psi.normalize()


def main():
    Hij = qu.ham_heis(2).astype(DTYPE)

    peps = qtn.PEPS.rand(LX, LY, bond_dim=2, dtype=DTYPE)

    hterms = {coos: Hij for coos in peps.gen_horizontal_bond_coos()}
    vterms = {coos: Hij for coos in peps.gen_vertical_bond_coos()}

    compute_expec_opts = dict(
                    cutoff=0.0, 
                    max_bond=9, 
                    contract_optimize='random-greedy')
    
    optmzr = TNOptimizer(
        peps, 
        loss_fn=state_energy,
        # norm_fn=normalize_state,
        loss_constants={'hterms': hterms,
                        'vterms': vterms},
        loss_kwargs= compute_expec_opts,
        autodiff_backend=autodiff_backend,
        **autodiff_backend_opts)

    peps_opt = optmzr.optimize(10)
    return peps_opt



if __name__ == '__main__':
    main()
