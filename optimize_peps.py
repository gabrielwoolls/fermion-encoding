
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer

LX, LY = 2, 2
DTYPE = 'complex128'

def state_energy(psi, hterms, vterms, **opts):
    
    he = psi.compute_local_expectation(
        hterms, normalized=False, contract_optimize='random-greedy',**opts)

    ve = psi.compute_local_expectation(
        vterms, normalized=False, contract_optimize='random-greedy',**opts)        
    
    return he + ve


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
        norm_fn=normalize_state,
        loss_constants={'hterms': hterms,
                        'vterms': vterms},
        loss_kwargs=   {'opts': compute_expec_opts},
        autodiff_backend='tensorflow',
    )

    peps_opt = optmzr.optimize(1)
    return peps_opt



if __name__ == '__main__':
    main()
