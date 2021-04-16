'''Do TEBD on `ePEPS` and `ePEPSvector` -type TNs. The code
is based on Johnnie Gray's quimb TEBD code for 2D TNs, but 
needs to be modified to allow 3-body interactions. 
'''
import quimb as qu
import quimb.tensor as qtn

class qubitTEBD(qtn.TEBD2D):
    '''Overrides `quimb` TEBD2D class to implement
    "encoded" Hamiltonians, which may have 3-body 
    operators in addition to 1- and 2-body.

    stabilizer_ham: ``HamStab``, optional

    check_fermion_stab: bool, optional
    compute_fermion_stab_every: int, optional
    '''
    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        chi=None,
        stabilizer_ham=None,
        # imag=True,
        gate_opts=None,
        # ordering=None,
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        check_fermion_stab=False, #
        compute_fermion_stab_every=None,
        # compute_energy_fn=None,
        # compute_energy_per_site=False,
        callback=None,
        keep_best=False,
        progbar=True,
        **kwargs,
    ):
        # code-stabilizer "hamiltonian"
        self.stabilizer_ham = stabilizer_ham 

        gate_opts = (
            dict() if gate_opts is None else
            dict(gate_opts))
        
        # important for `ePEPS.gate`, which allows 3-body gates
        gate_opts.setdefault('contract', 'auto_split')
        gate_opts.setdefault('method', 'svd')

        super().__init__(psi0, ham, tau, D, chi, 
            imag=True, # force True, this is i-time EBD
            ordering='random', #passed to ham.get_auto_ordering
            gate_opts=gate_opts,
            # force False, since 'num sites' is ambiguous for us?
            compute_energy_per_site=False,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            # compute_energy_fn=compute_energy_fn,
            compute_energy_fn=compute_energy_func_epeps,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
            **kwargs)
        
    
def compute_energy_func_epeps(tebd):
    '''Supplied as ``qubitTEBD.compute_energy_fn``
    '''
    psi, ham_terms = tebd.state, tebd.ham.terms
    envs, plaqmap = psi.calc_plaquette_envs_and_map(
        terms=ham_terms)
    
    return psi.compute_local_expectation(terms=ham_terms, 
        plaquette_envs=envs, plaquette_map=plaqmap, 
        **tebd.compute_energy_opts)


def compute_fermionic_stability(tebd):
    '''Uses ``tebd.stabilizer_ham`` to compute the 
    fermionic "stability" of the current state. 
    
    For each stabilizer operator ``S`` in ``stabilizer_ham``, 
    compute and store the (normalized) expectation 
    (<psi|S|psi> / norm), where psi is ``tebd.state``.
    '''
    
    stab_ham = tebd.stabilizer_ham

    ket = tebd.state.copy()
    ket.add_tag('KET')
    bra = ket.retag({'KET': 'BRA'})
    bra.conj_('*')

    contract_opts = {'layer_tags': ('BRA','KET'),
        'max_bond': tebd.chi, 'cutoff': 0.0}

    norm = (bra | ket).contract_boundary(**contract_opts)

    stab_expecs = []
    for s in stab_ham.gen_ham_stabilizer_lists():
        
        qubits, gates = s # each gate acts on just 1 qubit
        coos = (ket.qubit_to_coo_map[q] for q in qubits)

        # compute S|ket> = (stabilizer)|psi>
        s_ket = ket.copy()
        for c, g in zip(coos, gates):
            s_ket.gate_(G=g, coos=c, contract=True) 
        
        # compute <bra|S|ket>
        expec = (bra | s_ket).contract_boundary(
            **contract_opts)
        stab_expecs.append(expec / norm)
    
    return stab_expecs
        
        
# def sweep_and_stabilizers(tebd, tau):
    
#     # usual sweep with exponentiated tebd.ham.terms
#     # i.e. energy trotter gates
#     tebd.sweep(tau) 

#     # sweep with stabilizer trotter gates
#     stab_ham = tebd.stabilizer_ham
#     for qubits, _ in stab_ham.gen_ham_stabilizer_lists():
#         coos = (tebd.state.qubit_to_coo_map[q] for q in qubits)
#         U = stab_ham.get_gate_expm(qubits, -tau)
#         tebd.gate(U, where)



    
        




