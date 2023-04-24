'''Do TEBD on `ePEPS` and `ePEPSvector` -type TNs. The code
is based on quimb's TEBD2D code for 2D TNs, but modified to
allow 3-body interactions. 
'''
import quimb as qu
import quimb.tensor as qtn

class qubitTEBD(qtn.TEBD2D):
    '''Overrides `quimb` TEBD2D class to implement
    "encoded" Hamiltonians, which may have 3-body 
    operators in addition to 1- and 2-body.

    simulator_ham: ``CoordinateHamiltonian``
        Hamiltonian acting on qubit space
    
    stabilizer_ham: ``StabilizerHam``, optional

    check_fermion_stab: bool, optional
        Whether to compute <psi|stabilizers|psi>
        as we run TEBD.

    # compute_fermion_stab_every: int, optional
    '''
    def __init__(
        self,
        psi0,
        simulator_ham,
        stabilizer_ham=None,
        tau=0.01,
        D=None,
        chi=None,
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

        super().__init__(psi0, 
            ham=simulator_ham, 
            tau=tau, 
            D=D, 
            chi=chi, 
            imag=True, 
            ordering='random', # supplied to ham.get_auto_ordering
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


    def qubit_to_coo(self, q):
        '''Map qubit numbers to coordinates, e.g. 0 --> (0,0)
        '''
        return self.state.qubit_to_coo_map[q]


    def sweep(self, tau):
        '''Override ``TEBD2D.sweep`` to include Trotter
        gates of 'stabilizer' terms as well as energy terms.

        If applying exp(tau * stabilizer) gates: to make sure we target
        the "codespace" i.e. eigenvalue +1 space, Trotterize with 
            
            tau -> tau * stab_ham.multiplier, 
        
        multiplier +1 (stabilizers unchanged) ==> supress -1 eigenspace
        '''
        
        # apply energy trotter gates as usual
        super().sweep(tau) 

        if self.stabilizer_ham is None:
            return # exit if there's no stabilizers to apply
        
        # otherwise, apply the exp[stabilizer] gates

        stab_ham = self.stabilizer_ham
        
        for qubits, _ in stab_ham.gen_ham_stabilizer_lists():
            coos = map(self.qubit_to_coo, qubits)
            U = stab_ham.get_gate_expm(qubits, stab_ham.multiplier * tau)
            self.gate(U, coos)


    
def compute_energy_func_epeps(tebd):
    '''Supplied as ``qubitTEBD.compute_energy_fn``.
    Compute the energy of ``tebd.state``, assuming it's
    an ``ePEPS``-type TN (i.e. inherits from quimb's
    TensorNetwork2D class). 
    '''
    psi, ham_terms = tebd.state, tebd.ham.terms
    envs, plaqmap = psi.calc_plaquette_envs_and_map(
        terms=ham_terms)
    
    return psi.compute_local_expectation(terms=ham_terms, 
        plaquette_envs=envs, plaquette_map=plaqmap, 
        **tebd.compute_energy_opts)


def compute_energy_func_qev(tebd):
    '''Compute the energy of ``tebd.state``, assuming it's
    a ``QubitEncodeVector``-type TN. Converts the TN to
    an ePEPS, to use quimb's native boundary contraction
    method.
    '''
    psi_qev, ham_terms = tebd.state, tebd.ham.terms
    epeps = psi_qev.convert_to_ePEPS()

    # proceed with epeps energy contraction
    envs, plaqmap = epeps.calc_plaquette_envs_and_map(
        terms=ham_terms)
    
    return epeps.compute_local_expectation(terms=ham_terms, 
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
        




