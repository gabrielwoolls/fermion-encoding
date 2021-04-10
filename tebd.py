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
    
    :

    '''
    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        chi=None,
        # imag=True,
        gate_opts=None,
        ordering=None,
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        # compute_energy_per_site=False,
        callback=None,
        keep_best=False,
        progbar=True,
        **kwargs,
    ):
        
        self.gate_opts = (
            dict() if gate_opts is None else
            dict(gate_opts))
        
        # important for `ePEPS.gate`, which allows 3-body gates
        self.gate_opts.setdefault('contract', 'auto_split')

        if ordering is None:
            pass # need to implement ham.get_auto_ordering
            
        super().__init__(psi0, ham, tau, D, chi, 
            imag=True, # force True, this is i-time EBD
            gate_opts=gate_opts,
            ordering=ordering,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            # force False, since 'num sites' is ambiguous for us?
            compute_energy_per_site=False,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
            **kwargs)




