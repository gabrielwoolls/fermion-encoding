import quimb as qu
import quimb.tensor as qtn
import dense_qubits
import functools
from collections import defaultdict


def number_op():
    '''Fermionic number operator, aka
    qubit spin-down projector
    '''
    return qu.qu([[0, 0], [0, 1]])


class CoordinateHamiltonian():
    '''Wrapper class for previously-defined Hamiltonians.
    
    If `Ham` previously generated terms like 
        
        (sequence of qubit numbers, gate),  e.g. 
        ([0, 1, 9] , pauli(XYX))
    
    the equivalent `CoordinateHamiltonian` will generate terms like

        (sequence of qubit coordinates, gate) e.g. 
        ([(4,0), (4,2), (3,1)], pauli(XYX))

    Obviously depends on a coordinate-choice! Will need to specify a 
    mapping from qubit numbers `q = 0, 1, ... M` to lattice coordinates
    `(x0, y0), (x1, y1), ... (xM, yM)`. 

    Attributes:
    -----------
    '''
    def __init__(self, coo_ham_terms, qubit_to_coo_map):

        self._qubit_to_coo_map = qubit_to_coo_map
        self._coo_ham_terms = coo_ham_terms

    @property
    def terms(self):
        return self._coo_ham_terms

    def gen_ham_terms(self):
        return iter(self._coo_ham_terms.items())

    def get_term_at_sites(self, *coos):
        return self._coo_ham_terms[tuple(coos)]
    
    def get_auto_ordering(self, order='sort'):
        """Get an ordering of the terms to use with TEBD, for example. The
        default is to sort the coordinates then greedily group them into
        commuting sets.

        Parameters
        ----------
        order : {'sort', None, 'random', str}
        See ``quimb.tensor.LocalHam2D.get_auto_ordering``

        Returns
        -------
        list[tuple[tuple[int]]]
            Sequence of coordinate pairs.
        """
        if order is None:
            pairs = self.terms
        elif order == 'sort':
            pairs = sorted(self.terms)
        elif order == 'random':
            pairs = list(self.terms)
            random.shuffle(pairs)

        # x can be length 2 or 3
        pairs = {x: None for x in pairs}

        cover = set()
        ordering = list()
        while pairs:
            for pair in tuple(pairs):
                if all((ij not in cover for ij in pair)):
                    ordering.append(pair)
                    pairs.pop(pair)
                    for ij in pair:
                        cover.add(ij)
            cover.clear()

        return ordering


class MasterHam():
    '''Commodity class to combine a simulator Ham `Hsim`
    and a stabilizer pseudo-Ham `Hstab` to generate each of 
    their gates in order.

    Attributes:
    ----------
    `gen_ham_terms()`: generate Hsim terms followed by Hstab terms

    `gen_trotter_gates(tau)`: trotter gates for Hsim followed by Hstab
    '''
    def __init__(self, Hsim, Hstab):
        self.Hsim=Hsim
        self.Hstab = Hstab

    def gen_ham_terms(self):
        return chain(self.Hsim.gen_ham_terms(),
                     self.Hstab.gen_ham_terms())

    def gen_trotter_gates(self, tau):
        return chain(self.Hsim.gen_trotter_gates(tau),
                     self.Hstab.gen_trotter_gates(tau))


### *********************** ###

class HamStab():
    '''Pseudo-Hamiltonian of stabilizers,
        
            H_stab = multiplier * (S1 + S2 + ... + Sk), 
        
        i.e. sum of all the stabilizers multiplied by ``multiplier``.

        Stores 8-site gates corresponding to the loop
        stabilizer operators, intending to be added to a simulator
        Hamiltonian `H_sim` for TEBD.
        '''

    def __init__(self, Lx, Ly, multiplier = -1.0):
        
        
        self.qlattice = dense_qubits.QubitLattice(Lx, Ly, local_dim=0)

        self.multiplier = multiplier
        
        #map coos to `loopStabOperator` objects
        coo_stab_map = self.qlattice.make_coo_stabilizer_map()

        self._stab_gates = self.make_stab_gate_map(coo_stab_map, store='gate')
        self._exp_stab_gates = dict()
        self._stab_lists = self.make_stab_gate_map(coo_stab_map, store='tuple')



    def make_stab_gate_map(self, coo_stab_map, store='gate'):
        '''TODO: NOW MAPS coos to (where, gate) tuples.

        Return
        -------
        `gate_map`: dict[tuple : (tuple, qarray)] 
            Maps coordinates (x,y) in the *face* array (empty 
            faces!) to pairs (where, gate) that specify the 
            stabilizer gate and the sites to be acted on.

        Param:
        ------
        coo_stab_map: dict[tuple : dict]
            Maps coordinates (x,y) in the face array of the lattice
            to `loop_stab` dictionaries of the form
            {'inds' : (indices),   'opstring' : (string)}
        
        store: 'gate' or 'tuple', optional
            Whether to store a 'dense' 2**8 x 2**8 array or a tuple
            of 8 (ordered) 2 x 2 arrays
        '''
        gate_map = dict()

        for coo, loop_stab in coo_stab_map.items():
            #tuple e.g. (1,2,4,5,6)
            qubits = tuple(loop_stab['inds'])

            #string e.g. 'ZZZZX'
            opstring = loop_stab['opstring']
            
            if store == 'gate':            
                #qarray
                gate = qu.kron(*(qu.pauli(Q) for Q in opstring))
                gate *= self.multiplier
                gate_map[coo] = (qubits, gate)
            
            elif store == 'tuple':

                signs = [self.multiplier] + [1.0] * (len(opstring) - 1)
                gates = tuple(signs[k] * qu.pauli(Q) for k, Q in enumerate(opstring))
                gate_map[coo] = (qubits, gates)
            
        
        return gate_map



    def reset_multiplier(self, multiplier):
        '''Reset multiplier, re-compute stabilizer gates 
        and erase previous expm(gates)
        '''
        self.multiplier = multiplier
        self._stab_gates = self.make_stab_gate_map(coo_stab_map)
        self._exp_stab_gates = dict()


    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for acting with 
        the 8-site stabilizer gates on specified sites.

        Note the *keys* in the dictionary are coos and not used!
        '''
        for where, gate in self._stab_gates.values():
            yield (where, gate)

    def gen_ham_stabilizer_lists(self):
        '''Generate ``(where, gates)`` pairs for each stabilizer term.
        where: tuple[int]
            The qubits to be acted on
        gates: tuple[array]
            The one-site qubit gates (2x2 arrays)
        '''
        for where, gatelist in self._stab_lists.values():
            yield (where, gatelist)
    
    def _get_exp_stab_gate(self, coo, tau):
        '''
        Returns 
        -------
        `exp(tau * multiplier * stabilizer)`: qarray
        
                Expm() of stabilizer centered on empty face at `coo`.

        Params
        -------
        `coo`: tuple (x,y)
            (Irrelevant) location of the empty face that 
            the stabilizer corresponds to. Just a label.
        
        `tau`: float
            Imaginary time for the exp(tau * gate)
        '''
        key = (coo, tau)

        if key not in self._exp_stab_gates:
            where, gate = self._stab_gates[coo]
            el, ev = do('linalg.eigh', gate)
            expgate = ev @ do('diag', do('exp', el*tau)) @ dag(ev)
            self._exp_stab_gates[key] = (where, expgate)
        
        return self._exp_stab_gates[key]


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau*gate)`) pairs for acting
        with exponentiated stabilizers on lattice.
        '''
        for coo in self.empty_face_coos():
            yield self._get_exp_stab_gate(coo, tau)


    def empty_face_coos(self):
        return self._stab_gates.keys()


###  **********************************  ###


class SimulatorHam():
    '''Generic class for simulator 
    (i.e. qubit-space) Hamiltonians. 

    Takes a `qlattice` object to handle lattice geometry/edges, 
    and a mapping `ham_terms` of edges to two/three site gates.
    '''
    
    def __init__(self, Lx, Ly, phys_dim, ham_terms):
        '''
        Lx: number of vertex qubit rows
        Ly: number vertex qubit columns                 
                 
                :       :           
          x+1  ─●───────●─   < vertex row
                │ \   / │           
                │   ●   │    < face row
                │ /   \ │           
           x   ─●───────●─   < vertex row
                :       :           

        => Total number of vertices = Lx * Ly        

        phys_dim: int
            Local site dimension (d=2 for 
            simple qubits)
        
        ham_terms: dict{tuple[int]: qarray}
            Mapping of qubit numbers (integer 
            labels) to raw operators.

        '''
        self._Lx = Lx
        self._Ly = Ly
        self._phys_dim = phys_dim

        self._ham_terms = ham_terms
        self._exp_gates = dict()

        
    def get_gate(self, edge):
        '''Local term corresponding to `edge`
        '''
        return self._ham_terms[edge]


    def get_expm_gate(self, edge, t):
        '''Local term for `edge`, matrix-exponentiated
        by `t`.
        '''
        # return self._expm_cached(self.get_gate(edge), x)
        key = (edge, t)
        if key not in self._exp_gates:
            gate = self.get_gate(edge)
            el, ev = do('linalg.eigh',gate)
            self._exp_gates[key] = ev @ do('diag', do('exp', el*t)) @ dag(ev)
        return self._exp_gates[key]

    

    def _trotter_gate_group(self, group, x):
        '''Returns mapping of edges (in ``group``) to
        the corresponding exponentiated gates.
        
        Returns: dict[edge : exp(Ham gate)]
        '''
        edges = self.get_edges(group)
        gate_map = {edge : self.get_expm_gate(edge,x) for edge in edges}
        return gate_map
    

    #TODO: add _ internal method
    def get_edges(self, which):
        '''Retrieves (selected) edges from internal 
        qlattice object.
        '''
        return self.qlattice.get_edges(which)

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly


    def ham_params(self):
        '''Relevant Hamiltonian parameters.
        '''
        pass


    def gen_ham_terms(self):
        pass
    
    def gen_horizontal_ham_terms(self):
        pass
    
    def gen_vertical_ham_terms(self):
        pass

    def gen_trotter_gates(self, tau):
        pass


## ******************* ##
# Subclass Hamiltonians
## ******************* ##

class SpinlessSimHam(SimulatorHam):
    '''Qubit Hamiltonian: spinless fermion Hubbard Ham,
    encoded as a qubit simulator Ham.

    H =   t  * hopping
        + V  * repulsion
        - mu * occupation
    '''

    def __init__(self, Lx, Ly, t=1.0, V=1.0, mu=0.5):
        '''
        Lx: num vertex qubit rows
        Ly: num vertex qubit columns

        t: hopping parameter
        V: nearest-neighbor repulsion
        mu: single-site chemical potential
        '''
        # Hubbard parameters
        self._t = t
        self._V = V
        self._mu = mu

        # to handle the fermion-to-qubit encoding & lattice geometry
        self.qlattice = dense_qubits.QubitLattice(Lx=Lx, Ly=Ly, local_dim=0)
        
        terms = self._make_ham_terms()

        super().__init__(Lx=Lx, Ly=Ly, phys_dim=0, ham_terms=terms)


    def convert_to_coordinate_ham(self, qubit_to_coo_map):
        '''Switch the {qubits: gate} dict for a 
        {coordinates: gate} dict by mapping all the target 
        qubits to their lattice coordinates.

        qubit_to_coo_map: callable, int --> tuple[int]
            Map each qubit number to the corresponding
            lattice coordinate (x, y)

        Returns:
        -------
        Equivalent `CoordinateHamiltonian` object. 
        '''
        mapped_ham_terms = dict()
        for (i,j,f), gate in self._ham_terms.items():
            qubits = (i,j) if f is None else (i,j,f)
            qcoos = tuple(map(qubit_to_coo_map, qubits))
            mapped_ham_terms.update({qcoos: gate})
    
        return CoordinateHamiltonian(coo_ham_terms=mapped_ham_terms,
                qubit_to_coo_map = qubit_to_coo_map)


    def get_term_at(self, i, j, f=None):
        '''Array acting on edge `(i,j,f)`.
        `i,j` are vertex sites, optional `f` is 
        the face site.
        '''
        return self._ham_terms[(i,j,f)]


    def ham_params(self):
        '''Ham coupling constants

        t: hopping parameter,
        V: nearest-neighbor repulsion,
        mu: chemical potential
        '''
        return (self._t, self._V, self._mu)


    @property
    def terms(self):
        '''To match `qtn.LocalHam2D.terms` property. 
        '''
        return dict(self.gen_ham_terms())


    def gen_ham_terms(self):
        '''Generate ``(where, gate)`` pairs for every group 
        of qubits (i.e. every graph edge ``where``) to be acted
        on with a Ham term. Drops any `None` empty qubits.
        '''
        for (i,j,f), gate in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, gate


    def gen_horizontal_ham_terms(self):
        '''Only those terms in the Hamiltonian acting on horizontal
        graph edges, i.e. the ``where`` qubit sites must correspond
        to a *horizontal* (vertex, vertex, face or None).
        '''

        for (i, j, f) in self.get_edges(which='horizontal'):
            gate = self.get_term_at(i, j, f)
            where = (i, j) if f is None else (i, j, f)
            yield where, gate
    
    def gen_vertical_ham_terms(self):
    
        for (i, j, f) in self.get_edges(which='vertical'):
            gate = self.get_term_at(i, j, f)
            where = (i, j) if f is None else (i, j, f)
            yield where, gate
    


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau * gate)`) pairs, for each 
        location (edge) `where` and gate exponentiated by
        `tau`. 

        Generated in ordered groups of edges:
        1. Horizontal-even
        2. Horizontal-odd
        3. Vertical-even
        4. Vertical-odd
        '''
        for group in ['he', 'ho', 've', 'vo']:

            for edge, exp_gate in self._trotter_gate_group(group, tau).items():
                
                i,j,f = edge
                where = (i,j) if f is None else (i,j,f)
                yield where, exp_gate


    def _make_ham_terms(self):
        '''Get all terms in Ham as two/three-site gates, 
        in a dict mapping edges to qarrays.
        
        Returns:  dict{edge (i,j,f): gate (qarray)}

        If `f` is None, the corresponding gate will be two-site 
        (vertices only). Otherwise, gate acts on three sites.
        '''
        t, V, mu = self.ham_params()
        qlattice = self.qlattice

        terms = dict()

        #vertical edges
        for direction, sign in (('down', 1), ('up', -1)):

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] = sign * t * self._two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())
                
                #three-site
                else:
                    terms[(i,j,f)] = sign * t * self._three_site_hop_gate(edge_dir='vertical')
                    terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('horizontal'):

            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_gate()
                    terms[(i,j,f)] += V * (number_op()&number_op())

            #three-site    
            else:
                terms[(i,j,f)] =  t * self._three_site_hop_gate(edge_dir='horizontal')
                terms[(i,j,f)] += V * (number_op()&number_op()&qu.eye(2))

        
        if mu == 0.0:
            return terms


        n_op = number_op() #one-site number operator 

        # map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))


        #for each vertex in lattice, absorb chemical potential term
        #uniformly into the edge terms that include it
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(edges)

            #should appear in at least two edge terms!
            if not (num_edges > 1 or qlattice.num_faces == 0):
                raise ValueError("Something's wrong")


            for (i,j,f) in edges:

                v_place = (i,j,f).index(vertex) #vertex is either i or j

                if f is None: #ham_term should act on two sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*2, inds=v_place)

                else: #act on three sites
                    terms[(i,j,f)] -= mu * (1/num_edges) * qu.ikron(n_op, dims=[2]*3, inds=v_place)

        return terms


    def _two_site_hop_gate(self):
        '''Hopping between two vertices, with no face site.
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        return 0.5* ((X & X) + (Y & Y))


    def _three_site_hop_gate(self, edge_dir):
        '''Hop gate acting on two vertices and a face site.
        '''
        X, Y = (qu.pauli(mu) for mu in ['x','y'])
        O_face = {'vertical': X, 'horizontal':Y} [edge_dir]

        return 0.5 * ((X & X & O_face) + (Y & Y & O_face))
        


################################

class MPOSpinlessHam():

    def __init__(self, qlattice, t, V, mu):

        self._t = t
        self._V = V
        self._mu = mu

        self._site_tag_id = 'Q{}'
        self._ham_terms = self._make_ham_terms(qlattice)


    def ham_params(self):
        return (self._t, self._V, self._mu)
    

    def get_term_at(self, i, j, f=None):
        '''MPO acting on edge `(i,j,f)`.
        `i,j` are vertex sites, optional `f` is 
        the face site.
        '''
        return self._ham_terms[(i,j,f)]
    

    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for every location
        (i.e. edge) to be acted on with a Ham term
        '''
        for (i,j,f), mpo in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, mpo


    def _make_ham_terms(self, qlattice):
        
        site_tag = self._site_tag_id
        t, V, mu = self.ham_params()

        terms = dict()
        
        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    mpo = sign * t * self._two_site_hop_mpo()
                    mpo += V * self._two_site_nnint_mpo()
                    terms[(i,j,f)] = mpo
                
                #three-site
                else:
                    mpo = sign * t * self._three_site_hop_mpo(edge_dir='vertical')
                    mpo += V * self._three_site_nnint_mpo()
                    terms[(i,j,f)] = mpo

        
        #horizontal edges
        for (i,j,f) in qlattice.get_edges('horizontal'):

            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_mpo()
                    terms[(i,j,f)] += V * self._two_site_nnint_mpo()

            #three-site    
            else:
                terms[(i,j,f)] =  t * self._three_site_hop_mpo(edge_dir='horizontal')
                terms[(i,j,f)] += V * self._three_site_nnint_mpo()

        

        if mu == 0.0:
            return terms


        n_op = number_op() #one-site number operator 
        Ident = qu.eye(2)

        #map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))

        
        mpo_NI = qtn.MatrixProductOperator(arrays=[n_op.reshape(2,2,1),
                                                  Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)

        mpo_IN = qtn.MatrixProductOperator(arrays=[Ident.reshape(2,2,1),
                                                   n_op.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        mpo_NII = qtn.MatrixProductOperator(arrays=[n_op.reshape(2,2,1),
                                                    Ident.reshape(1,2,2,1),
                                                    Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        mpo_INI = qtn.MatrixProductOperator(arrays=[Ident.reshape(2,2,1),
                                                   n_op.reshape(1,2,2,1),
                                                   Ident.reshape(1,2,2)],
                                            shape='ludr',
                                            site_tag_id=site_tag)


        two_site_mpos = (mpo_NI, mpo_IN)
        three_site_mpos = (mpo_NII, mpo_INI)

        #for each vertex in lattice, absorb occupation term
        #evenly into the edge terms that include it
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            cover_edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(cover_edges)

            assert num_edges>1 or qlattice.num_faces()==0 #should appear in at least two edge terms!


            for (i,j,f) in cover_edges:
                
                #Number op acts on either the i or j vertex (first or second site)
                which_vertex = (i,j,f).index(vertex) #`which` can be 0 or 1


                #no face, so use 2-site MPO
                if f is None: 
                    #choose either NI or IN
                    terms[(i,j,f)] -= mu * (1/num_edges) * two_site_mpos[which_vertex]
                                
                else: #include face, use 3-site MPO
                    #choose either NII or INI
                    terms[(i,j,f)] -= mu * (1/num_edges) * three_site_mpos[which_vertex]

        return terms


    def _two_site_hop_mpo(self):
        ''' 0.5 * (XX + YY)
        '''
        X, Y = (qu.pauli(q) for q in ['x','y'])
        site_tag = self._site_tag_id
    
        mpoXX = qtn.MatrixProductOperator(arrays=[X.reshape(2,2,1), 
                                                  X.reshape(1,2,2)], 
                                        shape='ludr',
                                        site_tag_id=site_tag)
        
        mpoYY = qtn.MatrixProductOperator(arrays=[Y.reshape(2,2,1), 
                                                  Y.reshape(1,2,2)], 
                                        shape='ludr',
                                        site_tag_id=site_tag)
        
        return (mpoXX + mpoYY) / 2


    def _three_site_hop_mpo(self, edge_dir):
        '''(XXO + YYO)/2
        '''
        X, Y = (qu.pauli(q) for q in ['x','y'])
        O = {'vertical': X, 'horizontal':Y} [edge_dir]
        site_tag = self._site_tag_id

        mpo_XXO = qtn.MatrixProductOperator(arrays=[X.reshape(2,2,1), 
                                                    X.reshape(1,2,2,1),
                                                    O.reshape(1,2,2)], 
                                            shape='ludr',
                                            site_tag_id=site_tag)

        mpo_YYO = qtn.MatrixProductOperator(arrays=[Y.reshape(2,2,1), 
                                                    Y.reshape(1,2,2,1),
                                                    O.reshape(1,2,2)], 
                                            shape='ludr',
                                            site_tag_id=site_tag)

        return (mpo_XXO + mpo_YYO) / 2                                                    


    def _two_site_nnint_mpo(self, third_site_identity=False):
        '''Two-site nearest-neighbor interaction, optionally padded
        with an identity to act on a third site.
        '''
        Nop = number_op()
        site_tag = self._site_tag_id

        if third_site_identity:
            oplist = [Nop.reshape(2,2,1), 
                      Nop.reshape(1,2,2,1), 
                      qu.eye(2).reshape(1,2,2)]
        
        else:
            oplist = [Nop.reshape(2,2,1), 
                      Nop.reshape(1,2,2)]


        return qtn.MatrixProductOperator(arrays=oplist,
                                        shape='ludr',
                                        site_tag_id=site_tag)
    

    _three_site_nnint_mpo = functools.partialmethod(_two_site_nnint_mpo,
                                third_site_identity=True)


    

        

################################

class SpinhalfSimHam(SimulatorHam):
    '''Simulator Hamiltonian, acting on qubit space,
    that encodes the Fermi-Hubbard model Ham for 
    spin-1/2 fermions. Each local site is 4-dimensional.

    Gates act on 2 or 3 sites (2 vertices + 1 possible face)
    '''

    def __init__(self, qlattice, t, U):
        
        self._t = t
        self._U = U

        terms = self._make_ham_terms(qlattice)

        super().__init__(qlattice, terms)
    

    def gen_ham_terms(self):
        '''Generate (`where`, `gate`) pairs for every location
        (edge) to be acted on with a Ham term
        '''
        for (i,j,f), gate in self._ham_terms.items():
            where = (i,j) if f is None else (i,j,f)
            yield where, gate


    def gen_trotter_gates(self, tau):
        '''Generate (`where`, `exp(tau * gate)`) pairs, for each 
        location (edge) `where` and gate exponentiated by
        `tau`. 

        Generated in ordered groups of edges:
        1. Horizontal-even
        2. Horizontal-odd
        3. Vertical-even
        4. Vertical-odd
        '''
        for group in ['he', 'ho', 've', 'vo']:

            for edge, exp_gate in self._trotter_gate_group(group, tau).items():
                
                i,j,f = edge
                where = (i,j) if f is None else (i,j,f)
                yield where, exp_gate


    def _make_ham_terms(self, qlattice):
        '''Get all Hamiltonian terms, as two/three-site gates, 
        in a dict() mapping edges to arrays.
        
        `terms`:  dict[edge (i,j,f) : gate [qarray] ]

        If `f` is None, the corresponding gate will be two-site 
        (vertices only). Otherwise, gate acts on three sites.
        '''
        t, U = self.ham_params()

        terms = dict()

        #vertical edges
        for direction, sign in [('down', 1), ('up', -1)]:
            
            spin_up_hop = self._two_site_hop_gate(spin=0)
            spin_down_hop = self._two_site_hop_gate(spin=1)

            for (i,j,f) in qlattice.get_edges(direction):
                
                #two-site
                if f is None:
                    terms[(i,j,f)] =  sign * t * spin_up_hop
                    terms[(i,j,f)] += sign * t * spin_down_hop
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
                #three-site
                else:
                    terms[(i,j,f)] =  sign * t * self._three_site_hop_gate(spin=0, edge_dir='vertical')
                    terms[(i,j,f)] += sign * t * self._three_site_hop_gate(spin=1, edge_dir='vertical')
                    # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)


        #horizontal edges
        for (i,j,f) in qlattice.get_edges('right+left'):
            
            #two-site 
            if f is None:
                    terms[(i,j,f)] =  t * self._two_site_hop_gate(spin=0)
                    terms[(i,j,f)] += t * self._two_site_hop_gate(spin=1)
                    # terms[(i,j,f)] += U * self.onsite_gate()
                
            #three-site    
            else:
                terms[(i,j,f)] =  sign * t * self._three_site_hop_gate(spin=0, edge_dir='horizontal')
                terms[(i,j,f)] += sign * t * self._three_site_hop_gate(spin=1, edge_dir='horizontal')
                # terms[(i,j,f)] += U * self.onsite_gate() & qu.eye(4)
        
        
        if U == 0.0:
            return terms


        G_onsite = self._onsite_int_gate() #on-site spin-spin interaction

        #map each vertex to the list of edges where it appears
        self._vertices_to_covering_terms = defaultdict(list)
        for edge in terms:
            (i,j,f) = edge
            self._vertices_to_covering_terms[i].append(tuple([i,j,f]))
            self._vertices_to_covering_terms[j].append(tuple([i,j,f]))


        #for each vertex in lattice, absorb onsite repulsion term
        #uniformly into the edge terms that include it
        for vertex in qlattice.vertex_sites():
            
            #get edges that include this vertex
            edges = self._vertices_to_covering_terms[vertex]
            num_edges = len(edges)

            assert num_edges>1 or qlattice.num_faces()==0  #should appear in at least two edge terms!

            for (i,j,f) in edges:
                
                # ham_term = terms[(i,j,f)]

                v_place = (i,j,f).index(vertex) #vertex is either i or j

                if f is None: #term should act on two sites
                    terms[(i,j,f)] += U * (1/num_edges) * qu.ikron(G_onsite, dims=[4]*2, inds=v_place)

                else: #act on three sites
                    terms[(i,j,f)] += U * (1/num_edges) * qu.ikron(G_onsite, dims=[4]*3, inds=v_place)

        return terms


    def ham_params(self):
        '''t (hopping), U (repulsion)
        '''
        return (self._t, self._U)


    def _two_site_hop_gate(self, spin):
        '''(Encoded) hopping between two vertex sites, for 
        fermions in the `spin` sector.
        '''
        spin = {0: 'u',
                1: 'd',
                'up': 'u',
                'down': 'd',
                'u': 'u',
                'd': 'd'
                }[spin]
            
        X, Y, I = (qu.pauli(mu) for mu in ['x','y','i'])

        #`spin` says which spin sector to act on

        X_s = { 'u': X & I, 
                'd': I & X}[spin]

        Y_s = { 'u': Y & I,
                'd': I & Y}[spin]

        return 0.5* ((X_s & X_s) + (Y_s & Y_s))
    

    def _three_site_hop_gate(self, spin, edge_dir):
        '''Hop gate acting on two vertices and a face site,
        in `spin` sector. Action on the face site depends
        on `edge_dir`: {'vertical', 'horizontal'}
        '''
        spin = {0: 'up',
                1: 'down',
                'up': 'up',
                'down': 'down',
                'u': 'up',
                'd': 'down'
                }[spin]

        X, Y, I = (qu.pauli(mu) for mu in ['x','y','i'])
        
        #operators acting on the correct spin sector
        if spin == 'up':
            X_s = X & I
            Y_s = Y & I
        
        elif spin == 'down':
            X_s = I & X
            Y_s = I & Y

        #operator on face qubit (in `spin` sector)
        Of_s = {'vertical': X_s, 'horizontal':Y_s} [edge_dir]

        return 0.5 * ((X_s & X_s & Of_s) + (Y_s & Y_s & Of_s))


    def _onsite_int_gate(self):
        '''Spin-spin interaction at a single vertex.
        '''
        return number_op() & number_op()
    
    
