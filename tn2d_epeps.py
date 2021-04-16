import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import is_lone_coo
from quimb.tensor.tensor_core import tags_to_oset
from quimb.utils import check_opt

from autoray import do, dag
# import tqdm
import functools
# import opt_einsum as oe
# from operator import add

import numpy as np
import re
from random import randint
from collections import defaultdict
from numbers import Integral
from itertools import product, chain, starmap, cycle, combinations, permutations

import utils.dk_lattice_geometry as dk_lattice
import three_body_op

import qubit_networks
# from qubit_networks import insert_identity_between_tensors

class ePEPS(qtn.tensor_2d.TensorNetwork2DFlat,
            qtn.TensorNetwork2D,
            qtn.TensorNetwork):
    
    '''
    Notice that we are going from the convention 
    
       {site_tag_id='Q{}',
        grid_tag_id='S{},{}',
        aux_tag_id='IX{}Y{}',
        phys_ind_id='q{}'
        } 
        
        to 
        
       {site_tag_id='S{},{}',
        site_ind_id='k{},{}
        }

        --------------
        _SPECIAL_TAGS:
            -'aux_identity' is used for any dummy identity tensors
            -'adj_to_vertex_face'.format(V,F) is used if a dummy 
            identity `I` is 'paired with', or adjacent to, vertex and
            face sites with qubit numbers V, F, resp.
    '''

    _EXTRA_PROPS = (
        '_Lx',
        '_Ly',
        '_site_tag_id',
        '_site_ind_id',
        '_row_tag_id',
        '_col_tag_id',
        '_qubit_to_coo_map',
        # '_phys_dim',
        # '_grid_tag_id',
        # '_aux_tag_id', ... aux_tag_id='IX{}Y{}',
        # '_phys_ind_id',
    )
    
    _SPECIAL_TAGS = {
        'aux_identity': 'AUX', 
        'adj_to_vertex_face': 'ADJ{},{}' 
    }

    def __init__(self, tn, *, 
            Lx=None, Ly=None, 
            qubit_to_coo_map=None,
            site_tag_id='S{},{}', 
            site_ind_id='k{},{}',
            row_tag_id='ROW{}',
            col_tag_id='COL{}',
            **tn_opts):
        '''
        -`Lx` and `Ly` are different from the "true"
        spin lattice, as they count both face and 
        vertex qubits.
                    
                     original            ePEPS
                     lattice             lattice
                    :       :           :   :   :          
              x+1  ─●───────●─         ─●───●───●─  x+2
                    │ \   / │           │   │   │ 
                    │   ●   │    ==>   ─●───●───●─  x+1
                    │ /   \ │           │   │   │ 
               x   ─●───────●─         ─●───●───●─   x
                    :       :           :   :   : 

                     Lx * Ly        (2Lx-1) * (2Ly-1)

        -`site_tag_id` and `site_ind_id` should be 
        of the form "...x,y".
        '''
        
        if isinstance(tn, ePEPS):
            super().__init__(tn)
            return

        self._Lx = Lx
        self._Ly = Ly

        self._site_tag_id = site_tag_id # "S{x},{y}"
        self._site_ind_id = site_ind_id # "k{x},{y}"
        self._row_tag_id = row_tag_id # "ROW{x}"
        self._col_tag_id = col_tag_id # "COL{y}"

        # map {q: (x,y)}
        self._qubit_to_coo_map = qubit_to_coo_map
        
        # inverse map {(x,y): q}
        self._coo_to_qubit_map = {coo: q for q, coo in 
                            self._qubit_to_coo_map.items()}


        # Pass tn.tensors, i.e. a raw sequence of tensors, to avoid
        # inheriting any of the Lx, Ly, tag attributes, etc, from `tn`
        super().__init__(tn.tensors, **tn_opts)


    @property
    def aux_identity_tag(self):
        # Tags on dummy identity tensors, e.g. "AUX"
        return self.__class__._SPECIAL_TAGS['aux_identity']

    
    @property
    def adjacent_aux_tag(self):
        # Tag like "ADJ{V},{F}", used for a dummy identity that may
        # be 'reabsorbed' into qubit site "Q{V}"
        return self.__class__._SPECIAL_TAGS['adj_to_vertex_face']

    @property
    def site_ind_id(self):
        return self._site_ind_id

    @property
    def qubit_to_coo_map(self):
        '''Dict mapping coordinates to qubit 
        integer labels {(x,y): q}
        '''                            
        return self._qubit_to_coo_map


    def coo_to_qubit_map(self, coo=None):
        '''The integer label ``q`` of the qubit at
        (x,y)-coordinate ``coo``.
        '''       
        if not hasattr(self, '_coo_to_qubit_map'):
            self._coo_to_qubit_map = {coo: q 
                for q, coo in self.qubit_to_coo_map.items()}

        if coo is None:
            return self._coo_to_qubit_map
        
        return self._coo_to_qubit_map[coo]


    def find_tensor_coo(self, t, get='coo'):
        '''Get the coo of the given tensor `t`.
        Important: assumes coo tag is of the form "...x,y",
        e.g. "Sx,y". 
        
        get: {'coo', 'tag', 'ind'}
            -'coo': returns tuple[int]
            -'tag': returns tag string e.g. "Sx,y"
            -'ind': returns index string e.g. "kx,y"
        '''
        check_opt("get", get, ('coo', 'tag', 'ind'))

        p = re.compile(self.site_tag_id.format('\d','\d'))
        coo_tag, = (s for s in t.tags if p.match(s))
        coo = int(coo_tag[-3]), int(coo_tag[-1]) # This assumes tag is ~ "...x,y"

        return {'coo': coo,
            'tag': coo_tag,
            'ind': self.site_ind_id.format(*coo)
            }[get]
         

    def draw(
        self, 
        fix_lattice=True, 
        layer_tags=None, 
        with_gate=False, 
        auto_detect_layers=True,
        **graph_opts):
        '''
        TODO: DEBUG ``fix_tags`` PARAM
              GRAPH GATE LAYERS

        Overload ``TensorNetwork.draw`` for aesthetic lattice-fixing,
        unless `fix_lattice` is set False. 

        Geometry: coordinate origin at bottom left.

       Lx,0─── ...          ───Lx,Ly                               
         │                       │  
         :       :               :   
         │       │               │  
       (1,0)───(1,1)─── ... ───(1,Ly)
         │       │       │       │
       (0,0)───(0,1)─── ... ───(0,Ly)


        `auto_detect_layers` will check if 'BRA', 'KET' are in `self.tags`,
        and if so will attempt to graph as a two-layer sandwich.
        Can also specify `layer_tags` specifically for TNs with other layers.
        '''
        
        graph_opts.setdefault('color', ['QUBIT', 'AUX'])
        # graph_opts.setdefault('custom_colors', ((0.8, 0.22, 0.78, 0.9),(0.65, 0.94, 0.19, 0.9)))
        graph_opts.setdefault('show_tags', False)
        graph_opts.setdefault('show_inds', True)

        if all((auto_detect_layers == True,
                'BRA' in self.tags,
                'KET' in self.tags)):
            layer_tags=('BRA', 'KET')

        scale_X, scale_Y = 1, 1.5
        fix_tags = {self.site_tag_id.format(x,y): (y * scale_Y, x * scale_X) 
            for (x,y) in product(range(self.Lx), range(self.Ly))}
        
        super().draw(fix=fix_tags, **graph_opts)
        return
    

    @property
    def site_tag_id(self):
        '''Format string for the tag identifiers of local sites,
        i.e. the coordinates 'S{x},{y}'
        '''
        return self._site_tag_id
    

    def qubit_site_tags(self):
        '''
        The site 'Sx,y' tags, only for qubit
        sites (i.e. non-identity tensors)
        '''
        return (t for t in self.site_tags if ('QUBIT' in self[t].tags))


    def gate(
        self, 
        G,
        coos,
        contract='auto_split',
        tags=('GATE',),
        inplace=False,
        info=None,
        **compress_opts,
    ):
        '''
        contract: {False, 'reduce_split', 'triangle_absorb', 'reduce_split_lr'}
                -False: leave gate uncontracted at sites
            [For 2-body ops:]
                -reduce_split: Absorb dummy, apply gate with `qtn.tensor_2d.reduce_split`, 
                    then reinsert dummy. (NOTE: this one seems very slow)
                -reduce_split_lr: leave dummy as-is, treat gate as a LR interaction.
                    The final bonds are much smaller this way!
            [For 3-body ops:]
                -triangle_absorb: use `three_body_op.triangle_gate_absorb` to 
                    apply the 3-body gate. Assumes `coos` is ordered like 
                    ~ (vertex, vertex, face)!
            [For any n-body:]
                -auto_split: will automatically choose depending on n.
                    n=1 -> contract = True
                    n=2 -> contract = 'reduce_split_lr'
                    n=3 -> contract = 'triangle_absorb'
        '''

        check_opt("contract", contract, (False, True, 'reduce_split', 
                            'triangle_absorb', 'reduce_split_lr', 
                            'auto_split'))

        psi = self if inplace else self.copy()

        if is_lone_coo(coos):
            coos = (coos,)
        else:
            coos = tuple(coos)

        numsites = len(coos) #num qubits acted on

        if contract == 'auto_split': 
            contract = {1: True, 
                        2: 'reduce_split_lr', 
                        3: 'triangle_absorb'}[numsites]

        #inds like 'k{x},{y}'
        site_inds = [self._site_ind_id.format(*c) for c in coos] 
        # physical dimension, d=2 for qubits
        dp = self.ind_size(site_inds[0]) 
        gate_tags = tags_to_oset(tags)

        G = qtn.tensor_1d.maybe_factor_gate_into_tensor(G, dp, numsites, coos)

        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        reindex_map = dict(zip(site_inds, bond_inds))
        TG = qtn.Tensor(G, inds=site_inds + bond_inds, left_inds=bond_inds, tags=gate_tags)

        if contract is False:
            #attach gates without contracting any bonds
            #
            #     'qA' 'qB' 
            #       │   │      <- site_inds
            #       GGGGG
            #       │╱  │╱     <- bond_inds
            #     ──●───●──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)
            psi |= TG
            return psi
        

        elif (contract is True) or (numsites == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bond_inds, which='any')
            # pop the sites, contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
            psi |= qtn.tensor_contract(*pts, TG)
            return psi


        elif contract == 'triangle_absorb' and numsites == 3:
            # absorbs 3-body gate while preserving lattice structure.
            psi.absorb_three_body_tensor_(TG=TG, coos=coos, 
            reindex_map=reindex_map, phys_inds=site_inds,
            gate_tags=gate_tags, **compress_opts)
            return psi
        
        # NOTE: this one seems very inefficient for 
        # "next-nearest" neighbor interactions.
        elif contract == 'reduce_split' and numsites == 2:
            # First absorb identity into a site, then 
            # restore after gate has been applied.
            # 
            # 1. Absorb identity step:
            # 
            #       │     │    Absorb     │     │
            #       GGGGGGG     ident.    GGGGGGG    
            #       │╱  ╱ │╱     ==>      │╱    │╱╱   
            #     ──●──I──●──           ──●─────●─  
            #    a ╱  ╱  ╱ b             ╱     ╱╱    
            #
            # 2. Gate 'reduce_split' step: 
            # 
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            # 3. Reinsert identity:
            # 
            #       │╱    │╱╱           │╱  ╱ │╱
            #     ──G┄┄┄┄┄G──   ==>   ──G┄┄I┄┄G──      
            #      ╱     ╱╱            ╱  ╱  ╱         
            # 

            (x1,y1), (x2,y2) = coos
            mid_coo = (int((x1+x2)/2), int((y1+y2)/2))
            dummy_coo_tag = psi.site_tag_id.format(*mid_coo)

            # keep track of dummy identity's tags and neighbors
            prev_dummy_info = {'tags': psi[dummy_coo_tag].tags, 
                      'neighbor_tags': tuple(t.tags for t in 
                        psi.select_neighbors(dummy_coo_tag))} 
            

            which_bond = int(psi.bond_size(coos[0], mid_coo) >= 
                             psi.bond_size(coos[1], mid_coo))


            if which_bond == 0: # (vertex_0 ── identity) bond is larger
                vertex_tag = psi.site_tag_id.format(*coos[0])

            else:  # (vertex_1 -- identity) bond larger
                vertex_tag = psi.site_tag_id.format(*coos[1])

            tids = psi._get_tids_from_tags(
                        tags=(vertex_tag, dummy_coo_tag), which='any')
        
            # pop and reattach the (vertex & identity) tensor
            pts = [psi._pop_tensor(tid) for tid in tids]
            new_vertex = qtn.tensor_contract(*pts)
            
            # new_vertex.drop_tags(prev_dummy_info['tags'] - )
            new_vertex.drop_tags(pts[1].tags - pts[0].tags)
            
            psi |= new_vertex # reattach [vertex & identity] 

            # insert 2-body gate!
            qtn.tensor_2d.gate_string_reduce_split_(TG=TG, where=coos,
                string=coos, original_ts=[psi[c] for c in coos], 
                bonds_along=(psi.bond(*coos),), reindex_map=reindex_map,
                site_ix=site_inds, info=info, **compress_opts)

            # now restore the dummy identity between vertices
            vtensor = psi[coos[which_bond]] # the vertex we absorbed dummy into
            
            ts_to_connect = set(psi[tags] for tags in 
                prev_dummy_info['neighbor_tags']) - set([vtensor])

            for T2 in ts_to_connect: # restore previous dummy bonds
                psi |= qubit_networks.insert_identity_between_tensors(
                    T1=vtensor, T2=T2, add_tags='TEMP')

            # contract new dummies into a single identity
            psi ^= 'TEMP'
            for t in prev_dummy_info['tags']:
                psi['TEMP'].add_tag(t) # restore previous dummy tags
            psi.drop_tags('TEMP')

            return psi.fuse_multibonds_()
        

        elif contract == 'reduce_split_lr' and numsites == 2:
            # There will be a 'dummy' identity tensor between the
            # sites, so the 2-body operator will look "long-range"
            # 
            #       │     │              
            #       GGGGGGG                 
            #       │╱  ╱ │╱              │╱  ╱ │╱
            #     ──●──I──●──    ==>    ──G┄┄I┄┄G── 
            #      ╱  ╱  ╱               ╱  ╱  ╱
            #
            (x1,y1), (x2,y2) = coos
            mid_coo = (int((x1 + x2)/2), int((y1 + y2)/2))
            
            dummy_coo_tag = psi.site_tag_id.format(*mid_coo)
            string = (coos[0], mid_coo, coos[1])
            
            original_ts = [psi[coo] for coo in string]
            bonds_along = [next(iter(qtn.bonds(t1, t2)))
                    for t1, t2 in qu.utils.pairwise(original_ts)]

            qtn.tensor_2d.gate_string_reduce_split_(TG=TG, 
                where=coos, string=string, original_ts=original_ts, 
                bonds_along=bonds_along, reindex_map=reindex_map,
                site_ix=site_inds, info=info, **compress_opts)
            
            return psi


    gate_ = functools.partialmethod(gate, inplace=True)


    # def absorb_three_body_gate(
    #     self, 
    #     G, 
    #     coos, 
    #     DEBUG=False,
    #     gate_tags=('GATE',),
    #     inplace=False,
    #     **compress_opts
    #     ):
    #     '''
    #     G: dense array
    #         3-body operator to apply.

    #     coos: sequence of tuples (x,y)
    #         The three coordinate pairs of qubits to
    #         act on. The 'face' qubit should be last!
        
    #     gate_tags: None or sequence of str, optional
    #         Sites acted on with ``G`` will be tagged 
    #         with these.

    #     **compress_opts:            
    #         Passed to `triangle_gate_absorb`, which 
    #         passes it to `qtn.tensor.split`.

    #     '''
    #     psi = self if inplace else self.copy()

    #     vertex_a, vertex_b, face_coo = coos
        
    #     face_qnum = psi.coo_to_qubit_map(face_coo)
        
    #     ## keep dummies' tags, inds, etc to restore them later
    #     dummy_identities_info = {}

    #     # absorb appropriate identity tensors into vertex sites
    #     for k, vcoo in enumerate((vertex_a, vertex_b)):

    #         vertex_qnum = psi.coo_to_qubit_map(vcoo)            
            
    #         # tag of identity to be absorbed
    #         adjacent_tag = psi.adjacent_aux_tag.format(
    #                                 vertex_qnum, 
    #                                 face_qnum) # "AUX{V},{F}"
            
    #         vertex_tag = psi._site_tag_id.format(*vcoo)
            
    #         dummy_identities_info.update({
    #             (k, 'neighbor_tags'): tuple(t.tags for t in 
    #                 psi.select_neighbors(adjacent_tag))
    #         })

    #         tids = psi._get_tids_from_tags(
    #             tags=(vertex_tag, adjacent_tag), which='any')
            
    #         # pop and reattach the contracted tensors
    #         pts = [psi._pop_tensor(tid) for tid in tids]
    #         new_vertex = qtn.tensor_contract(*pts)
            
    #         dummy_identities_info.update({
    #             (k, 'tags'): pts[1].tags, #dummy tags 
    #             (k, 'inds'): pts[1].inds, #dummy indices
    #         })
            
    #         new_vertex.drop_tags(pts[1].tags - pts[0].tags) # drop dummy tags from vertex site
    #         psi |= new_vertex


    #     vertex_tensors = [psi[coo] for coo in (vertex_a, vertex_b)]
    #     face_tensor = psi[face_coo] 
        
    #     ###vv Here differs from `absorb_three_body_tensor` vv###

    #     gate_tags = qtn.tensor_2d.tags_to_oset(gate_tags)

    #     # assuming physical dimension = 2
    #     G = qtn.tensor_1d.maybe_factor_gate_into_tensor(G, dp=2, ng=3, where=coos)

    #     #new physical indices "k{x},{y}"
    #     phys_inds = [psi._site_ind_id.format(*c) for c in coos] 
    #     # old physical indices joined to new gate
    #     bond_inds = [qtn.rand_uuid() for _ in range(3)] 
    #     # replace physical inds with gate bonds
    #     reindex_map = dict(zip(phys_inds, bond_inds)) 

    #     TG = qtn.Tensor(G, inds=phys_inds + bond_inds, left_inds=bond_inds, tags=gate_tags)
        
    #     three_body_op.triangle_gate_absorb(TG=TG, reindex_map=reindex_map, 
    #                 vertex_tensors=vertex_tensors, 
    #                 face_tensor=face_tensor, phys_inds=phys_inds,
    #                 gate_tags=gate_tags, **compress_opts) # apply gate!


    #     if DEBUG:
    #         return psi

    #     # now insert new dummy identities where they used to be
    #     for k in range(2):
    #         vt = vertex_tensors[k]
            
    #         ts_to_connect = set(
    #             psi[tags] for tags in dummy_identities_info[(k, 'neighbor_tags')]
    #             ) - set([vt])

    #         for T2 in ts_to_connect:
    #             psi |= insert_identity_between_tensors(T1=vt, T2=T2, add_tags='TEMP')

    #         # contract new dummies into a single identity
    #         psi ^= 'TEMP'
    #         # restore previous tags, and drop temporary tag
    #         for tag in dummy_identities_info[(k, 'tags')]:
    #             psi['TEMP'].add_tag(tag)
    #         psi.drop_tags('TEMP')

    #     return psi

    def absorb_three_body_gate(
        self, 
        G, 
        coos,
        gate_tags=('GATE',),
        restore_dummies=True,
        inplace=False,
        **compress_opts
    ):
        '''Converts the raw gate ``G`` into a tensor and passes it
        to ``self.absorb_three_body_tensor``.

        G: raw qarray
            The gate to apply
        coos: sequence of tuple[int]
            The 3 coos to act on, e.g. ((0,0),(0,2),(1,1))
        restore_dummies: bool, optional
            Whether to "restore" dummy identities
            and keep square lattice structure or 
            "triangles" in the lattice.
        '''
        gate_tags = qtn.tensor_2d.tags_to_oset(gate_tags)
        # assuming physical dimension = 2
        G = qtn.tensor_1d.maybe_factor_gate_into_tensor(G, dp=2, ng=3, where=coos)
        # new physical indices "k{x},{y}"
        phys_inds = [self._site_ind_id.format(*c) for c in coos] 
        # old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(3)] 
        # replace physical inds with gate bonds
        reindex_map = dict(zip(phys_inds, bond_inds)) 

        TG = qtn.Tensor(G, inds=phys_inds+bond_inds, left_inds=bond_inds, tags=gate_tags)
        
        return self.absorb_three_body_tensor(TG, coos, reindex_map, phys_inds, 
            gate_tags, restore_dummies=restore_dummies, inplace=inplace,
            **compress_opts)
        

    absorb_three_body_gate_ = functools.partialmethod(absorb_three_body_gate, 
                                inplace=True)

    def absorb_three_body_tensor(
        self,
        TG,
        coos,
        reindex_map,
        phys_inds,
        gate_tags,
        restore_dummies=True,
        inplace=False,
        **compress_opts
    ):
        '''Serves the same purpose as ``self.absorb_three_body_gate``, 
        but assumes gate has already been shaped into a tensor and
        appropriate indices have been gathered.

        TG: qtn.Tensor
            The 3-body gate (shape [2]*8) as a tensor.
        
        coos: sequence of tuple[int, int]
            The (x,y)-coordinates for 3 qubit sites to
            hit with the gate.
            
        phys_inds: sequence of str
            The target qubits' physical indices "k{x},{y}"
        
        reindex_map: dict[str: str]
            Map `phys_inds` to the bonds between sites and
            gate acting on those sites.
        
        gate_tags: None or sequence of str, optional
            Sites acted on with ``TG`` will have these
            tags added to them.
        
        inplace: bool
            If False, will make a copy of ``self`` and
            act on that instead.
        '''
        psi = self if inplace else self.copy()

        vertex_a, vertex_b, face_coo = coos
        
        face_qnum = psi.coo_to_qubit_map(face_coo)
        
        ## keep track of dummies' tags & neighbor tensors
        dummy_identities_info = {}

        # absorb appropriate identity tensors into vertex sites
        for k, vcoo in enumerate((vertex_a, vertex_b)):

            vertex_qnum = psi.coo_to_qubit_map(vcoo)            
            
            # tag of identity to be absorbed
            adjacent_tag = psi.adjacent_aux_tag.format(
                            vertex_qnum, face_qnum) # "AUX{V},{F}"
            
            # tag of vertex to absorb identity _into_
            vertex_tag = psi._site_tag_id.format(*vcoo)
            
            dummy_identities_info.update({
                (k, 'neighbor_tags'): tuple(t.tags for t in 
                    psi.select_neighbors(adjacent_tag))
            })

            tids = psi._get_tids_from_tags(
                tags=(vertex_tag, adjacent_tag), which='any')
            
            # pop and reattach the contracted tensors
            pts = [psi._pop_tensor(tid) for tid in tids]
            new_vertex = qtn.tensor_contract(*pts)
            
            dummy_identities_info.update({
                (k, 'tags'): pts[1].tags, #dummy tags 
                (k, 'coo'): psi.find_tensor_coo(pts[1]), #coo (x,y)
                # (k, 'inds'): pts[1].inds, #dummy indices
            })
            
            # drop dummy's tags from vertex site
            new_vertex.drop_tags(pts[1].tags - pts[0].tags) 
            psi |= new_vertex

        vertex_tensors = [psi[coo] for coo in (vertex_a, vertex_b)]
        face_tensor = psi[face_coo] 
        
        # apply gate!
        three_body_op.triangle_gate_absorb(TG=TG, reindex_map=reindex_map, 
                    vertex_tensors=vertex_tensors, 
                    face_tensor=face_tensor, phys_inds=phys_inds,
                    gate_tags=gate_tags, **compress_opts)


        if not restore_dummies:
            return psi

        # now insert new dummy identities where they used to be

        # for k in range(2):
        #     vt = vertex_tensors[k]
            
        #     ts_to_connect = set(
        #         psi[tags] for tags in dummy_identities_info[(k, 'neighbor_tags')]
        #         ) - set([vt])

        #     for T2 in ts_to_connect:
        #         psi |= insert_identity_between_tensors(T1=vt, T2=T2, add_tags='TEMP')

        #     # contract new dummies into a single identity
        #     psi ^= 'TEMP'
        #     # restore previous tags, and drop temporary tag
        #     for tag in dummy_identities_info[(k, 'tags')]:
        #         psi['TEMP'].add_tag(tag)
        #     psi.drop_tags('TEMP')


        for k, vcoo in enumerate((vertex_a, vertex_b)):
            
            vtensor = psi[vcoo]
            vertex_ind = psi.site_ind_id.format(*vcoo) # kx,y
            
            ts_to_connect = set(psi[tags] for tags in 
                dummy_identities_info[(k, 'neighbor_tags')]) - set([vtensor])
            
            dummy_coo = dummy_identities_info[(k,'coo')] # (x,y)

            # free_inds = [ix for ix in vtensor.inds if 
            #     len(psi.ind_map[ix]) == 1]
            
            # bonds connecting to dummy's neighbors
            dummy_bonds = qu.oset.union(
                *(qtn.bonds(vtensor, t) for t in ts_to_connect))

            # pop the vertex tensor, to split and reattach soon
            vtensor, = (psi._pop_tensor(x) for x in 
                psi._get_tids_from_inds(vertex_ind))
            
            # the dummy may not have a physical "kx,y" index
            dummy_phys_ix = psi.site_ind_id.format(*dummy_coo) 
            if dummy_phys_ix in vtensor.inds:
                dummy_bonds |= qu.oset([dummy_phys_ix])

            # split into vertex & dummy
            new_vertex, new_dummy = vtensor.split(
                left_inds=None, method='qr', get='tensors',
                right_inds=dummy_bonds)


            new_dummy.drop_tags()
            for t in dummy_identities_info[(k, 'tags')]:
                new_dummy.add_tag(t)

            psi |= new_vertex
            psi |= new_dummy

        return psi
    

    absorb_three_body_tensor_ = functools.partialmethod(absorb_three_body_tensor,
                                    inplace=True)
    

    def add_fake_phys_inds(self, dp):
        '''
        Add "k{x},{y}" indices to all the tensor sites,
        including auxiliary sites, so that every site 
        has a 'physical' index. Inplace operation.

        dp: int, the size of the fake physical indices
        '''

        for x, y in product(range(self.Lx), range(self.Ly)):
            ind_xy = self._site_ind_id.format(x,y)
            
            # skip the 'kx,y' indices that already exist
            if ind_xy in self.ind_map.keys(): 
                continue

            # add the new index
            self[x,y].new_ind(name=ind_xy, size=dp)
                
            

class ePEPSvector(ePEPS, 
            qtn.tensor_2d.TensorNetwork2DVector,
            qtn.tensor_2d.TensorNetwork2DFlat,
            qtn.TensorNetwork2D, 
            qtn.TensorNetwork):

    
    _EXTRA_PROPS = (
        '_Lx',
        '_Ly',
        '_site_tag_id',
        '_site_ind_id',
        '_row_tag_id',
        '_col_tag_id',
        '_qubit_to_coo_map',
        # '_phys_dim',
        # '_grid_tag_id',
        # '_aux_tag_id', ... aux_tag_id='IX{}Y{}',
        # '_phys_ind_id',
    )
    

    _SPECIAL_TAGS = {
        'aux_identity': 'AUX', 
        'adj_to_vertex_face': 'ADJ{},{}' 
    }


    def is_qubit_coo(self, x, y):
        '''Whether (x,y) lattice site is a genuine 
        qubit site (rather than dummy site).

        Note that 'empty' face sites yield True! i.e.
        >>> psi.is_qubit_coo(x,y) == ('QUBIT' in psi[x,y].tags)

        will be True except at empty face sites, which are
        'genuine' but are empty due to DK encoding.
        '''
        return all((x % 2 == y % 2, 
                0 <= x < self.Lx,
                0 <= y < self.Ly))


    def is_vertex_coo(self, x, y):
        return all((x % 2 == 0, 
                    y % 2 == 0,
                    0 <= x < self.Lx,
                    0 <= y < self.Ly))


    def is_face_coo(self, x, y):
        return all((x % 2 == 1, 
                    y % 2 == 1,
                    0 <= x < self.Lx,
                    0 <= y < self.Ly))


    def calc_plaquette_map(self, plaquettes, include_3_body=True):
        """Generate a dictionary of all the coordinate pairs in ``plaquettes``
        mapped to the 'best' (smallest) rectangular plaquette that contains them.
        
        Will optionally compute for 3-length combinations as well, to
        capture 3-local qubit interactions like (vertex, vertex, face)
        interactions.

        Args:
        -----
            plaquettes: sequence of tuple[tuple[int]]
                Sequence of plaquettes like ((x0, y0), (dx, dy))

            include_3_body: bool, optional
                Whether to include 3-local interactions as well 
                as 2-local (pairwise).


        TODO: if not include_3_body we can just use the super class method. 
        """
        if not include_3_body:
            return super().calc_plaquette_map(plaquettes)
        
        # sort in descending total plaquette size
        plqs = sorted(plaquettes, key=lambda p: (-p[1][0] * p[1][1], p))
        
        mapping = dict()
        for p in plqs:
            sites = qtn.tensor_2d.plaquette_to_sites(p)

            # pairwise (2-local) interactions
            for coo_pair in combinations(sites, 2):
                if all(tuple(starmap(self.is_qubit_coo, coo_pair))):
                    mapping[coo_pair] = p

            # 3-local interactions
            for coo_triple in combinations(sites, 3):
                if all(tuple(starmap(self.is_qubit_coo, coo_triple))):
                    # make sure face qubit is the third entry
                    if self.is_face_coo(*coo_triple[0]):
                        coo_triple = (coo_triple[2], coo_triple[1], coo_triple[0])

                    elif self.is_face_coo(*coo_triple[1]):
                        coo_triple = (coo_triple[0], coo_triple[2], coo_triple[1])
                    
                    mapping[coo_triple] = p
        
        return mapping


    def calc_plaquette_envs_and_map(self, terms, autogroup=True, **plaquette_env_options):
        '''Returns the plaquette_envs and plaquette_map needed to
        compute local expectations, overriding `calc_plaquette_map`
        to include 3-body interactions.
        '''
        norm = self.make_norm(return_all=False)

        # set some sensible defaults
        plaquette_env_options.setdefault('layer_tags', ('KET', 'BRA'))

        plaquette_envs = dict()
        for x_bsz, y_bsz in qtn.tensor_2d.calc_plaquette_sizes(terms.keys(), autogroup):
            plaquette_envs.update(norm.compute_plaquette_environments(
                x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options))

        # work out which plaquettes to use for which terms
        plaquette_map = self.calc_plaquette_map(plaquette_envs)
        
        # adjust plaqmap to the Hamiltonian term ordering
        for coos in terms.keys():
            
            if coos in plaquette_map: 
                continue    # good

            for perm in permutations(coos):
                if perm in plaquette_map: 
                    plaquette_map.update({coos: plaquette_map[perm]})
            
        return plaquette_envs, plaquette_map
