import numpy as np
from itertools import product
import quimb as qu
import quimb.tensor as qtn
import qubit_networks
from numbers import Integral
from quimb.utils import check_opt
import functools

class PEPS(qtn.TensorNetwork):

    _EXTRA_PROPS = (
        '_Lx',
        '_Ly',
        '_chi',
    )
    
    _site_tag_id = 'Q{}'
    _phys_ind_id = 'q{}'
    
    def __init__(self, tn, Lx, Ly, chi):

        self._Lx = Lx
        self._Ly = Ly
        self._chi = chi

        super().__init__(tn, structure=PEPS._site_tag_id)
    

    def _compatible_2d(self, other):
        return (
            isinstance(other, PEPS) and 
            all(getattr(self, e)==getattr(other,e)
                for e in PEPS._EXTRA_PROPS)
            )


    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_2d(other):
            new.view_as_(PEPS, 
                        like=self, 
                        Lx=self._Lx,
                        Ly=self._Ly,
                        chi=self._chi)
        return new


    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_2d(other):
            new.view_as_(PEPS, 
                        like=self, 
                        Lx=self._Lx,
                        Ly=self._Ly,
                        chi=self._chi)
        return new


    @classmethod
    def random_peps(cls, Lx, Ly, chi=5):
        '''Make a vector (PEPS shape) w/ random tensors
        '''
        vtn = qubitNetworks.make_vertex_net(Lx, Ly, chi, site_tag_id=PEPS._site_tag_id)
        vtn.retag({'VERT':'KET'}, inplace=True)
        vtn.randomize(inplace=True)
        return cls(vtn, Lx, Ly, chi)


    def __getitem__(self, site):
        '''Can fetch PEPS tensors via the one-number labeling scheme
        (`site` is an Integer) or the two-number coo scheme (`site` is a 
        tuple like (x,y)).
        '''
        return super().__getitem__(self.maybe_convert_coo(site))


    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``(i, j)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)


    def coo_to_tag(self, i, j):
        return PEPS._site_tag_id.format(i*self._Ly + j)
    
    def site_to_tag(self, k):
        return PEPS._site_tag_id.format(k)


    def gen_vert_coos(self):
        '''Generate *vertex* coordinates (i,j)
        '''
        return product( range(self._Lx),
                        range(self._Ly))


    def maybe_convert_coo(self, site):
        '''If site is a tuple coo (i,j) *or* an integer k
        return the `tag` label of the site e.g. `Q{k}`

        site: int or tuple(int, int)
            If site is neither, return it unchanged.
        '''
        if isinstance(site, Integral):
                return PEPS._site_tag_id.format(site)

        if not isinstance(site, str):
            try:
                i,j = map(int, site)
                return PEPS._site_tag_id.format((i*self._Ly) + j)

            except:
                pass          
        
        return site



    # def copy(self):
    #     return self.__class__(self, self._Lx, self._Ly, self._chi)
    

    # __copy__ = copy


    def graph(self, fix_lattice=True, color=['KET','BRA','GATE'],**graph_opts):
        
        if not fix_lattice: 
            super().graph(color, **graph_opts)
        
        
        else:
            try:
                Lx,Ly = self._Lx, self._Ly
                fix = {
                    **{(f'Q{i*Ly+j}'): (j, -i) for i,j in product(range(Lx),range(Ly))}
                    }
                super().graph(color=color, fix=fix, show_inds=True, **graph_opts)
            
            except:
                super().graph(color, **graph_opts)


    def norm_tensor(self):
        bra = self.retag({'KET':'BRA'}).H
        return bra|self

    
    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one.
        """
        tn = self if inplace else self.copy()

        # for k in range(self.nsites):
        #     tn = tn ^ k
        for i,j in self.gen_vert_coos():
            tn ^= (i,j)

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn
        # return tn.view_as_(TensorNetwork2DFlat, like=self)


    def canonize_row(self, i, sweep, yrange=None, **canonize_opts):
        check_opt('sweep',sweep,('right','left'))

        if yrange is None:
            yrange = (0, self._Ly-1)
        
        if sweep == 'right':
            for j in range(min(yrange), max(yrange),+1):
                self.canonize_between((i, j), (i, j+1), **canonize_opts)
        
        else:
            for j in range(max(yrange), min(yrange),-1):
                self.canonize_between((i, j), (i, j-1), **canonize_opts)
    

    def canonize_column(self, j, sweep, xrange=None, **canonize_opts):
        check_opt('sweep', sweep, ('up','down'))

        if xrange is None:
            xrange = (0, self._Lx-1)
        
        if sweep == 'down':
            for i in range(min(xrange), max(xrange), +1):
                self.canonize_between((i, j), (i+1, j), **canonize_opts)
        
        else:
            for i in range(max(xrange), min(xrange), -1):
                self.canonize_between((i, j), (i-1, j), **canonize_opts)

            
    def compress_row(self, i, sweep, yrange=None, **compress_opts):
        check_opt('sweep', sweep, ('right', 'left'))
        compress_opts.setdefault('absorb', 'right')

        if yrange is None:
            yrange = (0, self._Ly-1)
        
        if sweep == 'right':
            for j in range(min(yrange), max(yrange), +1):
                self.compress_between((i, j), (i, j+1), **compress_opts)
        
        else:
            for j in range(max(yrange), min(yrange), -1):
                self.compress_between((i, j), (i, j-1), **compress_opts)
        

    def compress_column(self, j, sweep, xrange=None, **compress_opts):
        check_opt('sweep', sweep, ('up', 'down'))

        if xrange is None:
            xrange = (0, self._Lx - 1)

        if sweep == 'down':
            compress_opts.setdefault('absorb', 'right')
            for i in range(min(xrange), max(xrange), +1):
                self.compress_between((i, j), (i + 1, j), **compress_opts)
        
        else:
            compress_opts.setdefault('absorb', 'left')
            for i in range(max(xrange), min(xrange), -1):
                self.compress_between((i - 1, j), (i, j), **compress_opts)
        


    def _contract_boundary_from_top_single(
        self,
        xrange,
        yrange,
        canonize=True,
        compress_sweep='left',
        layer_tag=None,
        **compress_opts
        ):
        '''For a single 2D layer of tensors
        '''

        canonize_sweep = {
            'left':'right',
            'right':'left'
        }[compress_sweep]
        
        print(xrange)


        for i in range(min(xrange), max(xrange)):

            for j in range(min(yrange), max(yrange)+1):

                tag1 = self.coo_to_tag(i, j)
                tag2 = self.coo_to_tag(i + 1, j)
                
                print(f'Tag 1 {tag1}')
                print(f'Tag 2 {tag2}')

                if layer_tag is None:
                    #contract any tensors with these coordinates
                    self.contract_((tag1,tag2), which='any')
                
                else:
                    #method only exists in `tensor_2d` branch!
                    self.contract_between(tag1, (tag2, layer_tag))

            if canonize:
                self.canonize_row(i, sweep=canonize_sweep, yrange=yrange)
        

            self.compress_row(i, sweep=compress_sweep, 
                            yrange=yrange, **compress_opts)
        

    def _contract_boundary_from_top_multilayer(
        self,
        xrange,
        yrange,
        layer_tags,
        canonize=True,
        compress_sweep='left',
        **compress_opts        
    ):
        for i in range(min(xrange), max(xrange)):
            #make sure exterior sites are a single tensor
            for j in range(min(yrange), max(yrange)+1):
                self ^= (i,j)
            
            for tag in layer_tags:
                #contract interior sites from layer `tag`
                self._contract_boundary_from_top_single(
                    xrange=(i, i+1), yrange=yrange,
                    canonize=canonize, layer_tag=tag,
                    **compress_opts)
            
                #so we can still uniqely identify 'inner' tensors, drop inner
                #site tag merged into outer tensor for all but last tensor
                for j in range(min(yrange), max(yrange) + 1):
                    inner_tag = self.coo_to_tag(i + 1, j)
                    if len(self.tag_map[inner_tag]) > 1:
                        self[i, j].drop_tags(inner_tag)


    def contract_boundary_from_top(
        self,
        xrange, 
        yrange=None,
        canonize=True,
        compress_sweep='left',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()

        if yrange is None:
            yrange = (0, self._Ly - 1)
        


        if layer_tags is None:
            tn._contract_boundary_from_top_single(
                xrange, yrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        
        else:
            tn._contract_boundary_from_top_multilayer(
                xrange, yrange, layer_tags, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        
        return tn
    
    contract_boundary_from_top_ = functools.partialmethod(
        contract_boundary_from_top, inplace=True)


    def _contract_boundary_from_bottom_single(
        self,
        xrange,
        yrange,
        canonize=True,
        compress_sweep='left',
        layer_tag=None,
        **compress_opts
        ):
            canonize_sweep = {
                'left': 'right',
                'right': 'left'
            }[compress_sweep]

            for i in range(max(xrange), min(xrange), -1):

                for j in range(min(yrange), max(yrange)+1):
                    tag1 = self.coo_to_tag(i, j)
                    tag2 = self.coo_to_tag(i-1, j)
                    if layer_tag is None:
                        #contract any tensors with these coordinates
                        self.contract_((tag1,tag2), which='any')
                
                    else:
                        #method only exists in `tensor_2d` branch!
                        self.contract_between(tag1, (tag2, layer_tag))
                
                if canonize:
                    self.canonize_row(i, sweep=canonize_sweep, yrange=yrange)

                self.compress_row(i, sweep=compress_sweep, 
                                yrange=yrange, **compress_opts)



    def _contract_boundary_from_bottom_multilayer(
        self, 
        xrange,
        yrange,
        layer_tags=None,
        canonize=True,
        compress_sweep='left',
        **compress_opts
        ):
            canonize_sweep = {
                'left': 'right',
                'right': 'left'
            }[compress_sweep]

            for i in range(max(xrange), min(xrange), -1):
                
                #make sure exterior sites are a single tensor
                for j in range(min(yrange), max(yrange)+1):
                    self ^= (i,j)
                
                for tag in layer_tags:
                    
                    self._contract_boundary_from_bottom_single(
                        xrange=(i, i-1), yrange=yrange, canonize=canonize,
                        compress_sweep=compress_sweep, layer_tag=tag,
                        **compress_opts)

                    for j in range(min(yrange), max(yrange)+1):
                        inner_tag = self.coo_to_tag(i-1, j)

                        #leave the tag on *last* tensor (len==1)
                        if len(self.tag_map[inner_tag]) > 1:
                            self[i,j].drop_tags(inner_tag)


    def contract_boundary_from_bottom(
        self,
        xrange,
        yrange=None,
        canonize=True,
        compress_sweep='right',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()

        if yrange is None:
            yrange = (0, self._Ly-1)
        

        if layer_tags is None:
            tn._contract_boundary_from_bottom_single(
                xrange, yrange, canonize=canonize, 
                compress_sweep=compress_sweep, **compress_opts)
        
        else:
            tn._contract_boundary_from_top_multilayer(
                xrange, yrange, layer_tags, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        
        return tn


    contract_boundary_from_bottom_ = functools.partialmethod(
        contract_boundary_from_bottom, inplace=True)

## FROM LEFT

    def _contract_boundary_from_left_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='up',
        layer_tag=None,
        **compress_opts
    ):
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
        }[compress_sweep]


        for j in range(min(yrange), max(yrange)):
            #
            #     ●──●──       ●──
            #     │  │         ║
            #     ●──●──  ==>  ●──
            #     │  │         ║
            #     ●──●──       ●──
            #
            for i in range(min(xrange), max(xrange) + 1):
                tag1 = self.coo_to_tag(i, j)
                tag2 = self.coo_to_tag(i, j + 1)
                if layer_tag is None:
                    # contract any tensors with coordinates (i, j), (i, j + 1)
                    self.contract_((tag1, tag2), which='any')
                else:
                    # contract a specific pair
                    self.contract_between(tag1, (tag2, layer_tag))
            
            if canonize:
                #
                #     ●──       v──
                #     ║         ║
                #     ●──  ==>  v──
                #     ║         ║
                #     ●──       ●──
                #
                self.canonize_column(j, sweep=canonize_sweep, xrange=xrange)
            #
            #     v──       ●──
            #     ║         │
            #     v──  ==>  ^──
            #     ║         │
            #     ●──       ^──
            #
            self.compress_column(j, sweep=compress_sweep,
                                 xrange=xrange, **compress_opts)        
    
    def _contract_boundary_from_left_multilayer(
        self,
        yrange,
        xrange,
        layer_tags,
        canonize=True,
        compress_sweep='up',
        **compress_opts
    ):
        for j in range(min(yrange), max(yrange)):
            # make sure the exterior sites are a single tensor
            #
            #     ○──○──           ●──○──
            #     │╲ │╲            │╲ │╲       (for two layer tags)
            #     ●─○──○──         ╰─●──○──
            #      ╲│╲╲│╲     ==>    │╲╲│╲
            #       ●─○──○──         ╰─●──○──
            #        ╲│ ╲│             │ ╲│
            #         ●──●──           ╰──●──
            #
            for i in range(min(xrange), max(xrange)+1):
                self ^= (i, j)

            for tag in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #        ○──
                #      ╱╱ ╲        (first contraction if there are two tags)
                #     ●─── ○──
                #      ╲ ╱╱ ╲
                #       ^─── ○──
                #        ╲ ╱╱
                #         ^─────
                #
                self._contract_boundary_from_left_single(
                    yrange=(j, j + 1), xrange=xrange, canonize=canonize,
                    compress_sweep=compress_sweep, layer_tag=tag,
                    **compress_opts)

                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for i in range(min(xrange), max(xrange) + 1):
                    inner_tag = self.coo_to_tag(i, j + 1)
                    if len(self.tag_map[inner_tag]) > 1:
                        self[i, j].drop_tags(inner_tag)


    def contract_boundary_from_left(
        self,
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='up',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()

        if xrange is None:
            xrange = (0, self.Lx - 1)

        if layer_tags is None:
            tn._contract_boundary_from_left_single(
                yrange, xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        else:
            tn._contract_boundary_from_left_multilayer(
                yrange, xrange, layer_tags, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)

        return tn

    contract_boundary_from_left_ = functools.partialmethod(
        contract_boundary_from_left, inplace=True)
    

    def _contract_boundary_from_right_single(
        self,
        yrange,
        xrange,
        canonize=True,
        compress_sweep='down',
        layer_tag=None,
        **compress_opts
    ):
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
        }[compress_sweep]

        for j in range(max(yrange), min(yrange), -1):
            #
            #     ──●──●       ──●
            #       │  │         ║
            #     ──●──●  ==>  ──●
            #       │  │         ║
            #     ──●──●       ──●
            #
            for i in range(min(xrange), max(xrange) + 1):
                tag1, tag2 = self.coo_to_tag(i, j), self.coo_to_tag(i, j - 1)
                if layer_tag is None:
                    # contract any tensors with coordinates (i, j), (i, j - 1)
                    self.contract_((tag1, tag2), which='any')
                else:
                    # contract a specific pair
                    self.contract_between(tag1, (tag2, layer_tag))
            
            if canonize:
                #
                #   ──●       ──v
                #     ║         ║
                #   ──●  ==>  ──v
                #     ║         ║
                #   ──●       ──●
                #
                self.canonize_column(j, sweep=canonize_sweep, xrange=xrange)
            #
            #   ──v       ──●
            #     ║         │
            #   ──v  ==>  ──^
            #     ║         │
            #   ──●       ──^
            #
            self.compress_column(j, sweep=compress_sweep,
                                 xrange=xrange, **compress_opts)

    def _contract_boundary_from_right_multilayer(
        self,
        yrange,
        xrange,
        layer_tags,
        canonize=True,
        compress_sweep='down',
        **compress_opts
    ):
        for j in range(max(yrange), min(yrange), -1):
            # make sure the exterior sites are a single tensor
            #
            #         ──○──○           ──○──●
            #          ╱│ ╱│            ╱│ ╱│    (for two layer tags)
            #       ──○──○─●         ──○──●─╯
            #        ╱│╱╱│╱   ==>     ╱│╱╱│
            #     ──○──○─●         ──○──●─╯
            #       │╱ │╱            │╱ │
            #     ──●──●           ──●──╯
            #
            for i in range(min(xrange), max(xrange) + 1):
                self ^= (i, j)

            for tag in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #         ──○
                #          ╱ ╲╲     (first contraction if there are two tags)
                #       ──○────v
                #        ╱ ╲╲ ╱
                #     ──○────v
                #        ╲╲ ╱
                #     ─────●
                #
                self._contract_boundary_from_right_single(
                    yrange=(j, j - 1), xrange=xrange, canonize=canonize,
                    compress_sweep=compress_sweep, layer_tag=tag,
                    **compress_opts)

                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for i in range(min(xrange), max(xrange) + 1):
                    inner_tag = self.coo_to_tag(i, j - 1)
                    if len(self.tag_map[inner_tag]) > 1:
                        self[i, j].drop_tags(inner_tag)
    
    def contract_boundary_from_right(
        self,
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='down',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        """Contract a 2D tensor network inwards from the left, canonizing and
        compressing (top to bottom) along the way.

        compress_opts
            Supplied to
            :meth:`~compress_column`.
        """
        tn = self if inplace else self.copy()

        if xrange is None:
            xrange = (0, self.Lx - 1)

        if layer_tags is None:
            tn._contract_boundary_from_right_single(
                yrange, xrange, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)
        else:
            tn._contract_boundary_from_right_multilayer(
                yrange, xrange, layer_tags, canonize=canonize,
                compress_sweep=compress_sweep, **compress_opts)

        return tn

    contract_boundary_from_right_ = functools.partialmethod(
        contract_boundary_from_right, inplace=True)


    def contract_boundary(
        self, 
        around=None,
        layer_tags=None,
        max_separation=1,
        sequence=None,
        bottom=None,
        top=None,
        left=None,
        right=None,
        inplace=False,
        **boundary_contract_opts
    ):
        '''
        NOTE: minor changes from Johnnie's code e.g. lattice site conventions 
        change s.t. *upper left* corner is now (0, 0).

        
            ●──●──●──●       ●──●──●──●       ●──●──●
            │  │  │  │       │  │  │  │       ║  │  │
            ●──●──●──●       ●──●──●──●       ^──●──●       >══>══●       >──v
            │  │ij│  │  ==>  │  │ij│  │  ==>  ║ij│  │  ==>  │ij│  │  ==>  │ij║
            ●──●──●──●       ●══<══<══<       ^──<──<       ^──<──<       ^──<
            │  │  │  │
            ●──●──●──●

        Contract boundary inwards, optionally from any or all of the
        boundary, in multiple layers, and/or stopping around a region.
        
        Parameters
        ----------
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        max_separation : int, optional
            If ``around is None``, when any two sides become this far apart
            simply contract the remaining tensor network.
        sequence : sequence of {'b', 'l', 't', 'r'}, optional
            Which directions to cycle throught when performing the inwards
            contractions: 'b', 'l', 't', 'r' corresponding to *from the*
            bottom, left, top and right respectively. If ``around`` is
            specified you will likely need all of these!
        bottom : int, optional
            The initial bottom boundary row, defaults to 0.
        top : int, optional
            The initial top boundary row, defaults to ``Lx - 1``.
        left : int, optional
            The initial left boundary column, defaults to 0.
        right : int, optional
            The initial right boundary column, defaults to ``Ly - 1``..
        inplace : bool, optional
            Whether to perform the contraction in place or not.
        boundary_contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_bottom`,
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_left`,
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_top`,
            or
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_right`,
            including compression and canonization options.
        '''
        tn = self if inplace else self.copy()

        boundary_contract_opts['layer_tags'] = layer_tags

        #set default starting borders
        if bottom is None:
            bottom = tn._Lx-1
        if top is None:
            top = 0
        if left is None:
            left = 0
        if right is None:
            right = tn._Ly - 1
        
        if around is not None:
            if sequence is None:
                sequence = 'bltr'
            stop_i_min = min(x[0] for x in around)
            stop_i_max = max(x[0] for x in around)
            stop_j_max = max(x[1] for x in around)
            stop_j_max = max(x[1] for x in around)
        
        elif sequence is None:
            #contract inwards along short dimension
            if self._Lx >= self._Ly:
                sequence = 't'
            else:
                sequence='l'
        
        # keep track of whether we have hit the ``around`` region.
        reached_stop = {direction: False for direction in sequence}

        for direction in cycle(sequence):

            if direction=='b':
                #check if we have reached 'stop' region
                if (around is None) or (bottom - 1 > stop_i_max):
                    tn.contract_boundary_from_bottom_(
                        xrange=(bottom, bottom - 1),
                        yrange=(left, right),
                        compress_sweep='left',
                        **boundary_contract_opts,
                    )
                    bottom -= 1
            
                else:
                    reached_stop[direction]=True
            
            elif direction == 'l':
                if (around is None) or (left + 1 < stop_j_min):
                    tn.contract_boundary_from_left_(
                        xrange=(bottom, top),
                        yrange=(left, left + 1),
                        compress_sweep='up',
                        **boundary_contract_opts,
                    )
                    left += 1
                else:
                    reached_stop[direction]=True
            
            elif direction == 't':
                if (around is None) or (top + 1 < stop_i_min):
                    tn.contract_boundary_from_top_(
                        xrange=(top, top + 1),
                        yrange=(left, right),
                        compress_sweep='right',
                        **boundary_contract_opts
                    )
                    top += 1
                else:
                    reached_stop[direction] = True
            
            elif direction == 'r':
                if (around is None) or (right - 1 > stop_j_max):
                    tn.contract_boundary_from_right_(
                        xrange=(bottom, top),
                        yrange=(right, right - 1),
                        compress_sweep='down',
                        **boundary_contract_opts,
                    )
                    right -= 1
                else:
                    reached_stop[direction] = True
            
            else:
                raise ValueError("Sequence can only be from bltr")
            
            if around is None:
                # check if TN is thin enough to just contract
                thin_strip = (
                    (top - bottom <= max_separation) or
                    (right - left <= max_separation)
                )
                if thin_strip:
                    return tn.contract(all, optimize='auto-hq')
            
            elif all(reached_stop.values()):
                break
        

        return tn
        
        contract_boundary_ = functools.partialmethod(
            contract_boundary, inplace=True)
        
        
        