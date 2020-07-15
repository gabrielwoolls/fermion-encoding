import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn
import spinlessQubit
from quimb.tensor.tensor_1d import maybe_factor_gate_into_tensor
    

def make_qubit_TN(Lx, Ly, CHI=5, return_tags=False, show_graph=False):
    vtags, ftags = dict(), dict()
    

    vtensors = [[qtn.Tensor(tags={'VERT'}) 
                for j in range(Ly)] for i in range(Lx)]

    ftensors = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    for i, j in product(range(Lx-1), range(Ly-1)):
        if i%2 == j%2:
            ftensors[i,j] = qtn.Tensor(tags='FACE')
    
    for i,j in product(range(Lx), range(Ly)):
        vtensors[i][j].new_ind(f'q{i*Ly+j}',size=2)
        vtensors[i][j].add_tag(f'Q{i*Ly+j}')

        # vtags[(i,j)] = f'Q{i*Ly+j}'

        if i<=Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=CHI)
        if j<=Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=CHI)

    k=0    
    for i, j in product(range(Lx-1), range(Ly-1)):
        if not ftensors[i,j] is None:
            ftensors[i,j].new_ind(f'q{k+Lx*Ly}',size=2)
            ftensors[i,j].add_tag(f'Q{k+Lx*Ly}')
            # ftags[(i,j)] = f'Q{k+Lx*Ly}'

            ftensors[i,j].new_bond(vtensors[i][j],size=CHI)
            ftensors[i,j].new_bond(vtensors[i][j+1],size=CHI)
            ftensors[i,j].new_bond(vtensors[i+1][j+1],size=CHI)
            ftensors[i,j].new_bond(vtensors[i+1][j],size=CHI)
            k+=1

    alltensors = vtensors + [f for f in ftensors.flatten().tolist() if not f is None]
    vtn = qtn.TensorNetwork(alltensors)

    if show_graph:
        LAT_CONST = 50 #lattice constant for graphing

        fix = {
            **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
            }
        vtn.graph(color=['VERT','FACE'], show_tags=True, fix=fix)


    if return_tags:
        return vtags, ftags, vtn


    return vtn


# class OnsiteHam2D():

#     def __init__(Lx, Ly, H1):
#         '''
#         H1 can be dictionary mapping sites (i,j) to arrays, 
#         or a single array (becomes default local term for 
#         every site.)

#         '''
#         self.Lx = int(Lx)
#         self.Ly = int(Ly)

#         if hasattr(H1, 'shape'):
#             H1s = {None: H1}
#         else:
#             H1s = dict(H1)
        
#         or key, X in H1s.items():
#             if isinstance(X, qarray):
#                 H1s[key] = X.A

#         # possibly set the default single site term
#         default_H1 = H1s.pop(None, None)
#         if default_H1 is not None:
#             for i, j in product(range(self.Lx), range(self.Ly)):
#                 H1s.setdefault((i, j), default_H1)


class MyQubitTN(qtn.TensorNetwork):

    def __init__(self, Lx, Ly, chi):
        
        verts, faces = spinlessQubit.gen_lattice_sites(Lx,Ly)
        
        tensors = make_qubit_TN(Lx, Ly, chi)
        
        self._vert_coo_map = np.ndenumerate(verts)
        self._face_coo_map = np.ndenumerate(faces)
        
        self._psi = tensors

        self._Lx = Lx
        self._Ly = Ly

        self._edge_map = spinlessQubit.get_edge_map(verts, faces)
        
        self._site_tag_id = 'Q{}'
        self._phys_ind_id = 'q{}'
        self._d_physical = 2

        super().__init__(tensors)


    def vert_coo_map(self, i, j):
        return self._vert_coo_map[(i,j)]
    
    def face_coo_map(self, i, j):
        return self._face_coo_map[(i,j)]

    

    def vertex_coo_tag(self,i,j):
        return f'Q{self.vert_coo_map(i,j)}'
    
    def vertex_coo_ind(self,i,j):
        return f'q{self.vert_coo_map(i,j)}'



    def face_coo_tag(self,i,j):
        return f'Q{self.face_coo_map(i,j)}'

    def face_coo_ind(self,i,j):
        return f'q{self.face_site_map(i,j)}'


    def get_edges(self, key):
        '''
        Returns: list[tuple(int or None)]
            List of (three-tuple) edges,
            where edge (i,j,f) denotes vertices i,j 
            (ints) and face f  (int or None)

        key: {'u','d','r','l','r+l'}
            String specifying what edges of
            the graph to return.

        '''
        if key == 'r+l':
            return self._edge_map['r']+self._edge_map['l']
        
        else:
            return self._edge_map[key]



    def graph_psi(self, show_tags=False, auto=False):
        
        if auto:
            self._psi.graph(color=['VERT','FACE','GATE'])
        
        else:
            LAT_CONST = 50 #lattice constant for graphing
            Lx,Ly = self._Lx, self._Ly

            fix = {
                **{(f'Q{i*Ly+j}'): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
                }
            self._psi.graph(color=['VERT','FACE','GATE'], show_tags=show_tags, fix=fix)


    def apply_gate(self, psi, G, where, inplace=False):
        '''Apply gate `G` at sites specified in `where`,
        preserving physical indices.

        Params:
        psi: TensorNetwork
            
        G : array
            Gate to apply, should be compatible with 
            shape ``(.....)``
        
        where: sequence of ints
            The site labels on which to act, using the 
            default numbering that includes face and vertex
            sites.
        '''
        # psi = self._psi if inplace else self._psi.copy()
        psi = psi if inplace else psi.copy()

        if isinstance(where, int): 
            where = (where,)

        numsites = len(where) #gate is `numsites`-local
        dp = self._d_physical

        G = maybe_factor_gate_into_tensor(G, dp, numsites, where)

        #new physical indices
        site_inds = [f'q{i}' for i in where] 
        #old physical indices joined to new gate
        bond_inds = [qtn.rand_uuid() for _ in range(numsites)]
        #replace physical inds with gate/bond inds
        reindex_map = dict(zip(site_inds, bond_inds))

        TG = qtn.Tensor(G, inds=site_inds+bond_inds, left_inds=bond_inds, tags=['GATE'])
        
        psi.reindex_(reindex_map)
        psi |= TG
        return psi


    def make_norm_tensor(self, psi=None):
        '''
        Return <psi|psi> as a TensorNetwork.
        '''

        if psi is None:
            psi = self._psi

        ket = psi.copy()
        ket.add_tag('KET')

        bra = ket.retag({'KET':'BRA'})
        bra.conj_()

        return ket | bra


    def compute_hop_expecs(self, psi=None):
        '''
        Return <psi|H_hop|psi> expectation for the
        hopping terms in (qubit) Hubbard
        '''
        if psi is None: 
            psi = self._psi

        E_hop = 0
        bra = psi.H

        X,Y,Z = (qu.pauli(mu) for mu in ['x','y','z'])

        for (i,j,f) in self.get_edges('r+l'):
            Of = Y #operator to act on face qbit if it exists

            if f is None:
                G = 0.5 * (X&X + Y&Y)
                G_ket = self.apply_gate(psi, G, where=(i,j))
            
            else:
                G = 0.5 * (X&X&Of + Y&Y&Of)
                G_ket = self.apply_gate(psi, G, where=(i,j,f))
         
            E_hop += bra|G_ket ^ all
        
        ## DOWN

        for (i,j,f) in self.get_edges('d'):
            Of = X #operator to act on face qbit if it exists
            
            if f is None:
                G = 0.5 * (X&X + Y&Y)
                G_ket = self.apply_gate(psi, G, where=(i,j))
            
            else:
                G = 0.5 * (X&X&Of + Y&Y&Of)
                G_ket = self.apply_gate(psi, G, where=(i,j,f))

            E_hop += bra|G_ket ^ all

        ## UP

        for (i,j,f) in self.get_edges('u'):
            Of = X #operator to act on face qbit if it exists
            
            if f is None:
                G = 0.5 * (X&X + Y&Y)
                G_ket = self.apply_gate(psi, G, where=(i,j))
            
            else:
                G = 0.5 * (X&X&Of + Y&Y&Of)
                G_ket = self.apply_gate(psi, G, where=(i,j,f))

            E_hop += bra|G_ket ^ all

        return E_hop            

    
    # def compute_nnint_expecs(self, psi=None):
    #     '''
    #     Return <psi|H_int|psi> for the nearest-neighbor
    #     repulsion terms in 1D-Hubbard.
    #     '''
    #     if psi is None: 
    #         psi = self._psi



        




     