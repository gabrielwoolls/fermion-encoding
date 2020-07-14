import quimb as qu
import numpy as np
from itertools import product, chain
import quimb.tensor as qtn


def make_qubit_TN(Lx, Ly, CHI=5, return_tags=False, show_graph=False):
    vtags, ftags = dict(), dict()

    
    # vtensors = []
    # for i,j in product(range(Lx), range(Ly)):
        
    #     vtensors.append(qtn.Tensor(tags={'VERT', f'Q{i*Ly+j}'}))
    #     vtags[(i,j)] = f'Q{i*Ly+j}'

    vtensors = [[qtn.Tensor(tags={'VERT'}) 
                for j in range(Ly)] for i in range(Lx)]

    # ftags[(i,j)] = f'Q{k+Lx*Ly}'
    ftensors = np.ndarray(shape=(Lx-1,Ly-1), dtype=object)
    for i, j in product(range(Lx-1), range(Ly-1)):
        if i%2 == j%2:
            ftensors[i,j] = qtn.Tensor(tags='FACE')
    
    for i,j in product(range(Lx), range(Ly)):
        vtensors[i][j].new_ind(f'q{i*Ly+j}',size=2)
        vtensors[i][j].add_tag(f'Q{i*Ly+j}')
        vtags[(i,j)] = f'Q{i*Ly+j}'

        if i<=Lx-2:
            vtensors[i][j].new_bond(vtensors[i+1][j],size=CHI)
        if j<=Ly-2:
            vtensors[i][j].new_bond(vtensors[i][j+1],size=CHI)

    k=0    
    for i, j in product(range(Lx-1), range(Ly-1)):
        if not ftensors[i,j] is None:
            ftensors[i,j].new_ind(f'q{k+Lx*Ly}',size=2)
            ftensors[i,j].add_tag(f'Q{k+Lx*Ly}')
            ftags[(i,j)] = f'Q{k+Lx*Ly}'

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
            **{(vtags[(i,j)]): (LAT_CONST*j, -LAT_CONST*i) for i,j in product(range(Lx),range(Ly))}
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


# class Lattice2D():

#     def __init__(self, Lx, Ly, psi0):
#         self.ham = getLocalHam(Lx,Ly) #LocalHam2D
#         self._psi = psi0.copy() #TensorNetwork2DVector


#     def compute_energy(self, state):
#         return state.compute_local_expectation(self.ham.terms)

#     def gate(self, U, where):
#         """Perform single gate ``U`` at coordinate pair ``where``.
#         """
#         self._psi.gate_(U, where)

class MyQubitTN(qtn.TensorNetwork):

    def __init__(self, Lx, Ly, chi):

        vtags, ftags, tensors = make_qubit_TN(Lx, Ly, chi, return_tags=True)
        self._vtags = vtags
        self._ftags = ftags
    
        super().__init__(tensors)


    def vertex_tag(self,i,j):
        return self._vtags[(i,j)]
    
    def face_tag(self,i,j):
        return self._ftags[(i,j)]
