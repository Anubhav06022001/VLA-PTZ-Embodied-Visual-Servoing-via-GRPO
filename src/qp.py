import numpy as np
from scipy import sparse
import osqp

def cbf_qp_osqp(u_ref, grad_B, B_val, alpha=100.0):
    """
    Solve: min 1/2 ||u - u_ref||^2
           s.t. grad_B * u >= -alpha * B_val
    """
    P = sparse.eye(2, format='csc')
    q_vec = -u_ref
    
    A = sparse.csc_matrix(grad_B.reshape(1, 2))
    l = np.array([-alpha * B_val])
    u = np.array([np.inf])
    
    prob = osqp.OSQP()
    prob.setup(P, q_vec, A, l, u, verbose=False)
    res = prob.solve()
    
    if res.info.status != 'solved':
        return np.zeros(2)
    return res.x