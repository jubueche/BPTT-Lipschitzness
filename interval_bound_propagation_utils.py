from jax import config
config.FLAGS.jax_log_compiles=True
config.update('jax_disable_jit', False)

from jax import jit
import jax.numpy as jnp
from copy import deepcopy
import jax

@jit
def create_interval(X):
    return [X,deepcopy(X)]

@jit
def mat_mul(A,B):
    r_shape = (A[0].shape[0],B[0].shape[1])
    C = create_interval(jnp.empty(r_shape))
    N_cols = r_shape[1]
    def f(i,C):
        mat_vec_tmp = mat_vec(A,(B[0][:,i],B[1][:,i]))
        C[0] = jax.ops.index_update(C[0], jax.ops.index[:,i], mat_vec_tmp[0])
        C[1] = jax.ops.index_update(C[1], jax.ops.index[:,i], mat_vec_tmp[1])
        return C
    C = jax.lax.fori_loop(lower=0, upper=N_cols, body_fun=f, init_val=C)
    return C

@jit
def mat_vec(A,b):
    c = create_interval(jnp.empty(A[0].shape[0]))
    def f(i,c):
        scalar_tmp = vec_vec((A[0][i,:],A[1][i,:]),b)
        c[0] = jax.ops.index_update(c[0],i,scalar_tmp[0])
        c[1] = jax.ops.index_update(c[1],i,scalar_tmp[1])
        return c
    c = jax.lax.fori_loop(lower=0,upper=A[0].shape[0], body_fun=f, init_val=c)
    return c

@jit
def vec_vec(a,b):
    c = create_interval(0.0)
    lower_a,upper_a = a
    lower_b,upper_b = b
    bounds = [jnp.minimum(jnp.minimum(lower_a*lower_b,lower_a*upper_b),
                jnp.minimum(upper_a*lower_b,upper_a*upper_b)),
                jnp.maximum(jnp.maximum(lower_a*lower_b,lower_a*upper_b),
                jnp.maximum(upper_a*lower_b,upper_a*upper_b))]
    def f(i,c):
        c[0] += bounds[0][i]
        c[1] += bounds[1][i]
        return c
    c = jax.lax.fori_loop(lower=0, upper=len(a[0]),body_fun=f, init_val=c)
    return c

@jit
def elem_mult(a,b):
    lower_a,upper_a = a
    lower_b,upper_b = b
    return [jnp.minimum(jnp.minimum(lower_a*lower_b,lower_a*upper_b),jnp.minimum(upper_a*lower_b,upper_a*upper_b)),jnp.maximum(jnp.maximum(lower_a*lower_b,lower_a*upper_b),jnp.maximum(upper_a*lower_b,upper_a*upper_b))]

@jit
def add(A,B):
    lower_a,upper_a = A
    lower_b,upper_b = B
    return [lower_a+lower_b,upper_a+upper_b]

@jit
def neg(A):
    a_lower,a_upper = A
    return [-a_upper,-a_lower]

@jit
def subtract(A,B):
    minus_B = neg(B)
    return add(A,minus_B)

@jit
def less(A,B):
    lower_a,upper_a = A
    lower_b,upper_b = B
    # [a,A] < [b,B] = [A < b, a < B]
    return [(upper_a < lower_b).astype(jnp.float32), (lower_a < upper_b).astype(jnp.float32)]

@jit
def leq(A,B):
    lower_a,upper_a = A
    lower_b,upper_b = B
    return [(upper_a <= lower_b).astype(jnp.float32), (lower_a <= upper_b).astype(jnp.float32)]

@jit
def greater(A,B):
    # A > B == -A < -B
    return less(neg(A),neg(B))

@jit
def geq(A,B):
    return leq(neg(A),neg(B))

@jit
def div(A,B):
    lower_b,upper_b = B
    return elem_mult(A, [1 / upper_b,1 / lower_b])

@jit
def clip_min(A, MIN):
    lower_a,upper_a = A
    lower_min,upper_min = MIN
    lower_a = jnp.where(lower_a <= upper_min, lower_min, lower_a)
    upper_a = jnp.where(upper_a <= upper_min, lower_min, upper_a)
    return [lower_a,upper_a]

@jit
def clip_max(A, MAX):
    lower_a,upper_a = A
    lower_max,upper_max = MAX
    lower_a = jnp.where(lower_a >= lower_max, upper_max, lower_a)
    upper_a = jnp.where(upper_a >= lower_max, upper_max, upper_a)
    return [lower_a,upper_a]

@jit
def clip(A, MIN, MAX):
    A_MIN = clip_min(A, MIN)
    return clip_max(A_MIN, MAX)

if __name__ == "__main__":

    rnd = jax.random.PRNGKey(0)
    rnd, s1, s2, s3 = jax.random.split(rnd, 4)
    a = jnp.abs(jax.random.normal(s1, shape=(20,20)))
    b = jnp.abs(jax.random.normal(s2, shape=(20,1)))
    c = jnp.abs(jax.random.normal(s3, shape=(20,1)))
    A = [a,a]
    B = [-a,-a]
    
    # C = mat_mul(A,[b,b])
    # assert jnp.allclose(C[0],A[0]@b)

    C = mat_mul(A,[-a,-a])
    assert jnp.allclose(C[0],A[0]@(-a))

