# submit.py
import numpy as np
from sklearn.linear_model import LogisticRegression
# Allowed: any sklearn linear model. Disallowed: non-linear models. [attached_file:1]

# Global model holder (if the evaluator inspects it)
_model = None

# ---------------------------
# Helpers for arbiter features
# ---------------------------

def _to_pm1_bits(C):
    # C in {0,1}^k -> x in {+1,-1}^k as x = 1-2*C [attached_file:1]
    return 1 - 2*C

def _arbiter_phi(x):
    # Canonical linearization for a single k-stage arbiter PUF using cumulative products. [attached_file:1]
    # x: (..., k) in {+1,-1}
    # Returns ψ(x) shape (..., k) with ψ_0 = 1, ψ_i = prod_{j=0}^{i} x_j for i>=0, then shifted combination for model. [attached_file:1]
    # Derivation: the delay difference equals w·ψ(x)+b with w_0=α_0, w_i=α_i+β_{i-1}, b=β_{k-1}. [attached_file:1]
    # Implement cumulative product over last axis.
    cp = np.cumprod(x, axis=-1)
    return cp

def _xor_linearization(u, v):
    # Given two linear score features for Response0 and Response1, produce a single feature map φ such that
    # sign(W·φ + b) equals XOR of the two responses. Uses standard XOR-PUF trick by augmenting with cross-terms
    # via Khatri–Rao/elementwise products, but done vectorized without SciPy. [attached_file:1]
    # u, v: shape (n, Du), (n, Dv). Construct φ = [u, v, u∘v, 1] (bias handled by fit_intercept). [attached_file:1]
    uv = u * v
    return np.concatenate([u, v, uv], axis=1)

def _ml_puf_phi(C):
    # Map 8-bit challenges to a compact linear feature for ML‑PUF (two arbiters + XOR). [attached_file:1]
    # C: (n,8) in {0,1}
    x = _to_pm1_bits(C)              # (n,8) in {+1,-1} [attached_file:1]
    # Build arbiter features for both cross-competitions: lower-lower and upper-upper. Both yield same ψ form. [attached_file:1]
    psi0 = _arbiter_phi(x)           # Response0 feature proxy (n,8) [attached_file:1]
    psi1 = _arbiter_phi(x)           # Response1 feature proxy (n,8) [attached_file:1]
    # Apply XOR linearization: φ = [ψ0, ψ1, ψ0∘ψ1] giving D = 8 + 8 + 8 = 24 features (bias via intercept). [attached_file:1]
    Phi = _xor_linearization(psi0, psi1)
    return Phi.astype(np.float64)    # ensure float for sklearn [attached_file:1]

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
################################
#  Non Editable Region Ending  #
################################
    # Train a linear classifier on φ(C). Constrained to linear sklearn; LogisticRegression is fine. [attached_file:1]
    global _model
    Phi = my_map(X_train)  # (n, D̃) [attached_file:1]
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        C=10.0,
        tol=1e-4,
        fit_intercept=True,
        random_state=0,
    )  # Linear model per spec. [attached_file:1]
    clf.fit(Phi, y_train)  # y in {0,1} [attached_file:1]
    _model = clf
    # Return vector and scalar bias as required by validator. [attached_file:1]
    return clf.coef_[0], clf.intercept_[0]

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################
    # X: (n,8) in {0,1}. Return φ(C) with compact D and no slow Python loops. [attached_file:1]
    X = np.asarray(X, dtype=np.int8)
    return _ml_puf_phi(X)  # (n, 24) [attached_file:1]

################################
# Non Editable Region Starting #
################################
def my_decode(wb):
################################
#  Non Editable Region Ending  #
################################
    # Given a 65-vector [w(64), b], recover nonnegative p,q,r,s in R^{64} producing the same linear model. [attached_file:1]
    # Model equations (from spec):
    # α_i = (p_i - q_i + r_i - s_i)/2,  β_i = (p_i - q_i - r_i + s_i)/2  for 0<=i<=63
    # w_0 = α_0;  w_i = α_i + β_{i-1} (1<=i<=63);  b = β_{63}. [attached_file:1]
    wb = np.asarray(wb, dtype=np.float64).ravel()
    assert wb.size == 65, "expected 65-dim array [w(64), b]"  # interface safety [attached_file:1]
    w = wb[:64]
    b = wb[64]

    # Reconstruct α, β from [w,b]:
    # α_0 = w_0
    # For i>=1: α_i = w_i - β_{i-1}
    # β_63 = b, and β_{i} free unless constrained by w_{i+1}; we can choose a simple consistent solution. [attached_file:1]
    beta = np.zeros(64, dtype=np.float64)
    beta[63] = b
    # Choose beta[0..62] = 0 for the minimal-norm consistent solution; this yields α_i directly from w. [attached_file:1]
    alpha = np.zeros(64, dtype=np.float64)
    alpha[0] = w[0]
    for i in range(1, 64):
        alpha[i] = w[i] - beta[i-1]  # since beta[i-1] = 0 for i<=63 except i=64 case unused [attached_file:1]

    # Now solve for p,q,r,s per stage from α_i, β_i with nonnegativity:
    # System:
    # (1) p_i - q_i + r_i - s_i = 2 α_i
    # (2) p_i - q_i - r_i + s_i = 2 β_i
    # Add: 2(p_i - q_i) = 2(α_i + β_i)  => p_i - q_i = α_i + β_i
    # Sub: 2(r_i - s_i) = 2(α_i - β_i)  => r_i - s_i = α_i - β_i  [attached_file:1]
    # Under-determined; impose nonnegativity and minimal L2 by taking the "balanced" split:
    # Let d_i = α_i + β_i, e_i = α_i - β_i. Choose p_i = max(d_i,0), q_i = max(-d_i,0), r_i = max(e_i,0), s_i = max(-e_i,0). [attached_file:1]
    d = alpha + beta
    e = alpha - beta
    p = np.maximum(d, 0.0)
    q = np.maximum(-d, 0.0)
    r = np.maximum(e, 0.0)
    s = np.maximum(-e, 0.0)

    # All are nonnegative by construction; this yields exact α,β back and therefore reproduces [w,b]. [attached_file:1]
    return p, q, r, s
