#!/usr/bin/env python3
"""
02_bivariate_probit.py
Joint bivariate probit (full ML, correlated errors) with country and sector fixed effects
and RECount, country-clustered sandwich SEs, TRUE average marginal effects (delta method),
and cross-equation Wald tests (beta and AME) using the JOINT covariance matrix.

Bivariate normal CDF: Plackett identity with Gauss-Legendre quadrature (vectorised).
Output: Data/Processed/BiProbit_Joint_FE.xlsx (sheets: coef, cross_eq_tests, AME, diagnostics)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "Processed", "SME_Analytical_Final.csv")
OUT = os.path.join(ROOT, "Data", "Processed", "BiProbit_Joint_FE.csv")

IV = ['B2B_private', 'B2G_public', 'B2C', 'barrier_admin', 'barrier_expertise',
      'grant_receipt', 'advice_public']
CTRL = ['scr10', 'scr14', 're_action_count']
GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def bvn_cdf(h, k, rho):
    """Phi2(h,k;rho) via Plackett: Phi(h)Phi(k) + int_0^rho phi2(h,k;r) dr."""
    r = 0.5 * rho[:, None] * (GL_X[None, :] + 1.0)
    w = 0.5 * rho[:, None] * GL_W[None, :]
    hh, kk = h[:, None], k[:, None]
    one = 1.0 - r ** 2
    dens = np.exp(-(hh ** 2 - 2 * r * hh * kk + kk ** 2) / (2 * one)) / (2 * np.pi * np.sqrt(one))
    return stats.norm.cdf(h) * stats.norm.cdf(k) + (w * dens).sum(1)


def obs_ll(theta, X, y1, y2):
    k = X.shape[1]
    b1, b2, rho = theta[:k], theta[k:2 * k], np.tanh(theta[2 * k])
    q1, q2 = 2 * y1 - 1, 2 * y2 - 1
    p = bvn_cdf(q1 * (X @ b1), q2 * (X @ b2), q1 * q2 * rho * np.ones(len(y1)))
    return np.log(np.clip(p, 1e-300, None))


def scores(theta, X, y1, y2, h=1e-5):
    g = np.empty((len(y1), len(theta)))
    for j in range(len(theta)):
        e = np.zeros(len(theta)); e[j] = h
        g[:, j] = (obs_ll(theta + e, X, y1, y2) - obs_ll(theta - e, X, y1, y2)) / (2 * h)
    return g


def build(df):
    d = df[IV + CTRL + ['q9_binary', 'q14_binary', 'country_iso2', 'nace_b']].dropna().copy()
    X = pd.concat([d[IV + CTRL].astype(float),
                   pd.get_dummies(d['country_iso2'], prefix='c', drop_first=True, dtype=float),
                   pd.get_dummies(d['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)], axis=1)
    X.insert(0, 'const', 1.0)
    return d, X


def ames(theta, X, names, vars_):
    k = X.shape[1]
    b1, b2 = theta[:k], theta[k:2 * k]
    out = []
    for v in vars_:
        j = names.index(v)
        X1, X0 = X.copy(), X.copy(); X1[:, j] = 1; X0[:, j] = 0
        out.append([(stats.norm.cdf(X1 @ b) - stats.norm.cdf(X0 @ b)).mean() for b in (b1, b2)])
    return np.array(out)  # (len(vars), 2)


def main():
    df = pd.read_csv(DATA)
    d, Xdf = build(df)
    names = list(Xdf.columns)
    X = Xdf.values
    y1, y2 = d['q9_binary'].values.astype(float), d['q14_binary'].values.astype(float)
    k = X.shape[1]
    print(f"N = {len(d):,}; k = {k} per equation (country FE + sector FE + RECount)")

    # start from separate probit (statsmodels)
    import statsmodels.api as sm
    s1 = sm.Probit(y1, X).fit(disp=0, maxiter=200)
    s2 = sm.Probit(y2, X).fit(disp=0, maxiter=200)
    th0 = np.concatenate([s1.params, s2.params, [0.0]])
    f = lambda t: -obs_ll(t, X, y1, y2).sum()
    gr = lambda t: -scores(t, X, y1, y2).sum(0)
    res = minimize(f, th0, jac=gr, method='BFGS', options={'maxiter': 400, 'gtol': 1e-7})
    th = res.x
    print(f"converged={res.success}, ll={-res.fun:.2f}, rho={np.tanh(th[-1]):.4f}, |grad|max={np.abs(gr(th)).max():.2e}")

    # 24-pt vs 48-pt Gauss-Legendre quadrature accuracy check
    gl48_x, gl48_w = np.polynomial.legendre.leggauss(48)
    b1_hat, b2_hat, rho_hat = th[:k], th[k:2 * k], np.tanh(th[2 * k])
    q1_v, q2_v = 2 * y1 - 1, 2 * y2 - 1
    h_v, k_v, r_v = q1_v * (X @ b1_hat), q2_v * (X @ b2_hat), q1_v * q2_v * rho_hat * np.ones(len(y1))
    p24 = bvn_cdf(h_v, k_v, r_v)
    r48 = 0.5 * r_v[:, None] * (gl48_x[None, :] + 1.0)
    w48 = 0.5 * r_v[:, None] * gl48_w[None, :]
    one48 = 1.0 - r48 ** 2
    dens48 = np.exp(-(h_v[:, None] ** 2 - 2 * r48 * h_v[:, None] * k_v[:, None] + k_v[:, None] ** 2) / (2 * one48)) / (2 * np.pi * np.sqrt(one48))
    p48 = stats.norm.cdf(h_v) * stats.norm.cdf(k_v) + (w48 * dens48).sum(1)
    quad_diff_max = float(np.max(np.abs(np.log(np.clip(p24, 1e-300, None)) - np.log(np.clip(p48, 1e-300, None)))))

    # Hessian by finite-diff of the summed score; cluster sandwich
    n_par = len(th); H = np.empty((n_par, n_par)); h = 1e-4
    for j in range(n_par):
        e = np.zeros(n_par); e[j] = h
        H[:, j] = (-scores(th + e, X, y1, y2).sum(0) + scores(th - e, X, y1, y2).sum(0)) / (2 * h) * -1
    H = -(0.5 * (H + H.T))  # observed information = -d2 ll
    ev = np.linalg.eigvalsh(H)
    print(f"Hessian min eigenvalue = {ev.min():.3f}, max eigenvalue = {ev.max():.1f}, cond = {ev.max()/ev.min():.1f} ({'PD' if ev.min() > 0 else 'NOT PD'})")
    Hi = np.linalg.inv(H)
    S = scores(th, X, y1, y2)
    cl = d['country_iso2'].values
    G = np.vstack([S[cl == c].sum(0) for c in np.unique(cl)])
    G_c = len(G)
    meat = G.T @ G * (G_c / (G_c - 1)) * ((len(d) - 1) / (len(d) - n_par))
    V = Hi @ meat @ Hi
    V_rob = Hi @ (S.T @ S) @ Hi
    se = np.sqrt(np.diag(V))

    df_c = G_c - 1  # finite-cluster degrees of freedom (df = 27 - 1 = 26)
    rows = []
    for eq, off in [('Y1_Q9_green_offerings', 0), ('Y2_Q14_netzero_strategy', k)]:
        for v in IV + CTRL:
            j = names.index(v) + off
            t_stat = th[j] / se[j]
            rows.append(dict(
                eq=eq, var=v, beta=th[j], se_cluster=se[j], se_robust=np.sqrt(V_rob[j, j]),
                z_cluster=t_stat,
                p_cluster=2 * (1 - stats.norm.cdf(abs(t_stat))),
                p_t26=2 * (1 - stats.t.cdf(abs(t_stat), df=df_c))
            ))
    coef = pd.DataFrame(rows)

    # beta cross-equation Wald (joint covariance)
    tests = []
    for v in IV:
        j = names.index(v)
        d_b = th[j] - th[k + j]
        var = V[j, j] + V[k + j, k + j] - 2 * V[j, k + j]
        se_d = np.sqrt(var)
        z_d = d_b / se_d
        tests.append(dict(
            var=v, test='beta_diff', diff=d_b, se=se_d, z=z_d,
            p=2 * (1 - stats.norm.cdf(abs(z_d))),
            p_t26=2 * (1 - stats.t.cdf(abs(z_d), df=df_c))
        ))

    # true AMEs, delta method (numerical Jacobian)
    a0 = ames(th, X, names, IV)
    J = np.zeros((len(IV) * 2, n_par)); h = 1e-5
    for j in range(2 * k):
        e = np.zeros(n_par); e[j] = h
        diff = (ames(th + e, X, names, IV) - ames(th - e, X, names, IV)) / (2 * h)
        J[:, j] = diff.reshape(-1)
    Vam = J @ V @ J.T
    am_rows = []
    for i, v in enumerate(IV):
        i1, i2 = 2 * i, 2 * i + 1
        dd = a0[i, 0] - a0[i, 1]
        vd = Vam[i1, i1] + Vam[i2, i2] - 2 * Vam[i1, i2]
        se_dd = np.sqrt(vd)
        z_dd = dd / se_dd
        am_rows.append(dict(
            var=v, AME_Q9=a0[i, 0], se_Q9=np.sqrt(Vam[i1, i1]),
            AME_Q14=a0[i, 1], se_Q14=np.sqrt(Vam[i2, i2]),
            AME_diff=dd, se_AME_diff=se_dd,
            p_AME_diff=2 * (1 - stats.norm.cdf(abs(z_dd))),
            p_AME_diff_t26=2 * (1 - stats.t.cdf(abs(z_dd), df=df_c))
        ))
        tests[i]['p_AME_diff'] = am_rows[-1]['p_AME_diff']
        tests[i]['p_AME_diff_t26'] = am_rows[-1]['p_AME_diff_t26']
    tdf = pd.DataFrame(tests)

    # Holm adjustment over the beta-difference family (both normal and t_26)
    for col_in, col_out in [('p', 'p_holm'), ('p_t26', 'p_t26_holm')]:
        p = tdf[col_in].values; order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
        for r, idx in enumerate(order):
            run = max(run, (m - r) * p[idx]); adj[idx] = min(1, run)
        tdf[col_out] = adj

    # rho Wald (LR-free): delta on tanh
    rho = np.tanh(th[-1]); se_rho = (1 - rho ** 2) * np.sqrt(V[-1, -1])
    grad_max = float(np.abs(gr(th)).max())
    cond_H = float(ev.max() / ev.min()) if ev.min() > 0 else np.nan
    print(f"rho = {rho:.4f} (cluster SE {se_rho:.4f}); clusters = {G_c} (df = {df_c})")
    am = pd.DataFrame(am_rows)
    print(coef.round(4).to_string()); print(tdf.round(4).to_string()); print(am.round(4).to_string())
    coef.to_csv(OUT, index=False)
    with pd.ExcelWriter(OUT.replace('.csv', '.xlsx')) as w:
        coef.to_excel(w, sheet_name='coef', index=False)
        tdf.to_excel(w, sheet_name='cross_eq_tests', index=False)
        am.to_excel(w, sheet_name='AME', index=False)
        pd.DataFrame([dict(
            N=len(d), k_per_eq=k, total_params=n_par, clusters=G_c, df_cluster=df_c,
            rho=rho, se_rho=se_rho, z_rho=rho / se_rho,
            ll=-res.fun, converged=res.success, grad_inf_norm=grad_max,
            hess_min_eig=float(ev.min()), hess_max_eig=float(ev.max()), hess_cond_num=cond_H,
            quad_diff_24_vs_48=quad_diff_max
        )]).to_excel(w, sheet_name='diagnostics', index=False)


if __name__ == '__main__':
    main()

