#!/usr/bin/env python3
"""
05_robustness.py
Robustness checks re-estimated under the corrected specification (country + sector FE, RECount,
country-clustered SEs). Equation-by-equation probits (statsmodels) unless noted.
 1. Probability-scale moderation (public financial support x barriers), cluster-robust delta method
 2. Strict Q14 (excludes already climate-neutral firms)
 3. Survey-weighted probits (w1_sme)
 4. Sector-stratified probits (country FE within sector)
 5. Models without RECount (possible post-treatment control)
 6. Routed-only sample (Q5.3 support seekers) for public financial support
Output: Data/Processed/Robustness_FE.xlsx (sheets: robustness, moderation_clustered)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from patsy import build_design_matrices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "Data", "Processed", "SME_Analytical_Final.csv"))
OUT = os.path.join(ROOT, "Data", "Processed", "Robustness_FE.xlsx")
IV = ['B2B_private', 'B2G_public', 'B2C', 'barrier_admin', 'barrier_expertise', 'grant_receipt', 'advice_public']
BASE = ' + '.join(IV) + ' + scr10 + scr14 + re_action_count + C(country_iso2) + C(nace_b)'
NOREC = ' + '.join(IV) + ' + scr10 + scr14 + C(country_iso2) + C(nace_b)'
need = IV + ['scr10', 'scr14', 're_action_count', 'country_iso2', 'nace_b', 'q9_binary', 'q14_binary']


def probit(y, formula_rhs, d, weights=None):
    f = f"{y} ~ {formula_rhs}"
    if weights is None:
        m = smf.probit(f, d).fit(disp=0, maxiter=200, cov_type='cluster', cov_kwds={'groups': pd.factorize(d['country_iso2'])[0]})
    else:
        m = smf.glm(f, d, family=__import__('statsmodels.api').api.families.Binomial(link=__import__('statsmodels.api').api.families.links.Probit()),
                    freq_weights=d[weights]).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['country_iso2'])[0]})
    return m


def tidy(m, label, vars_=IV):
    rows = []
    for v in vars_:
        if v in m.params.index:
            rows.append(dict(spec=label, var=v, beta=m.params[v], se=m.bse[v], p=m.pvalues[v], N=int(m.nobs)))
    return rows


rows = []
d0 = df.dropna(subset=need).copy()
# main reference (separate probits, same spec as 02b)
for y in ['q9_binary', 'q14_binary']:
    rows += [dict(r, outcome=y) for r in tidy(probit(y, BASE, d0), 'Reference (sep. probit, FE, clustered)')]
    # 5. without RECount
    dn = df.dropna(subset=[c for c in need if c != 're_action_count']).copy()
    rows += [dict(r, outcome=y) for r in tidy(probit(y, NOREC, dn), 'No RECount')]
    # 3. weighted
    dw = d0.dropna(subset=['w1_sme'])
    rows += [dict(r, outcome=y) for r in tidy(probit(y, BASE, dw, 'w1_sme'), 'Survey-weighted (w1_sme)')]
    # 6. routed-only: firms with non-missing raw q6.1 before zero-fill are not recoverable; use support seekers q5.3==1 where available
    if 'q5.3' in df.columns:
        dr = d0[d0['q5.3'] == 1]
        if len(dr) > 500:
            rows += [dict(r, outcome=y) for r in tidy(probit(y, BASE, dr), 'Routed only (Q5.3=1)')]
# 2. strict Q14
ds = d0.copy()
if 'q14_strict' in ds.columns:
    ds = ds.dropna(subset=['q14_strict'])
    rows += [dict(r, outcome='q14_strict') for r in tidy(probit('q14_strict', BASE, ds), 'Strict Q14')]
# 4. sector stratified
for s, g in d0.groupby('nace_b'):
    rhs = ' + '.join(IV) + ' + scr10 + scr14 + re_action_count + C(country_iso2)'
    for y in ['q9_binary', 'q14_binary']:
        try:
            rows += [dict(r, outcome=y) for r in tidy(probit(y, rhs, g), f'Sector nace_b={int(s)}')]
        except Exception as e:
            print('sector fail', s, y, e)
rob = pd.DataFrame(rows)

# 1. probability-scale moderation, cluster-robust covariance
req = need + ['scr13']
sub = df.dropna(subset=[c for c in req if c in df.columns]).copy()
f = ('q14_binary ~ grant_receipt * barrier_admin + grant_receipt * barrier_expertise + B2B_private + B2G_public + B2C '
     '+ scr10 + scr14 + re_action_count + C(country_iso2) + C(nace_b)')
m = smf.logit(f, sub).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(sub['country_iso2'])[0]})
di, cov, beta = m.model.data.design_info, m.cov_params().values, m.params.values


def pg(g, a=None, e=None):
    d = sub.copy(); d['grant_receipt'] = g
    if a is not None: d['barrier_admin'] = a
    if e is not None: d['barrier_expertise'] = e
    X = np.asarray(build_design_matrices([di], d)[0]); p = expit(X @ beta)
    return p.mean(), ((p * (1 - p))[:, None] * X).mean(0)


mod = []
for dim, kw in [('Admin', 'a'), ('Expertise', 'e')]:
    P = {(g, b): pg(g, **{kw: b}) for g in (0, 1) for b in (0, 1)}
    for lab, est, gr in [
        (f'AME(support | {dim}=0)', P[1, 0][0] - P[0, 0][0], P[1, 0][1] - P[0, 0][1]),
        (f'AME(support | {dim}=1)', P[1, 1][0] - P[0, 1][0], P[1, 1][1] - P[0, 1][1]),
        (f'Delta-AME ({dim}=1 vs 0)', (P[1, 1][0] - P[0, 1][0]) - (P[1, 0][0] - P[0, 0][0]),
         (P[1, 1][1] - P[0, 1][1]) - (P[1, 0][1] - P[0, 0][1]))]:
        se = float(np.sqrt(gr @ cov @ gr))
        mod.append(dict(quantity=lab, est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
                        p=2 * (1 - stats.norm.cdf(abs(est / se))), N=int(m.nobs)))
mod = pd.DataFrame(mod)
with pd.ExcelWriter(OUT) as w:
    rob.to_excel(w, sheet_name='robustness', index=False)
    mod.to_excel(w, sheet_name='moderation_clustered', index=False)
pd.set_option('display.width', 200)
print(mod.round(4).to_string())
piv = rob[rob['var'].isin(['B2B_private', 'B2G_public', 'B2C', 'barrier_admin', 'barrier_expertise', 'grant_receipt', 'advice_public'])]
print(piv.assign(b=lambda x: x.beta.round(3), pp=lambda x: x.p.round(3))[['spec', 'outcome', 'var', 'b', 'pp', 'N']].to_string())

