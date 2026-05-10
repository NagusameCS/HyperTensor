import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
y = rng.standard_normal(9800)
tau, p = stats.kendalltau(np.arange(len(y)), y)
print(f'pure noise n=9800: tau={tau:+.4f} p={p:.4f}')
y2 = rng.standard_normal(9800) + np.linspace(0, 0.05, 9800)
tau2, p2 = stats.kendalltau(np.arange(len(y2)), y2)
print(f'tiny linear trend (0.05/n): tau={tau2:+.4f} p={p2:.4f}')
y3 = 0.11 - 0.1*np.exp(-np.arange(9800)/1000) + rng.standard_normal(9800)*0.001
tau3, p3 = stats.kendalltau(np.arange(len(y3)), y3)
print(f'saturating exp + small noise: tau={tau3:+.4f} p={p3:.4f}')
# Now mass-saturated with many ties
y4 = np.full(9800, 0.11)
y4[:200] = np.linspace(0, 0.11, 200)
y4 += rng.standard_normal(9800)*1e-6
tau4, p4 = stats.kendalltau(np.arange(len(y4)), y4)
print(f'fully saturated (heavy ties): tau={tau4:+.4f} p={p4:.4f}')
print()
print('Volume claim: tau=0.018, p=0.015, n~9800 -- consistent only with heavy ties.')
print('Critical: by the scripts own threshold (p_mk > 0.01), p=0.015 PASSES convergence;')
print('but a standard reader interpreting p<0.05 as significant would flag this as a TREND,')
print('contradicting the volume phrasing "confirms no post-saturation trend".')
