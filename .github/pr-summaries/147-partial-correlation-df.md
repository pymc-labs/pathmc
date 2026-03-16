# PR: Fix partial correlation test degrees of freedom

Closes #147

## Issue Summary

`_partial_correlation_test` in `identify.py` used `scipy.stats.pearsonr` on residuals, which internally uses `df = n - 2`. The correct degrees of freedom for a partial correlation are `df = n - k - 2` where k is the number of conditioning variables, leading to anti-conservative (too small) p-values.

## Root Cause

`stats.pearsonr` does not account for the degrees of freedom consumed by regressing out the conditioning variables. It always uses `df = n - 2`, but after partialing out k variables the residuals have only `n - k - 2` effective degrees of freedom.

## Solution

Replaced `stats.pearsonr(resid_x, resid_y)` with a manual t-test that uses the correct `df = n - k - 2`:

```python
r = np.corrcoef(resid_x, resid_y)[0, 1]
df = n - k - 2
t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r**2)
p = 2.0 * stats.t.sf(np.abs(t_stat), df)
```

The unconditional case (no conditioning variables) still uses `stats.pearsonr` which is correct when `k = 0`.

## Changes Made

- `pathmc/identify.py`: Replace `stats.pearsonr` on residuals with manual t-test using correct df in `_partial_correlation_test`

## Testing

- [x] Existing tests pass (388 passed)
- [x] Manual verification: confirmed corrected df produces more conservative p-values for k > 0

## Notes

- The error magnitude scales with `k/n`. For k=1 and large n the difference is negligible, but for k=5 and n=20 the old code used df=18 instead of the correct df=13.
- The unconditional branch (line 566) is unchanged — `pearsonr` is correct when there are no conditioning variables.
