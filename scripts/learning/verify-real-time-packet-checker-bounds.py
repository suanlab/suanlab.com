"""Uniform physical-domain size bounds, not an empirical parameter grid.

The interval-engine remainders are proved separately. These inequalities
budget magnitude and conditioning for all accepted packet input values.
"""
import json
from fractions import Fraction as F
from pathlib import Path


def power(exponent):
    return F(2 ** exponent) if exponent >= 0 else F(1, 2 ** -exponent)


minimum = F(1, 999999999999)
maximum = F(16)
assert minimum > power(-40)
delta_min, delta_max = minimum / 32, maximum / 2
assert delta_min > power(-45) and delta_max <= 8
s_max = maximum / delta_min
b_max = delta_max / minimum
assert s_max < power(49) and b_max < power(43)
v_max = maximum * delta_max ** 2
a_max = 1 + v_max / 2
c_scaled_max = v_max * (1 + v_max / 4)
assert max(a_max + 1, c_scaled_max + a_max) < power(20)
scaled_flow = power(20 * 32)
map_entry = scaled_flow * max(s_max, b_max, 1)
assert map_entry < power(689)
variance_min, variance_max = minimum ** 2, maximum ** 2
assert variance_min > power(-80) and variance_max <= power(8)
initial_covariance = max(variance_max, maximum * variance_max,
                         maximum ** 2 * variance_max + maximum ** 2 / (4 * variance_min))
assert initial_covariance < power(88)
final_covariance = 4 * map_entry ** 2 * initial_covariance
assert final_covariance < power(1470)
determinant_min = minimum ** 2 / 4
assert determinant_min > power(-82)
final_variance_min = determinant_min / final_covariance
assert final_variance_min > power(-1552)
eta_min, eta_max = minimum / (2 * variance_max), maximum / (2 * variance_min)
assert eta_min > power(-49) and eta_max < power(83)
# 1=AD-BC implies |A|+|B|>=1/map_entry. Since alpha=A+B*chi+iB*eta,
# |A|+|B| <= |alpha|*(1+(|chi|+1)/eta).
alpha_min = 1 / (map_entry * (1 + (maximum + 1) / eta_min))
assert alpha_min > power(-745)
center_max = 2 * map_entry * maximum
assert center_max < power(695)
sigma_upper = power(735)
action_upper = ((center_max ** 2 + maximum ** 2) / 2
                + center_max * 4 * sigma_upper + final_covariance * 8)
assert action_upper / minimum < power(1516)
# The exact rational cancellation Sigma_qp/V*dx^2=Sigma_qp*offset^2
# avoids a large interval division. Only the linear sqrt(V) term varies.
phase_sensitivity_upper = center_max * 4 / minimum * power(776) + sigma_upper + 16
assert phase_sensitivity_upper < power(1516)
work_bits = 3072
condition_bits = max(1552, 2 * 745, 1516)
guard_bits = 64
effective_bits = work_bits - condition_bits - guard_bits
assert effective_bits >= 512
result = dict(scope='all accepted physical packet inputs;not arbitrary exact rational maps',
              workBits=work_bits, observationBits=512, derivedPhaseMagnitudeBits=1516,
              mapEntryUpperPowerOfTwo=689, covarianceEntryUpperPowerOfTwo=1470,
              finalVarianceLowerPowerOfTwo=-1552, alphaMagnitudeLowerPowerOfTwo=-745,
              initialVarianceLowerPowerOfTwo=-80, phaseSensitivityUpperPowerOfTwo=1516,
              conditioningBits=condition_bits, guardBits=guard_bits, effectivePrecisionBits=effective_bits,
              allAssertionsPassed=True, registeredInputRangeReduced=False)
destination = Path('.runtime/learning/real-time-packet-wave-check')
destination.mkdir(parents=True, exist_ok=True)
(destination / 'physical-bounds.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
