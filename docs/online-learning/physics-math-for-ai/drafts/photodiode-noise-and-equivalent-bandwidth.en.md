## 1. Separate an average signal from its fluctuations

A constant light level can produce a steady mean photocurrent and a fluctuating measurement. At fixed wavelength and bias, model independent charge arrivals as a Poisson process. For mean total current $I=I_{\rm photo}+I_{\rm dark}\ge0$, the one-sided white shot-noise current power spectral density is $S_{\rm shot}=2qI$, where $q$ is the elementary charge. Subtracting a dark offset changes the mean while its fluctuations still contribute to the variance. [Hamamatsu, Si Photodiodes technical note, noise characteristics](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf)

## 2. Add resistor noise in the current domain

A load resistor $R$ at absolute temperature $T$ contributes classical Johnson noise. In the equivalent parallel-current representation its one-sided power spectral density is $S_{\rm th}=4k_B T/R$. The units of both spectra are $\mathrm{A^2/Hz}$. Independent zero-mean fluctuations have additive variances, so add these power spectra before taking square roots. A density in $\mathrm{A/\sqrt{Hz}}$ is the square root of the power spectral density.

## 3. Integrate the actual measurement filter

For the passive RC readout, current-to-voltage conversion is $Z(f)=R/(1+i2\pi fRC)$. If both spectra are white over the relevant range, the output variance is

$$
\sigma_v^2=R^2(S_{\rm shot}+S_{\rm th})B_{\rm eq},
\qquad
B_{\rm eq}=\int_0^\infty\frac{df}{1+(f/f_c)^2}=\frac{\pi}{2} f_c=\frac{1}{4RC}.
$$

The equivalent bandwidth is the width of a rectangle with height one and the same area as the normalized squared transfer function. It exceeds the −3 dB cutoff because the filter passes some noise above that frequency. The integration uses positive frequencies with the one-sided density convention.

## 4. Check what changes when resistance doubles

Substituting $B_{\rm eq}$ gives the shot contribution $qIR/(2C)$ and the thermal contribution $k_BT/C$. With fixed $C$ and $T$, doubling $R$ changes the thermal spectrum and bandwidth while their combined output variance stays constant. The shot contribution doubles at fixed current. With $I=1\,\mathrm{\mu A}$, $R=100\,\mathrm{k\Omega}$, $C=50\,\mathrm{pF}$, and $T=300\,\mathrm K$, the shot and thermal RMS voltages are about $12.7\,\mathrm{\mu V}$ and $9.10\,\mathrm{\mu V}$. Their combined RMS is about $15.6\,\mathrm{\mu V}$.

## 5. Reproduce the variance with a numerical integral

```python
from math import atan, isclose, pi, sqrt

q, k_B = 1.602176634e-19, 1.380649e-23
current, resistance, capacitance, temperature = 1e-6, 100e3, 50e-12, 300
cutoff = 1 / (2 * pi * resistance * capacitance)
s_shot = 2 * q * current
s_thermal = 4 * k_B * temperature / resistance

n = 20000
df = 100 * cutoff / n
partial = sum(df / (1 + (((j + .5) * df) / cutoff)**2) for j in range(n))
tail = cutoff * (pi / 2 - atan(100))
bandwidth = partial + tail
assert isclose(bandwidth, 1 / (4 * resistance * capacitance), rel_tol=1e-8)
variance = resistance**2 * (s_shot + s_thermal) * bandwidth
assert isclose(variance, q * current * resistance / (2 * capacitance) + k_B * temperature / capacitance, rel_tol=1e-8)
print(f"cutoff={cutoff/1e3:.2f} kHz, ENBW={bandwidth/1e3:.2f} kHz, RMS={sqrt(variance)*1e6:.2f} µV")
```

## 6. Use residuals to find missing noise sources

Measure dark and illuminated traces at several light levels with the same bias, sampling rate, and readout settings. Under the white-noise model, increasing mean photocurrent raises output variance linearly. A residual at low frequency may indicate drift or excess low-frequency noise. Saturation and quantization alter the measurement in different ways, so record clipping, range, and sample resolution. Carry these measurements into a dataset's uncertainty model before treating the observed variation as variation in the scene.
