## 1. Follow light through a measurement chain

A silicon photodiode can convert absorbed photons into mobile electron–hole pairs. In the depletion region an electric field separates the carriers and the external circuit reads a current. In a linear operating range at fixed wavelength, write the incremental photocurrent as $i_{\rm ph}=\mathcal{R}p$, where $p$ is optical power modulation and responsivity $\mathcal{R}$ has units A/W. Responsivity, junction capacitance, and dark current describe different aspects of the sensor. [Hamamatsu, Si Photodiodes technical note](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf)

## 2. Derive the passive readout

Consider a small-signal current source in parallel with a load resistance $R$ and effective capacitance $C$. The latter includes junction and input capacitance. The load dominates the shunt resistance, and the bias operating point is held fixed. Current through the capacitor is $C\dot v$ and current through the load is $v/R$, so

$$
RC\,\dot v+v=R\mathcal{R}p.
$$

A light step changes the input immediately. Capacitor voltage changes continuously while charge accumulates. For a step $p_0$ starting from zero, $v(t)=R\mathcal{R}p_0(1-e^{-t/(RC)})$. The final voltage increases with $R$, and the 10–90% rise time is $RC\ln 9$.

## 3. Read gain and phase from a frequency sweep

For sinusoidal modulation at frequency $f$, the optical-to-voltage gain is

$$
G(f)=\frac{R\mathcal{R}}{1+i2\pi fRC}.
$$

The relative amplitude is $1/\sqrt{1+(2\pi fRC)^2}$. At $f_c=1/(2\pi RC)$, the relative amplitude is $1/\sqrt 2$ (about −3 dB) and the phase is −45°. Doubling the resistance doubles the low-frequency gain and halves the cutoff. Halving capacitance instead doubles the cutoff while the low-frequency gain stays the same in this model.

With $R=100\,\mathrm{k\Omega}$, $C=50\,\mathrm{pF}$ and $\mathcal{R}=0.5\,\mathrm{A/W}$, the time constant is $5\,\mathrm{\mu s}$ and the cutoff is about $31.8\,\mathrm{kHz}$. A $10\,\mathrm{\mu W}$ step eventually gives $0.5\,\mathrm V$ and the 10–90% rise takes about $11\,\mathrm{\mu s}$.

## 4. Identify which part of a real circuit limits the response

Changing the reverse bias can change capacitance, collection time and dark current. A measured step can also include the light source's rise time, carrier transport, stray capacitance and the digitizer's filter. Measure the input waveform or use a calibrated source. Record bias, wavelength, load, and gain settings. Change one element at a time to see whether the measured cutoff follows the RC prediction. An active transimpedance amplifier uses feedback and has a different bandwidth and stability calculation. [Texas Instruments, Photodiode amplifier circuit](https://www.ti.com/tool/CIRCUIT060018)

## 5. Compare the step and frequency domains in Python

```python
from math import exp, isclose, log, pi, sqrt

responsivity = 0.5
optical_step = 10e-6

for resistance, capacitance in [(100e3, 50e-12), (200e3, 50e-12), (100e3, 25e-12)]:
    tau = resistance * capacitance
    cutoff = 1 / (2 * pi * tau)
    t10, t90 = -tau * log(0.9), -tau * log(0.1)
    final = resistance * responsivity * optical_step
    voltage = lambda t: final * (1 - exp(-t / tau))
    assert isclose(voltage(t10) / final, 0.1)
    assert isclose(voltage(t90) / final, 0.9)
    assert isclose(abs(1 / (1 + 2j * pi * cutoff * tau)), 1 / sqrt(2))
    print(f"final={final:.3f} V, cutoff={cutoff / 1e3:.1f} kHz, rise={(t90-t10)*1e6:.1f} µs")
```

## 6. Plan a calibration for an AI data pipeline

Suppose a model predicts an optical pulse from the sensor's digitized voltage. Two short pulses can yield similar outputs if the readout suppresses their high-frequency components. Record sampling rate and the analog transfer function along with the voltage traces. Test generalization with a changed load or capacitance. The model then has to account for the measurement response while inferring properties of the incident light.
