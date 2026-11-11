/** Integrate timestamped power readings with linear interpolation. */
export function calculateTaskEnergy(times: readonly number[], powers: readonly number[], tasks: number, idlePower: number) {
  if (times.length < 2 || times.length !== powers.length || !Number.isInteger(tasks) || tasks < 1 ||
      !Number.isFinite(idlePower) || idlePower < 0 || !times.every(Number.isFinite) ||
      !powers.every(value => Number.isFinite(value) && value >= 0)) {
    throw new RangeError('Expected aligned samples, a positive integer task count, and nonnegative power');
  }
  let gross = 0;
  for (let i = 1; i < times.length; i++) {
    const duration = times[i] - times[i - 1];
    if (duration <= 0) throw new RangeError('Timestamps must increase strictly');
    gross += duration * (powers[i - 1] / 2 + powers[i] / 2);
  }
  const duration = times[times.length - 1] - times[0];
  const incremental = gross - idlePower * duration;
  return { duration, gross, incremental, grossPerTask: gross / tasks, incrementalPerTask: incremental / tasks };
}
