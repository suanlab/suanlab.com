import { boostSpacetimeEvent } from './learning-lorentz-boost';

export function calculateCausalRelay(deadlinePosition: number, relayTime: number, relayPosition: number, beta: number) {
  if (![deadlinePosition, relayTime, relayPosition].every(Number.isFinite) || relayTime < 0 || relayTime > 6) {
    throw new RangeError('The relay time must be between the emission and deadline');
  }
  const remaining = 6 - relayTime;
  const lower = Math.max(-relayTime, deadlinePosition - remaining);
  const upper = Math.min(relayTime, deadlinePosition + remaining);
  const reachableFromSender = Math.abs(relayPosition) <= relayTime;
  const reachesDeadline = Math.abs(deadlinePosition - relayPosition) <= remaining;
  return {
    interval: lower <= upper ? { lower, upper } : null,
    reachableFromSender,
    reachesDeadline,
    deadlineTimeInBoostedFrame: boostSpacetimeEvent({ ct: 6, x: deadlinePosition }, beta).ct,
    deadlineCausalType: Math.abs(deadlinePosition) > 6 ? 'spacelike' : Math.abs(deadlinePosition) === 6 ? 'null' : 'timelike',
  };
}
