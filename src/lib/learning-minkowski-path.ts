import { intervalSquared, type SpacetimeEvent } from './learning-lorentz-boost';

export const departure: SpacetimeEvent = { ct: 0, x: 0 };
export const reunion: SpacetimeEvent = { ct: 6, x: 0 };

function leg(from: SpacetimeEvent, to: SpacetimeEvent): { squared: number; kind: 'timelike' | 'null' | 'spacelike'; properTime: number | null } {
  const squared = intervalSquared(from, to);
  const kind = squared < 0 ? 'spacelike' : squared === 0 ? 'null' : 'timelike';
  return { squared, kind, properTime: squared < 0 ? null : Math.sqrt(squared) };
}

export function compareMinkowskiPaths(turn: SpacetimeEvent) {
  if (!Number.isFinite(turn.ct) || turn.ct <= departure.ct || turn.ct >= reunion.ct || !Number.isFinite(turn.x)) {
    throw new RangeError('The turnaround must lie between the endpoint coordinate times');
  }
  const outgoing = leg(departure, turn);
  const returning = leg(turn, reunion);
  const properTime = outgoing.properTime === null || returning.properTime === null
    ? null : outgoing.properTime + returning.properTime;
  return { outgoing, returning, properTime, inertialProperTime: reunion.ct };
}
