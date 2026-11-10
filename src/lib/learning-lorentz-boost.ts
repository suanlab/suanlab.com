export interface SpacetimeEvent {
  ct: number;
  x: number;
}

export function boostSpacetimeEvent(event: SpacetimeEvent, beta: number) {
  if (!Number.isFinite(beta) || Math.abs(beta) >= 1 ||
      !Number.isFinite(event.ct) || !Number.isFinite(event.x)) {
    throw new RangeError('Expected finite coordinates and a subluminal boost');
  }
  const gamma = 1 / Math.sqrt((1 - beta) * (1 + beta));
  const rapidity = Math.atanh(beta);
  return {
    gamma, rapidity,
    ct: gamma * (event.ct - beta * event.x),
    x: gamma * (event.x - beta * event.ct),
  };
}

export function intervalSquared(from: SpacetimeEvent, to: SpacetimeEvent) {
  const ct = to.ct - from.ct;
  const x = to.x - from.x;
  return (ct - x) * (ct + x);
}
