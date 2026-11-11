/** Forward scalar recurrence and discrete adjoint of its terminal objective. */
export function calculateDiscreteAdjoint(
  initial: number,
  persistence: number,
  controls: readonly number[],
  target: number,
  penalty: number,
) {
  if (![initial, persistence, target, penalty].every(Number.isFinite) ||
      !controls.every(Number.isFinite) || penalty < 0) {
    throw new RangeError('Expected finite states and controls, and a nonnegative input penalty');
  }
  const states = [initial];
  let controlCost = 0;
  for (const control of controls) {
    states.push(persistence * states[states.length - 1] + control);
    controlCost += control ** 2;
  }
  const terminalError = states[controls.length] - target;
  const adjoint = new Array<number>(states.length);
  const controlGradient = new Array<number>(controls.length);
  adjoint[controls.length] = terminalError;
  let persistenceGradient = 0;
  for (let i = controls.length - 1; i >= 0; i--) {
    const next = adjoint[i + 1];
    adjoint[i] = persistence * next;
    controlGradient[i] = next + penalty * controls[i];
    persistenceGradient += next * states[i];
  }
  return { states, adjoint, controlGradient, persistenceGradient, loss: .5 * (terminalError ** 2 + penalty * controlCost) };
}
