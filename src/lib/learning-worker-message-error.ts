/** Register decode-error events on native Workers and minimal injected ports. */
type MessageErrorPort = Pick<Worker, 'onmessageerror'> & {
  addEventListener?(type: 'messageerror', listener: EventListener): void;
  removeEventListener?(type: 'messageerror', listener: EventListener): void;
};
const listeners = new WeakMap<MessageErrorPort, EventListener>();

export function clearWorkerMessageError(port: MessageErrorPort): void {
  const listener = listeners.get(port);
  if (listener) {
    port.removeEventListener?.('messageerror', listener);
    listeners.delete(port);
  }
  port.onmessageerror = null;
}

export function bindWorkerMessageError(port: MessageErrorPort, callback: () => void): void {
  clearWorkerMessageError(port);
  // Some native Workers expose EventTarget without an onmessageerror IDL setter.
  if (port.addEventListener && port.removeEventListener) {
    const listener: EventListener = () => callback();
    port.addEventListener('messageerror', listener);
    listeners.set(port, listener);
  } else {
    port.onmessageerror = callback;
  }
}
