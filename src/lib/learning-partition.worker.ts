import { evaluatePartitionRequest } from './learning-partition-evaluator';
import { partitionRecord, isPartitionInput, type PartitionStarted } from './learning-partition-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value: unknown = event.data;
  if (partitionRecord(value) && value.version === 1 && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 && isPartitionInput(value.input)) {
    const started: PartitionStarted = { version: 1, requestId: value.requestId as number, phase: 'started', kind: value.input.kind };
    scope.postMessage(started);
  }
  scope.postMessage(evaluatePartitionRequest(value));
};
