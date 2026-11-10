import { parseNumberDraft } from './learning-number-input';

/** Preserve lexical number evidence before JSON.parse rounds small values or counts. */
export function parseNumberJsonDraft(source: string, integerKeys: readonly string[] = []): unknown {
  const value: unknown = JSON.parse(source);
  // JSON syntax is checked first. Quoted strings, including escaped quotes, stay intact.
  const lexical: unknown = JSON.parse(source.replace(
    /"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g,
    token => token.startsWith('"') ? token : JSON.stringify(token),
  ));
  const integers = new Set(integerKeys);
  function check(current: unknown, raw: unknown, integer = false): void {
    if (typeof current === 'number') {
      if (typeof raw !== 'string' || parseNumberDraft(raw, -Number.MAX_VALUE, Number.MAX_VALUE, integer) === null)
        throw new RangeError('A JSON number is nonfinite, underflows to zero or violates an integer constraint');
    } else if (current !== null && typeof current === 'object') {
      const rawObject = raw as Record<string, unknown>;
      for (const [key, child] of Object.entries(current))
        check(child, rawObject[key], integer || integers.has(key));
    }
  }
  check(value, lexical);
  return value;
}
