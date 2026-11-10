/** Decimal input for finite Number models. Never turn a nonzero underflow into zero. */
export function parseNumberDraft(raw: string, min: number, max: number, integer = false): number | null {
  const text = raw.trim();
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text)) return null;
  const value = Number(text), [mantissa, exponent = '0'] = text.toLowerCase().split('e');
  if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) return null;
  if (value === 0 && /[1-9]/.test(mantissa)) return null;
  if (integer) {
    const places = (mantissa.split('.')[1]?.length ?? 0) - Number(exponent);
    if (places > 0 && /[1-9]/.test(mantissa.replace(/\D/g, '').slice(-places))) return null;
  }
  return value === 0 ? 0 : value;
}
