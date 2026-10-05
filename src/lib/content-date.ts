/** Reject rollover dates such as February 30 rather than silently normalizing. */
export function isCalendarDate(value: unknown): boolean {
  const date = value instanceof Date && !Number.isNaN(value.valueOf()) ? value.toISOString().slice(0, 10) : value;
  if (typeof date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return false;
  const parsed = new Date(`${date}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === date;
}
