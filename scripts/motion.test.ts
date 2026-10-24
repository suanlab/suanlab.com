import { test } from 'node:test';
import assert from 'node:assert/strict';
import { publicationYearCounts } from '../src/components/motion/publication-timeline-data';
import { publications } from '../src/data/publications';

test('publication timeline counts source records and keeps years chronological', () => {
  const records = publications.map(publication => ({ sortYear: Number(publication.date.match(/\b(19|20)\d{2}\b/)?.[0] ?? 0) }));
  const years = publicationYearCounts(records);
  assert.equal(years.reduce((sum, item) => sum + item.count, 0), publications.length);
  assert.deepEqual(years.map(item => item.year), [...years.map(item => item.year)].sort((a, b) => a - b));
  assert.equal(years.find(item => item.year === 2026)?.count, records.filter(item => item.sortYear === 2026).length);
});

test('timeline has no invented empty years or malformed year buckets', () => {
  const years = publicationYearCounts([{ sortYear: 2026 }, { sortYear: 2024 }, { sortYear: 2026 },
    { sortYear: 0 }, { sortYear: NaN }, { sortYear: 2025.5 }, { sortYear: Infinity }]);
  assert.deepEqual(years, [{ year: 2024, count: 1 }, { year: 2026, count: 2 }]);
  assert.deepEqual(publicationYearCounts([]), []);
});
