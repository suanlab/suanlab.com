import fs from 'node:fs';
import path from 'node:path';

/** Build workers reuse one snapshot. Development checks file stats for edits. */
export function createContentStore<T>(directory: string, parse: (filename: string, source: string) => T) {
  const entries = new Map<string, { signature: string; value: T }>();
  let snapshot: string[] | undefined;
  const production = process.env.NODE_ENV === 'production';

  function filenames(): string[] {
    if (production && snapshot) return snapshot.slice();
    const names = fs.existsSync(directory) ? fs.readdirSync(directory).filter(name => name.endsWith('.md')).sort() : [];
    for (const name of entries.keys()) if (!names.includes(name)) entries.delete(name);
    snapshot = names;
    return names.slice();
  }

  function get(filename: string): T | null {
    if (path.basename(filename) !== filename || !filename.endsWith('.md')) return null;
    const cached = entries.get(filename);
    if (production && cached) return cached.value;
    const fullPath = path.join(directory, filename);
    let stat: fs.Stats;
    try { stat = fs.statSync(fullPath); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      entries.delete(filename);
      return null;
    }
    if (!stat.isFile()) return null;
    const signature = `${stat.mtimeMs}:${stat.ctimeMs}:${stat.size}`;
    if (cached?.signature === signature) return cached.value;
    const value = parse(filename, fs.readFileSync(fullPath, 'utf8'));
    entries.set(filename, { signature, value });
    return value;
  }

  return { filenames, get, all: () => filenames().map(get).filter((value): value is T => value !== null) };
}
