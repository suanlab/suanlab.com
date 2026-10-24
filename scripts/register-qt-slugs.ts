import fs from 'node:fs';
import { registerQTSlugs } from '../src/lib/qt-slugs';

const file = 'src/config/qt-slugs.json';
const existing = JSON.parse(fs.readFileSync(file, 'utf8')) as Record<string, string>;
const filenames = fs.readdirSync('content/qt').filter(name => name.endsWith('.md'));
fs.writeFileSync(file, JSON.stringify(registerQTSlugs(filenames, existing), null, 2) + '\n');
console.log(`Registered ${filenames.length} QT files; existing and retired URLs preserved.`);
