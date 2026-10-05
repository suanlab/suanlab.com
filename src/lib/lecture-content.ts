import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeStringify from 'rehype-stringify';
import type { Slide } from './presentation';
const directory = path.join(process.cwd(), 'content/lectures');
export function getLectureContentSlugs(): string[] {
  return fs.existsSync(directory) ? fs.readdirSync(directory).filter(f => f.endsWith('.md')).map(f => f.slice(0, -3)).sort() : [];
}
export async function parseLectureContent(source: string): Promise<{ title: string; slides: Slide[] }> {
  const { data, content } = matter(source);
  if (typeof data.title !== 'string' || !data.title.trim()) throw new Error('Lecture title is required');
  // Parse top-level thematic breaks, so separators in fenced code are preserved.
  const tree = unified().use(remarkParse).parse(content);
  const boundaries = tree.children.filter(node => node.type === 'thematicBreak').map(node => node.position!);
  const sections: string[] = [];
  let start = 0;
  for (const boundary of boundaries) { sections.push(content.slice(start, boundary.start.offset)); start = boundary.end.offset!; }
  sections.push(content.slice(start));
  const slides = await Promise.all(sections.map(async (section) => {
    const heading = section.trim().match(/^# (.+)\r?\n/);
    if (!heading) throw new Error('Every slide must start with a level-one heading');
    const notes = section.match(/<!-- notes:([\s\S]*?)-->/)?.[1].trim();
    const markdown = section.trim().replace(/^# .+\r?\n/, '').replace(/<!-- notes:[\s\S]*?-->/g, '');
    const html = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype)
      .use(rehypeKatex).use(rehypeHighlight).use(rehypeStringify).process(markdown);
    return { title: heading[1], points: [], contentHtml: String(html), ...(notes ? { notes } : {}) };
  }));
  return { title: data.title, slides };
}
export async function getLectureContent(slug: string) {
  if (!getLectureContentSlugs().includes(slug)) return null;
  return parseLectureContent(fs.readFileSync(path.join(directory, `${slug}.md`), 'utf8'));
}
