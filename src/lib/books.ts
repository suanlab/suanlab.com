import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';

export interface BookPost {
  slug: string;
  title: string;
  subtitle?: string;
  author: string;
  publisher?: string;
  date: string;
  image?: string;
  url?: string;
  content: string;
}

export interface BookPostWithHtml extends BookPost {
  contentHtml: string;
}

const booksDirectory = path.join(process.cwd(), 'content/books');

export function getBookSlugs(): string[] {
  if (!fs.existsSync(booksDirectory)) {
    return [];
  }
  return fs.readdirSync(booksDirectory).filter((file) => file.endsWith('.md'));
}

export function getBookBySlug(slug: string): BookPost | null {
  const realSlug = slug.replace(/\.md$/, '');
  const fullPath = path.join(booksDirectory, `${realSlug}.md`);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);

  return {
    slug: realSlug,
    title: data.title || '',
    subtitle: data.subtitle,
    author: data.author || '',
    publisher: data.publisher,
    date: data.date || '',
    image: data.image,
    url: data.url,
    content,
  };
}

export async function getBookBySlugWithHtml(slug: string): Promise<BookPostWithHtml | null> {
  const book = getBookBySlug(slug);
  if (!book) return null;

  const processedContent = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype)
    .use(rehypeSlug)
    .use(rehypeStringify)
    .process(book.content);

  const contentHtml = processedContent.toString();

  return {
    ...book,
    contentHtml,
  };
}

export function getAllBookPosts(): BookPost[] {
  const slugs = getBookSlugs();
  const books = slugs
    .map((slug) => getBookBySlug(slug.replace(/\.md$/, '')))
    .filter((book): book is BookPost => book !== null)
    .sort((a, b) => (new Date(b.date) > new Date(a.date) ? 1 : -1));

  return books;
}

export async function getAllBookPostsWithHtml(): Promise<BookPostWithHtml[]> {
  const slugs = getBookSlugs();
  const booksPromises = slugs.map(async (slug) => {
    return await getBookBySlugWithHtml(slug.replace(/\.md$/, ''));
  });

  const books = await Promise.all(booksPromises);
  return books
    .filter((book): book is BookPostWithHtml => book !== null)
    .sort((a, b) => (new Date(b.date) > new Date(a.date) ? 1 : -1));
}
