import { getAllPosts } from '@/lib/blog';

const SITE_URL = 'https://suanlab.com';

function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function generateRssItem(post: {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
}): string {
  const pubDate = new Date(post.date).toUTCString();
  const link = `${SITE_URL}/blog/${post.slug}`;

  return `
    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${link}</link>
      <guid isPermaLink="true">${link}</guid>
      <description>${escapeXml(post.excerpt)}</description>
      <pubDate>${pubDate}</pubDate>
      <category>${escapeXml(post.category)}</category>
      ${post.tags.map((tag) => `<category>${escapeXml(tag)}</category>`).join('\n      ')}
    </item>`;
}

function generateRssFeed(posts: Array<{
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
}>): string {
  const lastBuildDate = posts.length > 0
    ? new Date(posts[0].date).toUTCString()
    : new Date().toUTCString();

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>SuanLab Blog</title>
    <link>${SITE_URL}/blog</link>
    <description>데이터 과학, 인공지능, 딥러닝에 관한 이야기를 공유하는 SuanLab 블로그입니다.</description>
    <language>ko</language>
    <lastBuildDate>${lastBuildDate}</lastBuildDate>
    <atom:link href="${SITE_URL}/blog/feed.xml" rel="self" type="application/rss+xml"/>
    <image>
      <url>${SITE_URL}/assets/images/logo.png</url>
      <title>SuanLab Blog</title>
      <link>${SITE_URL}/blog</link>
    </image>
    ${posts.map(generateRssItem).join('\n')}
  </channel>
</rss>`;
}

export async function GET() {
  const posts = getAllPosts();
  const feed = generateRssFeed(posts);

  return new Response(feed, {
    headers: {
      'Content-Type': 'application/xml; charset=utf-8',
      'Cache-Control': 'public, max-age=3600, s-maxage=3600',
    },
  });
}
