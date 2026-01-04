import { getAllPosts } from '@/lib/blog';

const BASE_URL = 'https://suanlab.com';

function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export async function GET() {
  const posts = getAllPosts();

  const rssItems = posts.slice(0, 50).map((post) => {
    const postUrl = `${BASE_URL}/blog/${post.slug}`;
    const pubDate = new Date(post.date).toUTCString();

    return `
    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${postUrl}</link>
      <guid isPermaLink="true">${postUrl}</guid>
      <description>${escapeXml(post.excerpt)}</description>
      <pubDate>${pubDate}</pubDate>
      <category>${escapeXml(post.category)}</category>
      ${post.tags.map((tag) => `<category>${escapeXml(tag)}</category>`).join('\n      ')}
      <author>suan@suanlab.com (이수안)</author>
      ${post.thumbnail ? `<enclosure url="${BASE_URL}${post.thumbnail}" type="image/jpeg" />` : ''}
    </item>`;
  });

  const rss = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:atom="http://www.w3.org/2005/Atom"
  xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>SuanLab Blog</title>
    <link>${BASE_URL}/blog</link>
    <description>이수안 교수의 데이터 사이언스 블로그 - AI, 딥러닝, 머신러닝, 자연어처리, 컴퓨터 비전</description>
    <language>ko</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${BASE_URL}/feed.xml" rel="self" type="application/rss+xml"/>
    <image>
      <url>${BASE_URL}/assets/images/logo.png</url>
      <title>SuanLab Blog</title>
      <link>${BASE_URL}/blog</link>
    </image>
    <managingEditor>suan@suanlab.com (이수안)</managingEditor>
    <webMaster>suan@suanlab.com (이수안)</webMaster>
    <copyright>Copyright ${new Date().getFullYear()} SuanLab. All rights reserved.</copyright>
    <ttl>60</ttl>
    ${rssItems.join('\n')}
  </channel>
</rss>`;

  return new Response(rss, {
    headers: {
      'Content-Type': 'application/xml; charset=utf-8',
      'Cache-Control': 'public, max-age=3600, s-maxage=3600',
    },
  });
}
