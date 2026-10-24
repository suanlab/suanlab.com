import { site } from '@/config/site';
import fs from 'node:fs';
import path from 'node:path';
import { getAllPosts } from '@/lib/blog';

const BASE_URL = site.url;

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

    const imageUrl = post.thumbnail ? new URL(post.thumbnail, BASE_URL) : null;
    const extension = imageUrl?.pathname.split('.').pop()?.toLowerCase();
    const imageType = extension === 'png' ? 'image/png' : extension === 'webp' ? 'image/webp' : 'image/jpeg';
    const imageFile = imageUrl?.origin === BASE_URL ? path.join(process.cwd(), 'public', decodeURIComponent(imageUrl.pathname)) : null;
    const enclosure = imageUrl && imageFile && fs.existsSync(imageFile)
      ? `<enclosure url="${escapeXml(imageUrl.href)}" type="${imageType}" length="${fs.statSync(imageFile).size}" />` : '';
    return `
    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${postUrl}</link>
      <guid isPermaLink="true">${postUrl}</guid>
      <description>${escapeXml(post.excerpt)}</description>
      <pubDate>${pubDate}</pubDate>
      <category>${escapeXml(post.category)}</category>
      ${post.tags.map((tag) => `<category>${escapeXml(tag)}</category>`).join('\n      ')}
      <author>${escapeXml(site.contact.emails[0])} (이수안)</author>
      ${enclosure}
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
    <managingEditor>${escapeXml(site.contact.emails[0])} (이수안)</managingEditor>
    <webMaster>${escapeXml(site.contact.emails[0])} (이수안)</webMaster>
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
