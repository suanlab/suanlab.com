/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Static HTML export for GitHub Pages
  images: {
    unoptimized: true,  // Required for static export
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'map2.daum.net',
      },
      {
        protocol: 'https',
        hostname: 'www.youtube.com',
      },
      {
        protocol: 'https',
        hostname: 'img.youtube.com',
      },
    ],
  },
  trailingSlash: true,
  // basePath: '/repo-name',  // GitHub Pages 서브경로 사용 시 활성화
};

export default nextConfig;
