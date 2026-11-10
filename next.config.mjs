/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Static HTML export for GitHub Pages
  // Check production sources here; npm run typecheck also checks all CLI/test code.
  typescript: { tsconfigPath: 'tsconfig.next.json' },
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
