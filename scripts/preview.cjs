const http = require('node:http');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');

const root = fs.realpathSync('out');
const port = Number(process.env.PORT || 4175);
const mime = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8', '.json': 'application/json',
  '.xml': 'application/xml; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.pdf': 'application/pdf', '.webmanifest': 'application/manifest+json',
};
const insideRoot = filename => filename === root || filename.startsWith(root + path.sep);
http.createServer(async (req, res) => {
  // Like public GitHub Pages assets, custom comment styles may be read cross-origin.
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405, { Allow: 'GET, HEAD, OPTIONS' }); res.end(); return; }
  try {
    const url = new URL(req.url, 'http://localhost');
    let filename = path.resolve(root, '.' + decodeURIComponent(url.pathname));
    if (!insideRoot(filename)) { res.writeHead(403); res.end(); return; }
    let status = 200;
    let stat;
    try { stat = await fsp.stat(filename); }
    catch (error) {
      if (!['ENOENT', 'ENOTDIR'].includes(error.code)) throw error;
      filename = path.join(root, '404.html'); status = 404; stat = await fsp.stat(filename);
    }
    if (stat.isDirectory()) {
      if (!url.pathname.endsWith('/')) { res.writeHead(308, { Location: url.pathname + '/' + url.search }); res.end(); return; }
      filename = path.join(filename, 'index.html'); stat = await fsp.stat(filename);
    }
    if (!insideRoot(await fsp.realpath(filename)) || !stat.isFile()) { res.writeHead(403); res.end(); return; }
    res.writeHead(status, { 'Content-Type': mime[path.extname(filename)] || 'application/octet-stream', 'Content-Length': stat.size });
    if (req.method === 'HEAD') res.end();
    else fs.createReadStream(filename).on('error', () => res.destroy()).pipe(res);
  } catch (error) {
    if (!res.headersSent) res.writeHead(error instanceof URIError ? 400 : 404);
    res.end();
  }
}).listen(port, '127.0.0.1', () => console.log(`Static preview: http://127.0.0.1:${port}`));
