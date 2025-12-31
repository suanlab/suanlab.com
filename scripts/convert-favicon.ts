import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';

const PUBLIC_DIR = path.join(process.cwd(), 'public');

async function convertFavicon(): Promise<void> {
  const sourceIcon = path.join(PUBLIC_DIR, 'icon-512.png');

  console.log('Converting icon to favicon sizes...');

  // Generate different sizes
  const sizes = [16, 32, 48, 64, 128, 180, 192, 512];

  for (const size of sizes) {
    const outputPath = path.join(PUBLIC_DIR, `favicon-${size}x${size}.png`);
    await sharp(sourceIcon)
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toFile(outputPath);
    console.log(`Created: favicon-${size}x${size}.png`);
  }

  // Create apple-touch-icon (180x180)
  await sharp(sourceIcon)
    .resize(180, 180, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toFile(path.join(PUBLIC_DIR, 'apple-touch-icon.png'));
  console.log('Created: apple-touch-icon.png (180x180)');

  // Create favicon.ico (multi-size ICO file using 16, 32, 48)
  // Sharp doesn't support ICO directly, so we'll create individual PNGs
  // and use the 32x32 as the primary favicon

  // For favicon.ico, we'll just copy the 32x32 version
  // In production, you'd want to use a proper ICO generator
  const favicon32 = await sharp(sourceIcon)
    .resize(32, 32, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toBuffer();

  // Create a simple ICO file structure (single 32x32 image)
  const icoBuffer = await createIco([
    { size: 16, buffer: await sharp(sourceIcon).resize(16, 16).png().toBuffer() },
    { size: 32, buffer: await sharp(sourceIcon).resize(32, 32).png().toBuffer() },
    { size: 48, buffer: await sharp(sourceIcon).resize(48, 48).png().toBuffer() },
  ]);

  await fs.writeFile(path.join(PUBLIC_DIR, 'favicon.ico'), icoBuffer);
  console.log('Created: favicon.ico (16, 32, 48)');

  // Create site.webmanifest
  const manifest = {
    name: 'SuanLab',
    short_name: 'SuanLab',
    description: 'Data Science & AI Research Lab',
    icons: [
      { src: '/favicon-192x192.png', sizes: '192x192', type: 'image/png' },
      { src: '/favicon-512x512.png', sizes: '512x512', type: 'image/png' },
    ],
    theme_color: '#1e40af',
    background_color: '#ffffff',
    display: 'standalone',
  };

  await fs.writeFile(
    path.join(PUBLIC_DIR, 'site.webmanifest'),
    JSON.stringify(manifest, null, 2)
  );
  console.log('Created: site.webmanifest');

  console.log('\nAll favicon files generated successfully!');
}

// Create a simple ICO file from PNG buffers
async function createIco(images: { size: number; buffer: Buffer }[]): Promise<Buffer> {
  // ICO file format:
  // - Header (6 bytes)
  // - Directory entries (16 bytes each)
  // - Image data (PNG format)

  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // Reserved
  header.writeUInt16LE(1, 2); // Type: 1 = ICO
  header.writeUInt16LE(images.length, 4); // Number of images

  const directoryEntries: Buffer[] = [];
  const imageData: Buffer[] = [];
  let dataOffset = 6 + images.length * 16;

  for (const img of images) {
    const entry = Buffer.alloc(16);
    entry.writeUInt8(img.size === 256 ? 0 : img.size, 0); // Width
    entry.writeUInt8(img.size === 256 ? 0 : img.size, 1); // Height
    entry.writeUInt8(0, 2); // Color palette
    entry.writeUInt8(0, 3); // Reserved
    entry.writeUInt16LE(1, 4); // Color planes
    entry.writeUInt16LE(32, 6); // Bits per pixel
    entry.writeUInt32LE(img.buffer.length, 8); // Image size
    entry.writeUInt32LE(dataOffset, 12); // Data offset

    directoryEntries.push(entry);
    imageData.push(img.buffer);
    dataOffset += img.buffer.length;
  }

  return Buffer.concat([header, ...directoryEntries, ...imageData]);
}

convertFavicon()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('Error:', err);
    process.exit(1);
  });
