import { execSync } from 'child_process';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';

const BLOG_IMAGE_DIR = path.join(process.cwd(), 'public/assets/images/blog');

interface FigureExtractionOptions {
  pdfBuffer: Buffer;
  slug: string;
  targetPage?: number;  // Default: auto-detect (usually page 2-3 for figures)
  cropRatio?: number;   // Crop bottom portion (0.3 = bottom 30%)
}

interface ExtractionResult {
  success: boolean;
  imagePath?: string;
  error?: string;
}

/**
 * Extract a figure from PDF and save as blog thumbnail
 * Prioritizes pages 1-3 where Figure 1 is typically located
 * Uses high DPI and extracts bottom portion of page 1 or full figure from page 2-3
 */
export async function extractFigureFromPdf(
  options: FigureExtractionOptions
): Promise<ExtractionResult> {
  const { pdfBuffer, slug, targetPage } = options;
  const tempDir = os.tmpdir();
  const tempPdfPath = path.join(tempDir, `${slug}-temp.pdf`);
  const tempImageBase = path.join(tempDir, `${slug}-page`);

  try {
    // Save PDF to temp file
    await fs.writeFile(tempPdfPath, pdfBuffer);

    // Convert first 4 pages at high resolution
    const pagesToTry = targetPage ? [targetPage] : [1, 2, 3, 4];
    const pageImages: { page: number; path: string; buffer: Buffer }[] = [];

    for (const page of pagesToTry) {
      try {
        // Convert PDF page to PNG using pdftoppm at 200 DPI
        const outputPath = `${tempImageBase}-${page}`;
        execSync(
          `pdftoppm -png -f ${page} -l ${page} -r 200 "${tempPdfPath}" "${outputPath}"`,
          { encoding: 'utf-8', timeout: 30000 }
        );

        // Find the generated file
        const imageFile = await findGeneratedImage(outputPath, page);
        if (imageFile) {
          const buffer = await fs.readFile(imageFile);
          pageImages.push({ page, path: imageFile, buffer });
        }
      } catch {
        continue;
      }
    }

    if (pageImages.length === 0) {
      return { success: false, error: 'Could not extract any page from PDF' };
    }

    // Process and select the best image
    const outputPath = path.join(BLOG_IMAGE_DIR, `${slug}.jpg`);
    await fs.mkdir(BLOG_IMAGE_DIR, { recursive: true });

    const targetWidth = 1200;
    const targetHeight = 630; // OG image standard

    // Strategy: For academic papers, Figure 1 is typically on page 2
    // Priority: page 2 (top half) > page 3 (top half) > page 1 (bottom half)

    // Try page 2 first - Figure 1 is usually here in 2-column papers
    const page2 = pageImages.find(p => p.page === 2);
    if (page2) {
      const metadata = await sharp(page2.buffer).metadata();
      const { width = 1200, height = 1600 } = metadata;

      // Extract top 55% of page 2 (where Figure 1 typically is)
      const cropHeight = Math.floor(height * 0.55);

      await sharp(page2.buffer)
        .extract({
          left: 0,
          top: 0,
          width: width,
          height: cropHeight
        })
        .resize(targetWidth, targetHeight, {
          fit: 'cover',
          position: 'south'  // Keep bottom of cropped area (figure content)
        })
        .jpeg({ quality: 90 })
        .toFile(outputPath);

      await cleanupTempFiles(tempPdfPath, tempImageBase, pagesToTry);
      return { success: true, imagePath: `/assets/images/blog/${slug}.jpg` };
    }

    // Try page 3 next
    const page3 = pageImages.find(p => p.page === 3);
    if (page3) {
      const metadata = await sharp(page3.buffer).metadata();
      const { width = 1200, height = 1600 } = metadata;

      const cropHeight = Math.floor(height * 0.5);

      await sharp(page3.buffer)
        .extract({
          left: 0,
          top: 0,
          width: width,
          height: cropHeight
        })
        .resize(targetWidth, targetHeight, {
          fit: 'cover',
          position: 'center'
        })
        .jpeg({ quality: 90 })
        .toFile(outputPath);

      await cleanupTempFiles(tempPdfPath, tempImageBase, pagesToTry);
      return { success: true, imagePath: `/assets/images/blog/${slug}.jpg` };
    }

    // Fall back to page 1 bottom half
    const fallbackPage = pageImages[0];
    const metadata = await sharp(fallbackPage.buffer).metadata();
    const { width = 1200, height = 1600 } = metadata;

    const cropTop = Math.floor(height * 0.5);
    const cropHeight = height - cropTop;

    await sharp(fallbackPage.buffer)
      .extract({
        left: 0,
        top: cropTop,
        width: width,
        height: cropHeight
      })
      .resize(targetWidth, targetHeight, {
        fit: 'cover',
        position: 'north'
      })
      .jpeg({ quality: 90 })
      .toFile(outputPath);

    await cleanupTempFiles(tempPdfPath, tempImageBase, pagesToTry);
    return { success: true, imagePath: `/assets/images/blog/${slug}.jpg` };

  } catch (error) {
    await cleanupTempFiles(tempPdfPath, tempImageBase, [1, 2, 3, 4]).catch(() => {});
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error during figure extraction'
    };
  }
}

/**
 * Find generated image file (pdftoppm adds various suffixes)
 */
async function findGeneratedImage(basePath: string, page: number): Promise<string | null> {
  const possibleNames = [
    `${basePath}-${page}.png`,
    `${basePath}-${String(page).padStart(2, '0')}.png`,
    `${basePath}.png`
  ];

  for (const name of possibleNames) {
    try {
      await fs.access(name);
      return name;
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Extract the first figure specifically (more targeted extraction)
 */
export async function extractFirstFigure(
  pdfBuffer: Buffer,
  slug: string
): Promise<ExtractionResult> {
  // For most academic papers, Figure 1 is on page 2 or 3
  return extractFigureFromPdf({
    pdfBuffer,
    slug,
    cropRatio: 0.5
  });
}

/**
 * Cleanup temporary files
 */
async function cleanupTempFiles(
  pdfPath: string,
  imageBase: string,
  pages: number[]
): Promise<void> {
  const filesToDelete = [
    pdfPath,
    ...pages.flatMap(p => [
      `${imageBase}-${p}.png`,
      `${imageBase}-${String(p).padStart(2, '0')}.png`,
      `${imageBase}.png`
    ])
  ];

  await Promise.all(
    filesToDelete.map(f => fs.unlink(f).catch(() => {}))
  );
}
