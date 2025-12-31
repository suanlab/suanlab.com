import fs from 'fs/promises';
import path from 'path';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

async function generateFavicon(): Promise<void> {
  if (!GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY is not set');
  }

  const prompt = `Generate an image: Create a modern, minimalist app icon/favicon for "SuanLab" - a Data Science and AI Research Lab.

Design requirements:
- Perfect square format (1:1 aspect ratio), 512x512 pixels
- Simple, recognizable symbol that works at small sizes (16x16 to 512x512)
- Modern tech aesthetic with clean geometric design
- Color scheme: Deep blue (#1e40af) to teal (#0891b2) gradient or solid
- Possible concepts: stylized "S" letter, neural network node, data/brain fusion symbol, abstract geometric shape representing AI/data
- NO text, NO letters - pure symbol/icon only
- Clean edges, no fine details that disappear at small sizes
- Professional, minimal, memorable
- Should look good on both light and dark backgrounds`;

  console.log('Generating favicon symbol with Gemini...');

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          responseModalities: ['IMAGE', 'TEXT']
        }
      })
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();

  // Extract image from response
  const parts = data.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData?.data) {
      const imageBuffer = Buffer.from(part.inlineData.data, 'base64');

      // Save as apple-touch-icon (will be used as base for other sizes)
      const outputPath = path.join(process.cwd(), 'public/apple-touch-icon.png');
      await fs.writeFile(outputPath, imageBuffer);

      console.log(`Icon saved to: ${outputPath}`);
      console.log(`File size: ${(imageBuffer.length / 1024).toFixed(2)} KB`);

      // Also save a copy for favicon generation
      const iconPath = path.join(process.cwd(), 'public/icon-512.png');
      await fs.writeFile(iconPath, imageBuffer);
      console.log(`Also saved to: ${iconPath}`);

      return;
    }
  }

  throw new Error('No image data in response');
}

generateFavicon()
  .then(() => {
    console.log('Done! Next steps:');
    console.log('1. Use an online tool like https://realfavicongenerator.net/ to generate all favicon sizes');
    console.log('2. Or install sharp: npm install sharp');
    console.log('3. The generated icon is saved as apple-touch-icon.png and icon-512.png');
    process.exit(0);
  })
  .catch((err) => {
    console.error('Error:', err);
    process.exit(1);
  });
