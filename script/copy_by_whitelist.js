const fs = require('fs');
const path = require('path');

// ====== Config ======
const sourceDir = 'D:/image-classification/data/train';
const targetDir = 'D:/image-classification/data/fgvc-plantae-tiny';
const whitelistCsv = 'D:/image-classification/script/fgvc-plantae_taxon_tiny.csv';
// ====================

// Read CSV and parse IDs into a Set (skip header line)
const csvContent = fs.readFileSync(whitelistCsv, 'utf-8');
const lines = csvContent.trim().split('\n');
const whitelistIds = new Set();

for (let i = 1; i < lines.length; i++) {
  const id = lines[i].trim();
  if (id) whitelistIds.add(id);
}

console.log(`Loaded ${whitelistIds.size} IDs from whitelist`);

// Iterate through categories and species
const categories = fs.readdirSync(sourceDir);
let totalCopied = 0;
let totalImages = 0;

for (const category of categories) {
  const categoryPath = path.join(sourceDir, category);
  if (!fs.statSync(categoryPath).isDirectory()) continue;
  if (category.startsWith('.')) continue;

  const speciesDirs = fs.readdirSync(categoryPath);

  for (const speciesId of speciesDirs) {
    if (!whitelistIds.has(speciesId)) continue;

    const speciesPath = path.join(categoryPath, speciesId);
    if (!fs.statSync(speciesPath).isDirectory()) continue;

    const destCategory = path.join(targetDir, category);
    const destSpecies = path.join(destCategory, speciesId);

    if (!fs.existsSync(destCategory)) {
      fs.mkdirSync(destCategory, { recursive: true });
    }

    console.log(`Copying ${category}/${speciesId} ...`);
    fs.cpSync(speciesPath, destSpecies, { recursive: true });

    const imageCount = fs.readdirSync(speciesPath).length;
    totalCopied++;
    totalImages += imageCount;
  }
}

console.log(`Done. Copied ${totalCopied} species, ${totalImages} images to ${targetDir}`);
