const fs = require('fs');
const path = require('path');

const trainDir = 'D:/image-classification/data/train';
const targetDir = 'D:/image-classification/data/fgvc-aves-tiny';
const csvFile = 'D:/image-classification/script/fgvc-aves_taxon_tiny.csv';

// Read CSV and parse taxonIDs into a Set
const csvContent = fs.readFileSync(csvFile, 'utf-8');
const lines = csvContent.trim().split('\n');
const taxonIdSet = new Set();

for (let i = 1; i < lines.length; i++) {
  const id = lines[i].trim();
  if (id) taxonIdSet.add(id);
}

console.log(`Loaded ${taxonIdSet.size} taxonIDs from CSV`);

// Iterate through categories and species
const categories = fs.readdirSync(trainDir);
let totalCopied = 0;
let totalImages = 0;

for (const category of categories) {
  const categoryPath = path.join(trainDir, category);
  if (!fs.statSync(categoryPath).isDirectory()) continue;
  if (category.startsWith('.')) continue;

  const speciesDirs = fs.readdirSync(categoryPath);

  for (const speciesId of speciesDirs) {
    if (!taxonIdSet.has(speciesId)) continue;

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
