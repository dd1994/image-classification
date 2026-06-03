const fs = require("fs");
const path = require("path");

const TRAIN_DIR = path.join(__dirname, "data", "train");

const speciesMap = new Map();

const classDirs = fs.readdirSync(TRAIN_DIR).filter((name) => {
  const stat = fs.statSync(path.join(TRAIN_DIR, name));
  return stat.isDirectory();
});

for (const classDir of classDirs) {
  const classPath = path.join(TRAIN_DIR, classDir);
  const speciesIds = fs.readdirSync(classPath).filter((name) => {
    const stat = fs.statSync(path.join(classPath, name));
    return stat.isDirectory();
  });

  for (const speciesId of speciesIds) {
    if (!speciesMap.has(speciesId)) {
      speciesMap.set(speciesId, []);
    }
    speciesMap.get(speciesId).push(classDir);
  }
}

const duplicates = Array.from(speciesMap.entries())
  .filter(([, dirs]) => dirs.length > 1)
  .map(([id, dirs]) => ({ species_id: id, class_dirs: dirs }));

if (duplicates.length === 0) {
  console.log("No duplicate species IDs found.");
} else {
  console.log(`Found ${duplicates.length} duplicate species ID(s):\n`);
  for (const dup of duplicates) {
    console.log(
      `  species_id: ${dup.species_id} -> [${dup.class_dirs.join(", ")}]`
    );
  }
}
