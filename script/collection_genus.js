//我有一个 "D:\image-classification\data\collection2" 文件夹，该文件夹下有Amphibia，Insetca, Plantea 等子文件夹表示类别，
//再下一层是物种 id 作为文件夹（这里的物种id都是到种(species)级别的唯一数字 id，没有到属(genus)级别的），再下一层是物种图片，且图片文件名是全局唯一的。
// 请写一段 node.js 脚本，实现如下功能:
// 将同一个属下种级别的图片合并到一个以属 id 命名的文件夹中，属文件夹下是该属的所有图片。并删除原来的种文件夹，Amphibia，Insetca, Plantea 等父文件夹保持不变，
// 其中种和属的关系由 species_with_genus.csv 文件指定，这个文件由一个 species 字段表示种id, genus 字段表示属 id

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

// 配置路径
const BASE_DIR = 'D:\\image-classification\\data\\collection2';
const CSV_PATH = path.join(__dirname, 'species_with_genus.csv');

// 存储种属映射关系
const speciesGenusMap = new Map();

// 1. 读取CSV文件建立映射关系
fs.createReadStream(CSV_PATH)
  .pipe(csv())
  .on('data', (row) => {
    speciesGenusMap.set(row.species, row.genus);
  })
  .on('end', () => {
    processCategories();
  })
  .on('error', (err) => {
    console.error('CSV读取失败:', err);
  });

// 2. 处理每个分类目录
function processCategories() {
  fs.readdirSync(BASE_DIR, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .forEach(({ name: category }) => {
      const categoryPath = path.join(BASE_DIR, category);

      fs.readdirSync(categoryPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .forEach(({ name: speciesID }) => {
          const genusID = speciesGenusMap.get(speciesID);

          if (!genusID) {
            console.warn(`跳过未找到属的物种: ${category}/${speciesID}`);
            return;
          }

          processSpecies(
            path.join(categoryPath, speciesID),
            path.join(categoryPath, genusID.toString())
          );
        });
    });
}

// 3. 处理单个物种目录
function processSpecies(speciesPath, genusPath) {
  try {
    // 创建属目录
    if (!fs.existsSync(genusPath)) {
      fs.mkdirSync(genusPath);
    }

    // 移动文件（直接使用原始文件名）
    fs.readdirSync(speciesPath).forEach(file => {
      const src = path.join(speciesPath, file);
      const dest = path.join(genusPath, file);

      if (fs.statSync(src).isFile()) {
        fs.renameSync(src, dest);
      }
    });

    // 删除空目录
    fs.rmdirSync(speciesPath);
    console.log(`成功处理物种目录: ${speciesPath}`);
  } catch (err) {
    console.error(`处理失败: ${speciesPath}`, err);
  }
}