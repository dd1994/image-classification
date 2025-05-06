const fs = require('fs');
const path = require('path');

const GENUS0_PATH = 'D:/image-classification/data/spiderGenus0';
const GENUS1_PATH = 'D:/image-classification/data/spiderGenus1';
const MAX_IMAGES = 1500;

function moveSpeciesImages() {
  // 遍历genus1的每个分类目录
  fs.readdirSync(GENUS1_PATH, { withFileTypes: true }).forEach(categoryDir => {
    if (!categoryDir.isDirectory()) return;

    const categoryPath = path.join(GENUS1_PATH, categoryDir.name);
    
    // 遍历每个物种目录
    fs.readdirSync(categoryPath, { withFileTypes: true }).forEach(speciesDir => {
      if (!speciesDir.isDirectory()) return;

      const sourcePath = path.join(categoryPath, speciesDir.name);
      const destPath = path.join(GENUS0_PATH, categoryDir.name, speciesDir.name);
      
      try {
        // 获取源目录文件列表
        const sourceFiles = fs.readdirSync(sourcePath)
          .filter(file => fs.statSync(path.join(sourcePath, file)).isFile());

        // 创建目标目录（如果不存在）
        if (!fs.existsSync(destPath)) {
          fs.mkdirSync(destPath, { recursive: true });
        }

        // 计算目标目录现有文件数
        const existingFiles = fs.readdirSync(destPath)
          .filter(file => fs.statSync(path.join(destPath, file)).isFile());
        
        // 检查数量限制
        if (existingFiles.length >= MAX_IMAGES) {
          console.log(`Skipped ${speciesDir.name}: Already has ${MAX_IMAGES} files`);
          return;
        }

        const availableSpace = MAX_IMAGES - existingFiles.length;
        const filesToMove = sourceFiles.slice(0, availableSpace);

        if (filesToMove.length === 0) {
          console.log(`Skipped ${speciesDir.name}: No space available`);
          return;
        }

        // 移动文件
        filesToMove.forEach(file => {
          const sourceFile = path.join(sourcePath, file);
          const destFile = path.join(destPath, file);
          
          try {
            fs.renameSync(sourceFile, destFile);
            console.log(`Moved: ${file}`);
          } catch (err) {
            if (err.code === 'EXDEV') { // 跨设备处理
              fs.copyFileSync(sourceFile, destFile);
              fs.unlinkSync(sourceFile);
              console.log(`Copied: ${file} (cross-device)`);
            } else {
              throw err;
            }
          }
        });

        console.log(`Moved ${filesToMove.length} files for ${speciesDir.name}`);

        // 尝试删除空源目录
        if (fs.readdirSync(sourcePath).length === 0) {
          fs.rmdirSync(sourcePath);
        }
      } catch (err) {
        console.error(`Error processing ${speciesDir.name}: ${err.message}`);
      }
    });
  });
}

moveSpeciesImages();
console.log('Merge completed');