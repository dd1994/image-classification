const fs = require('fs');
const path = require('path');
const { imgExt } = require('./const')
const trainDir = 'D:/image-classification/data/train-mini';
const collectionBase = path.join(path.dirname(trainDir), 'collection'); // 自动生成collection路径\
const ACTIVE = 'Aves'
const outputFile = `${ACTIVE}_stat.csv`
const minCount = 50
// 支持的图片扩展名集合
const IMAGE_EXTENSIONS = new Set(imgExt);

// 同步删除目录
const deleteDir = (dirPath) => {
  if (fs.existsSync(dirPath)) {
    fs.rmSync(dirPath, { recursive: true, force: true });
  }
};

// 移动目录的同步方法
const moveDirectory = (srcPath, destPath) => {
  try {
    if (!fs.existsSync(srcPath)) return;

    // 如果目标目录不存在，直接移动整个目录
    if (!fs.existsSync(destPath)) {
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      fs.renameSync(srcPath, destPath);
      console.log(`移动目录成功: ${srcPath} -> ${destPath}`);
      return;
    }

    // 目标目录存在时，合并文件
    const files = fs.readdirSync(srcPath);
    let movedCount = 0;

    files.forEach(file => {
      const srcFile = path.join(srcPath, file);
      const destFile = path.join(destPath, file);

      // 跳过已存在的文件
      if (fs.existsSync(destFile)) {
        console.log(`跳过已存在文件: ${destFile}`);
        return;
      }

      // 移动文件并保持目录结构
      fs.renameSync(srcFile, destFile);
      movedCount++;
    });

    console.log(`合并完成: 从 ${srcPath} 移动了 ${movedCount} 个文件到 ${destPath}`);

    // 删除已搬空的源目录
    if (fs.readdirSync(srcPath).length === 0) {
      fs.rmdirSync(srcPath);
      console.log(`删除空目录: ${srcPath}`);
    }
  } catch (err) {
    console.error(`操作失败: ${srcPath}`, err);
  }
};

try {
  const results = [];

  // 遍历第一层分类目录（如 Mollusca/Insecta）
  const classDirs = fs.readdirSync(trainDir);

  for (const className of classDirs) {
//    if(className !== ACTIVE) {
//        continue
//    }
    const classPath = path.join(trainDir, className);
    const classStats = fs.statSync(classPath);

    // 跳过隐藏目录和文件
    if (className.startsWith('.')) {
      deleteDir(classPath);
      console.log(`删除隐藏分类目录: ${classPath}`);
      continue;
    }

    if (!classStats.isDirectory()) {
      fs.unlinkSync(classPath);
      console.log(`删除非目录文件: ${classPath}`);
      continue;
    }

    // 遍历第二层物种目录（taxon_id）
    const taxonDirs = fs.readdirSync(classPath);

    for (const taxonId of taxonDirs) {
      const taxonPath = path.join(classPath, taxonId);
      const taxonStats = fs.statSync(taxonPath);

      // 处理隐藏目录和非目录文件
      if (taxonId.startsWith('.')) {
        deleteDir(taxonPath);
        console.log(`删除隐藏物种目录: ${taxonPath}`);
        continue;
      }

      if (!taxonStats.isDirectory()) {
        fs.unlinkSync(taxonPath);
        console.log(`删除非目录文件: ${taxonPath}`);
        continue;
      }

      // 处理物种目录内容
      const files = fs.readdirSync(taxonPath);
      const validFiles = [];

      for (const file of files) {
          const filePath = path.join(taxonPath, file);
          const fileStats = fs.statSync(filePath);

          if (fileStats.isDirectory()) {
              deleteDir(filePath);
              console.log(`删除嵌套目录: ${filePath}`);
              continue;
          }

          const ext = path.extname(file).toLowerCase();
          if (file.startsWith('.') || !IMAGE_EXTENSIONS.has(ext)) {
              fs.unlinkSync(filePath);
              console.log(`删除无效文件: ${filePath}`);
              continue;
          }

          // 记录有效文件及其修改时间
          validFiles.push({
              path: filePath,
              mtime: fileStats.mtime.getTime()
          });
      }

      // 根据数量决定保留或移动
     if(validFiles.length < minCount) {
          const destPath = path.join(collectionBase, className, taxonId);
          console.log(`图片数量 ${validFiles.length}，移动到 ${destPath}`);
          moveDirectory(taxonPath, destPath);
      } else {
        results.push({
          class: className,
          taxon_id: taxonId,
          photo_count: validFiles.length
      });
      }
    }
  }

  // 生成CSV报告
  const csvContent = [
    'class,taxon_id,photo_count',
    ...results.map(r => `${r.class},${r.taxon_id},${r.photo_count}`)
  ].join('\n');

  fs.writeFileSync(outputFile, csvContent);
  console.log(`生成统计文件成功，共 ${results.length} 个有效物种`);

} catch (err) {
  console.error('处理过程中发生错误:', err);
  process.exit(1);
}