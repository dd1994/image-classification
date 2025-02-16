const fs = require('fs');
const path = require('path');
const { imgExt } = require('./const')
const trainDir = 'D:/image-classification/data/train-small';
const outputFile = 'reptilia_stat.csv';
const collectionBase = path.join(path.dirname(trainDir), 'collection'); // 自动生成collection路径
console.log(imgExt)
// 支持的图片扩展名集合
const IMAGE_EXTENSIONS = new Set(imgExt);

// 同步删除目录
const deleteDir = (dirPath) => {
  if (fs.existsSync(dirPath)) {
    fs.rmSync(dirPath, { recursive: true, force: true });
  }
};

try {
  const results = [];

  // 遍历第一层分类目录（如 Mollusca/Insecta）
  const classDirs = fs.readdirSync(trainDir);

  for (const className of classDirs) {
    if(className !== 'Reptilia') {
        continue
    }
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

        results.push({
          class: className,
          taxon_id: taxonId,
          photo_count: validFiles.length
      });
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