const fs = require('fs');
const path = require('path');

const imgDir = 'D:/image-classification/data/train/Mollusca';
const ouputFile = 'trainMollusca.csv';

// 支持的图片扩展名列表
const IMAGE_EXTENSIONS = new Set([
'.jpg', '.jpeg', 'png', 'webp', '.heic', '.heif', '.bmp'
]);

// 同步删除目录（兼容Node 12+）
const deleteDir = (dirPath) => {
  if (fs.existsSync(dirPath)) {
    fs.rmSync(dirPath, { recursive: true, force: true });
  }
};

try {
  const results = [];
  const subdirs = fs.readdirSync(imgDir);

  for (const subdir of subdirs) {
    const subdirPath = path.join(imgDir, subdir);
    const stats = fs.statSync(subdirPath);

    // 处理隐藏目录和文件
    if (subdir.startsWith('.')) {
      deleteDir(subdirPath);
      console.log(`Deleted hidden: ${subdirPath}`);
      continue;
    }

    // 跳过非目录文件
    if (!stats.isDirectory()) {
      fs.unlinkSync(subdirPath);
      console.log(`Deleted file: ${subdirPath}`);
      continue;
    }

    // 处理子目录内容
    let fileCount = 0;
    const files = fs.readdirSync(subdirPath);

    for (const file of files) {
      const filePath = path.join(subdirPath, file);
      const fileStats = fs.statSync(filePath);

      // 删除子目录中的嵌套目录
      if (fileStats.isDirectory()) {
        deleteDir(filePath);
        console.log(`Deleted nested directory: ${filePath}`);
        continue;
      }

      // 处理隐藏文件和非图片文件
      if (file.startsWith('.') ||
          !IMAGE_EXTENSIONS.has(path.extname(file).toLowerCase())) {
        fs.unlinkSync(filePath);
        console.log(`Deleted invalid file: ${filePath}`);
        continue;
      }

      fileCount++;
    }

    // 最终数量检查
    if (fileCount > 49) {
      results.push({ taxonId: subdir, photoCount: fileCount });
    } else {
      deleteDir(subdirPath);
      console.log(`Deleted insufficient directory: ${subdirPath}`);
    }
  }

  // 生成CSV内容
  const csvContent = [
    'taxon_id,photo_count',
    ...results.map(({ taxonId, photoCount }) => `${taxonId},${photoCount}`)
  ].join('\n');

  fs.writeFileSync(path.join(__dirname, ouputFile), csvContent);
  console.log(`CSV文件已成功创建，包含 ${results.length} 个有效分类`);

} catch (err) {
  console.error('处理过程中发生错误:', err);
}