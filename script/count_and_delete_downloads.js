const fs = require('fs');
const path = require('path');

const imgDir = 'D:\\image-classification\\data\\large\\Reptilia';
const ouputFile = 'trainReptilia50.csv'

// 读取 plant 文件夹下的所有子文件夹
fs.readdir(imgDir, (err, subdirs) => {
  if (err) {
    console.error('无法读取文件夹:', err);
    return;
  }

  // 初始化结果数组
  const results = [];

  // 遍历每个子文件夹
  subdirs.forEach(subdir => {
    const subdirPath = path.join(imgDir, subdir);

    // 检查是否为文件夹
    if (fs.statSync(subdirPath).isDirectory()) {
      // 提取 taxon id 和 taxon name
      const taxonId = subdir;

      // 读取子文件夹中的所有文件
      const files = fs.readdirSync(subdirPath);

      // 过滤出图片文件（假设图片扩展名为 .jpg 或 .png）
    //   const imageFiles = files.filter(file => file.endsWith('.jpg') || file.endsWith('.png'));

      // 计算图片数量
      const photoCount = files.length;

      if(photoCount > 50) {
        // 将结果添加到数组中
        results.push({ taxonId, photoCount: photoCount });
      } else {
        // 删除包含少于 50 张照片的文件夹
         fs.rmdir(subdirPath, { recursive: true }, err => {
           if (err) {
             console.error(`无法删除文件夹 ${subdirPath}:`, err);
           } else {
             console.log(`文件夹 ${subdirPath} 已成功删除`);
           }
         });
      }
    }
  });

  // 将结果写入 CSV 文件
  const csvContent = [
    'taxon_id,photo_count',
    ...results.map(result => `${result.taxonId},${result.photoCount}`)
  ].join('\n');

  fs.writeFile(path.join(__dirname, ouputFile), csvContent, err => {
    if (err) {
      console.error('无法写入 CSV 文件:', err);
    } else {
      console.log('CSV 文件已成功创建');
    }
  });
});



