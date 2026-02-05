const fs = require('fs');
const path = require('path');

const trainDir = 'D:/image-classification/data/train-pre';
const outputFile = 'species_image_mapping.csv';

console.log('开始生成物种ID和图片ID映射文件...');

try {
  const results = [];

  // 遍历第一层分类目录（如 Reptilia/Amphibians/Plantae/Insecta 等）
  const classDirs = fs.readdirSync(trainDir);

  for (const className of classDirs) {
    const classPath = path.join(trainDir, className);
    const classStats = fs.statSync(classPath);

    // 跳过隐藏目录和文件
    if (className.startsWith('.') || !classStats.isDirectory()) {
      continue;
    }

    console.log(`处理分类: ${className}`);

    // 遍历第二层物种目录（taxon_id）
    const taxonDirs = fs.readdirSync(classPath);

    for (const taxonId of taxonDirs) {
      const taxonPath = path.join(classPath, taxonId);
      const taxonStats = fs.statSync(taxonPath);

      // 处理隐藏目录和非目录文件
      if (taxonId.startsWith('.') || !taxonStats.isDirectory()) {
        continue;
      }

      // 处理物种目录中的文件
      const files = fs.readdirSync(taxonPath);

      for (const file of files) {
        // 只处理 processed_inaturalist 开头格式的图片
        if (file.startsWith('processed_inaturalist_') && file.endsWith('.jpg')) {
          // 提取图片ID：去掉 processed_inaturalist_ 前缀和 .jpg 后缀
          const imageId = file.substring('processed_inaturalist_'.length, file.length - 4);
          
          // 验证图片ID是否为纯数字
          if (/^\d+$/.test(imageId)) {
            results.push({
              species_id: taxonId,
              image_id: imageId
            });
          }
        }
      }
    }
  }

  // 生成CSV文件
  const csvContent = [
    'species_id,image_id',
    ...results.map(r => `${r.species_id},${r.image_id}`)
  ].join('\n');

  fs.writeFileSync(outputFile, csvContent);
  
  console.log(`✅ 生成完成！`);
  console.log(`📊 统计信息:`);
  console.log(`   - 总共处理了 ${results.length} 张符合条件的图片`);
  console.log(`   - 涉及 ${new Set(results.map(r => r.species_id)).size} 个物种`);
  console.log(`   - 输出文件: ${outputFile}`);

} catch (err) {
  console.error('❌ 处理过程中发生错误:', err);
  process.exit(1);
}