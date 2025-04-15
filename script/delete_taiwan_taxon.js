const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

// 删除指定目录，主要用于删除台湾特有种

// 配置部分
const parentDir = 'D:/image-classification/data/collection'; // 父文件夹路径
const csvFilePath = './inat_blacklist.csv'; // CSV文件路径

// 安全校验函数
const validatePaths = () => {
  if (!fs.existsSync(parentDir)) {
    throw new Error(`父文件夹不存在: ${parentDir}`);
  }
  if (!fs.statSync(parentDir).isDirectory()) {
    throw new Error(`父文件夹路径不是目录: ${parentDir}`);
  }
};

// 使用流式读取CSV文件
const readTaxonIdsFromCSV = () => {
  return new Promise((resolve, reject) => {
    const results = [];

    fs.createReadStream(csvFilePath)
      .pipe(csv())
      .on('data', (row) => {
        const taxonId = row.taxon_id;
        if (taxonId) results.push(String(taxonId));
      })
      .on('end', () => {
        console.log(`从CSV读取到 ${results.length} 个需要处理的taxonId`);
        resolve(results);
      })
      .on('error', (error) => {
        console.error('CSV文件读取失败:');
        reject(error);
      });
  });
};

// 增强版删除逻辑
const deleteFoldersInSubdirectories = async (foldersToDelete) => {
  try {
    let totalDeleteCount = 0;

    // 获取 train 文件夹下的所有子文件夹
    const subdirectories = fs.readdirSync(parentDir).filter(item => {
      const itemPath = path.join(parentDir, item);
      return fs.statSync(itemPath).isDirectory();
    });

    for (const subdir of subdirectories) {
      const subdirPath = path.join(parentDir, subdir);
      let deleteCount = 0;

      const items = fs.readdirSync(subdirPath);
      for (const item of items) {
        const itemPath = path.join(subdirPath, item);
        const stats = fs.lstatSync(itemPath);

        if (stats.isDirectory() && foldersToDelete.includes(item)) {
          console.log(`[${++deleteCount}] 正在删除: ${itemPath}`);

          await fs.promises.rm(itemPath, {
            recursive: true,
            force: true,
            retryDelay: 100,
            maxRetries: 3
          });

          console.log(`✅ 成功删除: ${item}`);
        }
      }

      totalDeleteCount += deleteCount;
      console.log(`\n在 ${subdir} 中删除完成，共删除 ${deleteCount} 个目录`);
    }

    console.log(`\n总删除完成，共删除 ${totalDeleteCount} 个目录`);

  } catch (error) {
    console.error('删除过程中发生错误:');
    throw error;
  }
};

// 主流程
(async () => {
  try {
    validatePaths();

    console.log('正在读取CSV文件...');
    const foldersToDelete = await readTaxonIdsFromCSV();

    console.log('\n===== 操作确认 =====');
    console.log('待删除目录数量:', foldersToDelete.length);
    console.log('示例条目:', foldersToDelete.slice(0, 5));

    // 添加人工确认步骤
    const readline = require('readline').createInterface({
      input: process.stdin,
      output: process.stdout
    });

    await new Promise(resolve => {
      readline.question('\n确认要执行删除操作？(y/n) ', async (answer) => {
        if (answer.toLowerCase() === 'y') {
          console.log('\n开始删除操作...');
          await deleteFoldersInSubdirectories(foldersToDelete);
        } else {
          console.log('操作已取消');
          process.exit(0);
        }
        readline.close();
        resolve();
      });
    });

  } catch (error) {
    console.error('\n❌ 程序异常终止:');
    console.error(error.stack || error.message);
    process.exit(1);
  }
})();



