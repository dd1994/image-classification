const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

// 配置路径
const csvFilePath = path.join(__dirname, 'snake_chinesenames.csv');
const imagesDirPath = path.join(__dirname, '..', 'data', 'temp_data', 'snake_images');

// 存储中文到ID的映射
const chineseToIdMap = {};

// 读取CSV文件并构建映射关系
function buildMappingFromCSV() {
  return new Promise((resolve, reject) => {
    fs.createReadStream(csvFilePath)
      .pipe(csv())
      .on('data', (row) => {
        // 使用完整的名称作为键，包括括号部分（如"中国沼蛇[中国水蛇]"）
        const chineseName = row.chineseName.trim();
        const taxonId = row.taxon_id.trim();
        
        if (chineseName && taxonId) {
          chineseToIdMap[chineseName] = taxonId;
        }
      })
      .on('end', () => {
        console.log(`已加载 ${Object.keys(chineseToIdMap).length} 条映射关系`);
        resolve();
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// 重命名目录
function renameDirectories() {
  try {
    const directories = fs.readdirSync(imagesDirPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);
    
    console.log(`找到 ${directories.length} 个目录`);
    
    let renamedCount = 0;
    let notFoundCount = 0;
    
    directories.forEach(dirName => {
      const dirPath = path.join(imagesDirPath, dirName);
      
      // 检查是否有对应的ID映射
      if (chineseToIdMap[dirName]) {
        const newPath = path.join(imagesDirPath, chineseToIdMap[dirName]);
        
        // 检查目标路径是否已存在
        if (!fs.existsSync(newPath)) {
          fs.renameSync(dirPath, newPath);
          console.log(`重命名: ${dirName} -> ${chineseToIdMap[dirName]}`);
          renamedCount++;
        } else {
          console.warn(`目标目录已存在，跳过: ${dirName} -> ${chineseToIdMap[dirName]}`);
        }
      } else {
        console.warn(`未找到映射关系: ${dirName}`);
        notFoundCount++;
      }
    });
    
    console.log(`\n重命名完成: ${renamedCount} 个目录成功重命名, ${notFoundCount} 个目录未找到映射`);
  } catch (error) {
    console.error('重命名目录时出错:', error);
  }
}

// 主函数
async function main() {
  try {
    console.log('开始构建中文到ID的映射关系...');
    await buildMappingFromCSV();
    
    console.log('\n开始重命名目录...');
    renameDirectories();
    
    console.log('\n任务完成!');
  } catch (error) {
    console.error('执行过程中出错:', error);
  }
}

// 执行主函数
main();