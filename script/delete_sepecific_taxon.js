const fs = require('fs');
const path = require('path');

// 配置部分
const parentDir = 'D:/image-classification/data/train/Reptilia'; // 父文件夹路径
const foldersToDelete = [
"35484",
"35492",
"35493",
"35500",
"37668",
"517749",
"539787",
"539801",
"540025",
"797622",
"797632",
"797638"
]; // 需要删除的文件夹名数组

// 安全校验函数
const validatePaths = () => {
  if (!fs.existsSync(parentDir)) {
    throw new Error(`父文件夹不存在: ${parentDir}`);
  }
  if (!fs.statSync(parentDir).isDirectory()) {
    throw new Error(`父文件夹路径不是目录: ${parentDir}`);
  }
};

// 主删除逻辑
const deleteFolders = () => {
  try {
    // 读取父文件夹内容
    const items = fs.readdirSync(parentDir);

    items.forEach(item => {
      const itemPath = path.join(parentDir, item);
      const stats = fs.lstatSync(itemPath);

      // 只处理目录且名称匹配的项
      if (stats.isDirectory() && foldersToDelete.includes(item)) {
        console.log(`正在删除: ${itemPath}`);

        // 递归强制删除目录
        fs.rmSync(itemPath, {
          recursive: true,
          force: true,
          retryDelay: 100, // 重试间隔 100ms
          maxRetries: 3    // 最多重试 3 次
        });

        console.log(`成功删除: ${item}`);
      }
    });

    console.log('删除操作完成，请检查剩余目录确认结果');

  } catch (error) {
    console.error('删除过程中发生错误:');
    console.error(error.stack);
    process.exit(1);
  }
};

// 执行流程
try {
  validatePaths();
  console.log('开始删除操作...\n');
  deleteFolders();
} catch (error) {
  console.error('初始化校验失败:');
  console.error(error.message);
  process.exit(1);
}