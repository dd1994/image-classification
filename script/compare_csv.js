const fs = require('fs');
const path = require('path');

// 获取命令行参数
const [file1Path, file2Path] = process.argv.slice(2);

// 检查是否提供了两个文件路径
if (!file1Path || !file2Path) {
  console.error('请提供两个 CSV 文件路径作为参数。');
  process.exit(1);
}

// 读取文件内容
function readFile(filePath) {
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
}

// 比较两个文件内容
async function compareFiles(file1, file2) {
  try {
    const data1 = await readFile(file1);
    const data2 = await readFile(file2);

    if (data1 === data2) {
      console.log('两个 CSV 文件完全相同。');
    } else {
      console.log('两个 CSV 文件不相同。');
    }
  } catch (error) {
    console.error(`读取文件时出错: ${error.message}`);
  }
}

// 执行比较
compareFiles(file1Path, file2Path);