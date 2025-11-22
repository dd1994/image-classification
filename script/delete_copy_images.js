const fs = require('fs');
const path = require('path');
const { imgExt } = require('./const');

// 要扫描的根目录，可以根据实际情况修改
const rootDir = path.resolve(__dirname, '../data', 'train-pre');

// 递归遍历目录，查找并删除以 "copy" 开头的图片文件
function deleteCopyImages(dir) {
  try {
    const files = fs.readdirSync(dir);
    
    files.forEach(file => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        // 如果是目录，递归处理
        deleteCopyImages(filePath);
      } else {
        // 检查是否是图片文件且以 "copy" 开头
        const ext = path.extname(file).toLowerCase();
        const baseName = path.basename(file, ext);
        
        if (imgExt.includes(ext) && baseName.startsWith('copy')) {
          try {
            fs.unlinkSync(filePath);
            console.log(`已删除: ${filePath}`);
          } catch (err) {
            console.error(`删除文件失败: ${filePath}`, err);
          }
        }
      }
    });
  } catch (err) {
    console.error(`读取目录失败: ${dir}`, err);
  }
}

// 检查根目录是否存在
if (fs.existsSync(rootDir)) {
  console.log(`开始扫描目录: ${rootDir}`);
  deleteCopyImages(rootDir);
  console.log('删除完成');
} else {
  console.error(`目录不存在: ${rootDir}`);
  console.log('请检查脚本中的 rootDir 路径是否正确');
}