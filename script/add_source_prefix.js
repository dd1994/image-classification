//我有一个 "D:\image-classification\data\train-tiny" 文件夹，该文件夹下有Amphibia，Insetca, Plantea 等子文件夹表示类别，再下一层是物种 id 作为文件夹，再下一层是物种图片。
// 现在需要统一修改图片的文件名，将 processed_0eb129a3-e613-42e6-80fe-309e90e95562.jpg 这种以`processed_ `格式开头的改成 processed_kanyu_0eb129a3-e613-42e6-80fe-309e90e95562.jpg，
// 其中 `kanyu` 是一个固定值，跟物种类别无关。如果不以`processed_`则不处理。使用 node.js 脚本实现
const fs = require('fs');
const path = require('path');

const rootDir = 'D:\\image-classification\\data\\train-tiny';

function processDirectory(dirPath) {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);

        if (entry.isDirectory()) {
            // 递归处理子目录
            processDirectory(fullPath);
        } else {
            // 处理文件
            const fileName = entry.name;
            const ext = path.extname(fileName).toLowerCase();

            if (fileName.startsWith('processed_') && ext === '.jpg') {
                // 构造新文件名
                const newName = `processed_kanyu_${fileName.substring('processed_'.length)}`;
                const newPath = path.join(dirPath, newName);

                try {
                    fs.renameSync(fullPath, newPath);
                    console.log(`Renamed: ${path.relative(rootDir, fullPath)} -> ${newName}`);
                } catch (err) {
                    console.error(`Error renaming ${path.relative(rootDir, fullPath)}: ${err.message}`);
                }
            }
        }
    }
}

console.log('Starting file renaming...');
processDirectory(rootDir);
console.log('Renaming completed!');