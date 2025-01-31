const fs = require('fs');
const path = require('path');
const { createObjectCsvWriter } = require('csv-writer');

const rootDir = 'D:/image-classification/data/collection';
const outputFile = 'test.csv'
const result = [];

// 遍历目录结构的同步方法
function processDirectory() {
    // 读取类别目录（Amphibia, Insecta等）
    const classDirs = fs.readdirSync(rootDir, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);

    for (const className of classDirs) {
        const classPath = path.join(rootDir, className);

        // 读取物种ID目录
        const speciesDirs = fs.readdirSync(classPath, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);

        for (const speciesId of speciesDirs) {
            const speciesPath = path.join(classPath, speciesId);

            // 统计图片文件数量
            const files = fs.readdirSync(speciesPath);
            const imageCount = files.length;

            if (imageCount > 40) {
                result.push({
                    class: className,
                    species_id: speciesId,
                    image_count: imageCount
                });
            }
        }
    }
}

// 写入CSV文件
async function writeCSV() {
    const csvWriter = createObjectCsvWriter({
        path: outputFile,
        header: [
            { id: 'class', title: 'CLASS' },
            { id: 'species_id', title: 'SPECIES_ID' },
            { id: 'image_count', title: 'IMAGE_COUNT' }
        ]
    });

    await csvWriter.writeRecords(result);
    console.log('CSV文件已生成');
}

// 执行主流程
try {
    processDirectory();
    writeCSV();
} catch (error) {
    console.error('处理过程中发生错误:', error);
}