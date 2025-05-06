//我有一个 "D:\image-classification\data\collection" 文件夹，该文件夹下有Amphibia，Insetca, Plantea 等子文件夹表示类别，
//再下一层是物种 id 作为文件夹，再下一层是物种图片。
//现在写一段 node.js 脚本，统计出所有物种图片大于 50 的物种，
//并将对应的物种文件夹移动到 "D:\image-classification\data\train" 对应的文件夹下，并保持文件夹结构不变

const fs = require('fs');
const path = require('path');

// 配置路径
const sourceRoot = 'D:/image-classification/data/collection';
const targetRoot = 'D:/image-classification/data/train-mini';
const minImageCount = 49;

// 统计结果
const result = {
    totalProcessed: 0,
    moved: 0,
    failed: []
};

// 主处理函数
function processDirectories() {
    // 读取分类目录
    const classDirs = fs.readdirSync(sourceRoot, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);

    for (const className of classDirs) {
        if(!['Insecta'].includes(className)) {
            continue
        }
        const classPath = path.join(sourceRoot, className);

        // 读取物种目录
        const speciesDirs = fs.readdirSync(classPath, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);

        for (const speciesId of speciesDirs) {
            result.totalProcessed++;
            const speciesPath = path.join(classPath, speciesId);

            try {
                // 统计图片数量
                const files = fs.readdirSync(speciesPath)
                    .filter(file => fs.statSync(path.join(speciesPath, file)).isFile());

                if (files.length > minImageCount) {
                    // 构建目标路径
                    const targetPath = path.join(targetRoot, className, speciesId);

                    // 创建目标分类目录（仅在不存在时创建）
                    const targetClassDir = path.join(targetRoot, className);
                    if (!fs.existsSync(targetClassDir)) {
                        fs.mkdirSync(targetClassDir, {
                            recursive: true,
                            mode: 0o755
                        });
                    }

                    // 移动目录
                    fs.renameSync(speciesPath, targetPath);
                    console.log(`已移动：${className}/${speciesId} (${files.length}张图片)`);
                    result.moved++;
                }
            } catch (error) {
                result.failed.push({
                    path: `${className}/${speciesId}`,
                    error: error.message
                });
                console.error(`处理失败：${className}/${speciesId}`, error.message);
            }
        }
    }
}

// 执行并输出结果
try {
    processDirectories();
    console.log('\n处理结果：');
    console.log(`总处理物种数：${result.totalProcessed}`);
    console.log(`成功移动数：${result.moved}`);
    console.log(`失败数：${result.failed.length}`);

    if (result.failed.length > 0) {
        console.log('\n失败详情：');
        result.failed.forEach(item => {
            console.log(`[${item.path}] ${item.error}`);
        });
    }
} catch (error) {
    console.error('程序异常终止：', error);
}