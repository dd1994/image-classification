const fs = require('fs').promises;
const path = require('path');

// 图片文件扩展名
const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

const otherTypesDir = "D:/image-classification/data/other-types";
const trainDir = "D:/image-classification/data/train";
const trackingFile = "D:/image-classification/data/other_types_files.json";

// 映射规则：other-types 文件夹前缀 → train 类名
// AvesEgg/AvesFeather/AvesFemale/AvesSmall → Aves
// InsectaEgg/InsectaLarva/InsectaNymph/InsectaPupa → Insecta
const MAPPING = [
    { prefix: 'Aves', target: 'Aves' },
    { prefix: 'Insecta', target: 'Insecta' },
];

/**
 * 根据 other-types 文件夹名获取目标 train 类名
 */
function getTargetClass(folderName) {
    for (const { prefix, target } of MAPPING) {
        if (folderName.startsWith(prefix)) {
            return target;
        }
    }
    return null;
}

/**
 * 判断是否为图片文件
 */
function isImageFile(filename) {
    return imageExtensions.includes(path.extname(filename).toLowerCase());
}

/**
 * 获取目录中的所有图片文件（忽略隐藏文件）
 */
async function getImageFiles(dir) {
    try {
        const files = await fs.readdir(dir);
        return files.filter(f => isImageFile(f) && !f.startsWith('.'));
    } catch (err) {
        return [];
    }
}

/**
 * 创建目录（递归）
 */
async function ensureDir(dir) {
    try {
        await fs.mkdir(dir, { recursive: true });
    } catch (err) {
        if (err.code !== 'EEXIST') throw err;
    }
}

/**
 * 加载已存在的追踪文件（用于幂等性）
 * 返回一个 Set<string>，元素为绝对路径
 */
async function loadTrackingFile() {
    try {
        const data = await fs.readFile(trackingFile, 'utf-8');
        const arr = JSON.parse(data);
        console.log(`Loaded ${arr.length} existing entries from tracking file.`);
        return new Set(arr);
    } catch (err) {
        if (err.code === 'ENOENT') {
            console.log('No existing tracking file, starting fresh.');
        } else {
            console.error(`Warning: could not read tracking file: ${err.message}`);
        }
        return new Set();
    }
}

/**
 * 主函数
 */
async function main() {
    // 加载已存在的追踪记录
    const trackingSet = await loadTrackingFile();

    // 获取 other-types 下所有文件夹
    const typeFolders = (await fs.readdir(otherTypesDir))
        .filter(name => !name.startsWith('.'));

    console.log(`Found ${typeFolders.length} other-type folders: ${typeFolders.join(', ')}`);
    console.log('');

    let totalCopied = 0;
    let totalSkipped = 0;
    let totalErrors = 0;

    // 汇总统计
    const stats = {};  // { targetClass: { typeName: copiedCount } }

    for (const typeFolder of typeFolders) {
        const targetClass = getTargetClass(typeFolder);
        if (!targetClass) {
            console.log(`SKIP: "${typeFolder}" — no mapping rule found.`);
            continue;
        }

        const sourceTypeDir = path.join(otherTypesDir, typeFolder);
        const targetClassDir = path.join(trainDir, targetClass);

        console.log(`Processing: ${typeFolder} → ${targetClass}`);

        // 获取该形态下的所有物种文件夹
        const speciesIds = (await fs.readdir(sourceTypeDir))
            .filter(name => !name.startsWith('.'));

        let typeCopied = 0;
        let typeSkipped = 0;
        let typeErrors = 0;

        for (const speciesId of speciesIds) {
            const sourceSpeciesDir = path.join(sourceTypeDir, speciesId);
            const targetSpeciesDir = path.join(targetClassDir, speciesId);

            // 确保目标物种文件夹存在
            await ensureDir(targetSpeciesDir);

            // 获取源文件夹中的图片
            const imageFiles = await getImageFiles(sourceSpeciesDir);

            for (const filename of imageFiles) {
                const srcPath = path.join(sourceSpeciesDir, filename);
                const destPath = path.join(targetSpeciesDir, filename);

                // 检查目标文件是否已存在于追踪文件中（幂等）
                if (trackingSet.has(destPath)) {
                    typeSkipped++;
                    continue;
                }

                // 检查目标文件是否已存在（文件名冲突 → 跳过复制，但要记入追踪文件以保护 train 中已有文件）
                try {
                    await fs.stat(destPath);
                    // 文件已存在（原始 train 中就有的同名文件），跳过复制但加入追踪
                    trackingSet.add(destPath);
                    typeSkipped++;
                    continue;
                } catch {
                    // 文件不存在，可以复制
                }

                // 复制文件
                try {
                    await fs.copyFile(srcPath, destPath);
                    trackingSet.add(destPath);
                    totalCopied++;
                    typeCopied++;
                } catch (err) {
                    console.error(`  Error copying ${srcPath} → ${destPath}: ${err.message}`);
                    totalErrors++;
                    typeErrors++;
                }
            }
        }

        // 该形态文件夹的统计
        if (!stats[targetClass]) stats[targetClass] = {};
        stats[targetClass][typeFolder] = {
            copied: typeCopied,
            skipped: typeSkipped,
            errors: typeErrors,
        };

        console.log(`  Copied: ${typeCopied}, Skipped: ${typeSkipped}, Errors: ${typeErrors}`);
        console.log('');
    }

    // 保存追踪文件
    const trackingArray = Array.from(trackingSet);
    await fs.writeFile(trackingFile, JSON.stringify(trackingArray, null, 2), 'utf-8');
    console.log(`Tracking file saved: ${trackingFile} (${trackingArray.length} entries)`);
    console.log('');

    // 打印汇总
    console.log('========================================');
    console.log('              MERGE SUMMARY');
    console.log('========================================');
    for (const [targetClass, typeStats] of Object.entries(stats)) {
        console.log(`\n  Target: ${targetClass}`);
        let classTotal = 0;
        for (const [typeName, s] of Object.entries(typeStats)) {
            console.log(`    ${typeName}: ${s.copied} copied, ${s.skipped} skipped, ${s.errors} errors`);
            classTotal += s.copied;
        }
        console.log(`    [Class total: ${classTotal} images]`);
    }
    console.log(`\n  Grand total: ${totalCopied} copied, ${totalSkipped} skipped, ${totalErrors} errors`);
    console.log(`  Tracking file: ${trackingFile}`);
    console.log('========================================');
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
