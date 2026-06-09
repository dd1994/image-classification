const fs = require('fs').promises;
const path = require('path');

// 图片文件扩展名
const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

const trainDir = "D:/image-classification/data/temp";
const validDir = "D:/image-classification/data/valid";
const trackingFile = "D:/image-classification/data/other_types_files.json";

/**
 * 加载 other_types 追踪文件，返回 Set<绝对路径>
 * 这些文件来自 other-types 特殊形态，必须保留在训练集中
 */
async function loadProtectedFiles() {
    try {
        const data = await fs.readFile(trackingFile, 'utf-8');
        const arr = JSON.parse(data);
        console.log(`Loaded ${arr.length} protected files from ${trackingFile}`);
        return new Set(arr);
    } catch (err) {
        if (err.code === 'ENOENT') {
            console.log('No tracking file found — all files are movable.');
        } else {
            console.error(`Warning: could not read tracking file: ${err.message}`);
        }
        return new Set();
    }
}

// 获取目录中的图片文件，忽略以点开头的文件
async function getImageFiles(dir) {
    try {
        const files = await fs.readdir(dir);
        const imageFiles = files.filter(file =>
            imageExtensions.includes(path.extname(file).toLowerCase()) && !file.startsWith('.')
        );
        return imageFiles;
    } catch (err) {
        console.error(`Error reading directory ${dir}:`, err);
        return [];
    }
}

// 按修改时间排序文件
async function sortFilesByMtime(dir, files) {
    const fileStatsPromises = files.map(async file => {
        const stats = await fs.stat(path.join(dir, file));
        return { file, mtime: stats.mtime };
    });
    const fileStats = await Promise.all(fileStatsPromises);
    fileStats.sort((a, b) => b.mtime - a.mtime);
    return fileStats.map(f => f.file);
}

// 获取验证集目录中已有的图片文件名，忽略以点开头的文件
async function getExistingValidFiles(valid_species_path) {
    try {
        const files = await fs.readdir(valid_species_path);
        return new Set(files.filter(file => !file.startsWith('.')));
    } catch (err) {
        // 如果验证集目录不存在，返回空集合
        return new Set();
    }
}

// 创建目录
async function createDir(dir) {
    try {
        await fs.mkdir(dir, { recursive: true });
    } catch (err) {
        if (err.code !== 'EEXIST') {
            throw err;
        }
    }
}

// 删除目录及其内容
async function deleteDir(dir) {
    try {
        await fs.rm(dir, { recursive: true, force: true });
        console.log(`Deleted directory: ${dir}`);
    } catch (err) {
        console.error(`Error deleting directory ${dir}:`, err);
    }
}

// 主函数
async function main() {
    // 加载受保护文件集合（来自 other-types 的文件，不可移入验证集）
    const protectedFiles = await loadProtectedFiles();

    let totalProtectedCount = 0;  // 统计受保护文件总数

    // 获取所有大类文件夹，忽略以点开头的文件夹
    const classes = (await fs.readdir(trainDir)).filter(name => !name.startsWith('.'));
    for (const class_name of classes) {
        const class_path = path.join(trainDir, class_name);
        const valid_class_path = path.join(validDir, class_name);

        // 确保验证集目录中的大类文件夹存在
        await createDir(valid_class_path);

        // 获取训练集中的所有物种ID文件夹，忽略以点开头的文件夹
        const train_species_ids = (await fs.readdir(class_path)).filter(name => !name.startsWith('.'));
        const train_species_set = new Set(train_species_ids);

        // 获取验证集中的所有物种ID文件夹，忽略以点开头的文件夹
        const valid_species_ids = (await fs.readdir(valid_class_path)).filter(name => !name.startsWith('.'));
        const valid_species_set = new Set(valid_species_ids);

        // 删除验证集中不存在于训练集的物种文件夹
        for (const species_id of valid_species_ids) {
            if (!train_species_set.has(species_id)) {
                const species_valid_path = path.join(valid_class_path, species_id);
                await deleteDir(species_valid_path);
            }
        }

        // 处理训练集中存在的物种
        for (const species_id of train_species_ids) {
            const species_train_path = path.join(class_path, species_id);
            const species_valid_path = path.join(valid_class_path, species_id);

            // 确保验证集目录中的物种ID文件夹存在
            await createDir(species_valid_path);

            // 获取训练集中的图片文件
            const all_train_files = await getImageFiles(species_train_path);
            const train_count = all_train_files.length;

            // 将文件分为"可移动"和"受保护"两类
            const movable_files = [];
            let protected_count = 0;
            for (const file of all_train_files) {
                const fullPath = path.join(species_train_path, file);
                if (protectedFiles.has(fullPath)) {
                    protected_count++;
                } else {
                    movable_files.push(file);
                }
            }
            totalProtectedCount += protected_count;

            // 按修改时间排序可移动文件
            const sorted_movable = await sortFilesByMtime(species_train_path, movable_files);
            const movable_count = sorted_movable.length;

            // 计算验证集应有图片数量：
            // 基于总图片数计算比例（保持比例一致），但不超过可移动文件数
            let desired_valid_count = Math.floor(train_count * 0.05);
            desired_valid_count = Math.max(desired_valid_count, 5);
            desired_valid_count = Math.min(desired_valid_count, 40);

            // 不能超过可移动文件数（受保护文件不能移入验证集）
            if (desired_valid_count > movable_count) {
                console.warn(
                    `  WARNING: ${class_name}/${species_id}: desired_valid=${desired_valid_count} ` +
                    `> movable=${movable_count} (protected=${protected_count}), clamping to ${movable_count}`
                );
                desired_valid_count = movable_count;
            }

            // 获取验证集目录中已有的图片文件名
            const existing_valid_files = await getExistingValidFiles(species_valid_path);
            const current_valid_count = existing_valid_files.size;

            if (current_valid_count >= desired_valid_count) {
                continue;
            }

            // 计算需要移动的图片数量
            const need_to_move = desired_valid_count - current_valid_count;

            // 从可移动文件中选取最新的、尚未在验证集中的文件
            const files_to_move = sorted_movable
                .filter(file => !existing_valid_files.has(file))
                .slice(0, need_to_move);

            // 移动图片到验证集目录
            let moved_count = 0;
            for (const file of files_to_move) {
                const src = path.join(species_train_path, file);
                const dest = path.join(species_valid_path, file);
                try {
                    // 检查源文件是否存在
                    await fs.stat(src);
                    // 检查是否在同一盘符下
                    const sourceParsed = path.parse(src);
                    const destParsed = path.parse(dest);
                    if (sourceParsed.root === destParsed.root) {
                        await fs.rename(src, dest);
                    } else {
                        await fs.copyFile(src, dest);
                        await fs.unlink(src);
                    }
                    moved_count++;
                } catch (err) {
                    console.error(`Error moving ${src} to ${dest}:`, err);
                }
            }

            if (moved_count > 0) {
                console.log(`Species ${species_id} has ${moved_count} images moved to validation set.`);
            }
        }
    }

    console.log(`\nTotal protected files (from other-types, kept in train): ${totalProtectedCount}`);
}

// 运行主函数
main().catch(console.error);
