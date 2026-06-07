const fs = require('fs').promises;
const path = require('path');

// 定义图片文件扩展名
const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

const trainDir = "D:/image-classification/data/train";
const validDir = "D:/image-classification/data/valid";

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

            // 获取训练集中的图片文件，并按修改时间排序，忽略以点开头的文件
            const train_imageFiles = await getImageFiles(species_train_path);
            const sorted_train_files = await sortFilesByMtime(species_train_path, train_imageFiles);
            const train_count = sorted_train_files.length;

            // 计算验证集应有图片数量，取 5%，但是至少 5 张，至多 40 张
            let desired_valid_count = Math.floor(train_count * 0.05);
            desired_valid_count = Math.max(desired_valid_count, 5);
            desired_valid_count = Math.min(desired_valid_count, 40);

            // 获取验证集目录中已有的图片文件名，忽略以点开头的文件
            const existing_valid_files = await getExistingValidFiles(species_valid_path);
            const current_valid_count = existing_valid_files.size;

            if (current_valid_count >= desired_valid_count) {
                continue;
            }

            // 计算需要移动的图片数量
            const need_to_move = desired_valid_count - current_valid_count;

            // 从训练集中选择未存在于验证集中的最新的图片，忽略以点开头的文件
            const files_to_move = sorted_train_files.filter(file => !existing_valid_files.has(file)).slice(0, need_to_move);

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
}

// 运行主函数
main().catch(console.error);