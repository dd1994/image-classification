const fs = require('fs');
const path = require('path');

async function removeInvalidTaxa() {
    const csvPath = 'd:/image-classification/temp_valid_other_taxon_count_20251111.csv';
    const trainPrePath = 'd:/image-classification/data/train-pre';
    
    try {
        // 读取CSV文件
        console.log('正在读取CSV文件...');
        const csvContent = fs.readFileSync(csvPath, 'utf-8');
        const csvLines = csvContent.split('\n').filter(line => line.trim());
        
        // 获取有效的taxonID列表（跳过第一行header）
        const validTaxonIds = new Set();
        for (let i = 1; i < csvLines.length; i++) {
            const taxonId = csvLines[i].trim();
            if (taxonId && taxonId !== 'taxonID') {
                validTaxonIds.add(taxonId);
            }
        }
        
        console.log(`CSV文件中找到 ${validTaxonIds.size} 个有效的taxonID`);
        
        // 检查train-pre目录是否存在
        if (!fs.existsSync(trainPrePath)) {
            console.error(`错误: 找不到目录 ${trainPrePath}`);
            return;
        }
        
        // 获取所有大类文件夹
        const categoryFolders = fs.readdirSync(trainPrePath, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);
        
        console.log(`找到 ${categoryFolders.length} 个大类文件夹: ${categoryFolders.join(', ')}`);
        
        let totalDeletedFolders = 0;
        
        // 遍历每个大类文件夹
        for (const category of categoryFolders) {
            const categoryPath = path.join(trainPrePath, category);
            console.log(`\n正在处理大类: ${category}`);
            
            // 获取该类别下的所有物种文件夹
            const speciesFolders = fs.readdirSync(categoryPath, { withFileTypes: true })
                .filter(dirent => dirent.isDirectory())
                .map(dirent => dirent.name);
            
            console.log(`  发现 ${speciesFolders.length} 个物种文件夹`);
            
            let categoryDeletedFolders = 0;
            
            // 检查每个物种文件夹
            for (const speciesFolder of speciesFolders) {
                // 检查物种ID是否在有效列表中
                if (!validTaxonIds.has(speciesFolder)) {
                    const folderPath = path.join(categoryPath, speciesFolder);
                    
                    try {
                        // 删除文件夹及其所有内容
                        // fs.rmSync(folderPath, { recursive: true, force: true });
                        
                        categoryDeletedFolders++;
                        totalDeletedFolders++;
                        
                        console.log(`  删除: ${speciesFolder}`);
                        
                    } catch (error) {
                        console.error(`  删除失败: ${speciesFolder} - ${error.message}`);
                    }
                }
            }
            
            console.log(`  ${category} 大类中删除了 ${categoryDeletedFolders} 个文件夹`);
        }
        
        console.log(`\n=== 删除完成 ===`);
        console.log(`总共删除了 ${totalDeletedFolders} 个文件夹`);
        console.log(`保留了 ${validTaxonIds.size} 个有效taxonID对应的文件夹`);
        
    } catch (error) {
        console.error('执行过程中发生错误:', error);
    }
}

// 运行函数
removeInvalidTaxa();