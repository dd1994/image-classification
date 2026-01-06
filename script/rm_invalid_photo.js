const fs = require('fs');
const path = require('path');

async function removeInvalidPhotos() {
    const csvPath = 'd:/image-classification/script/invalid_photo.csv';
    const trainPrePath = 'd:/image-classification/data/collection-pre';
    
    try {
        console.log('正在读取CSV文件...');
        const csvContent = fs.readFileSync(csvPath, 'utf-8');
        const csvLines = csvContent.split('\n').filter(line => line.trim());
        
        if (csvLines.length <= 1) {
            console.log('CSV文件为空或只有表头');
            return;
        }
        
        const invalidPhotos = [];
        const headerLine = csvLines[0].toLowerCase();
        const hasHeader = headerLine.includes('taxon_id') && headerLine.includes('photo_id');
        
        const startIndex = hasHeader ? 1 : 0;
        
        for (let i = startIndex; i < csvLines.length; i++) {
            const line = csvLines[i].trim();
            if (!line) continue;
            
            const parts = line.split(',');
            if (parts.length >= 2) {
                const taxonId = parts[0].trim();
                const photoId = parts[1].trim();
                if (taxonId && photoId) {
                    invalidPhotos.push({ taxonId, photoId });
                }
            }
        }
        
        console.log(`CSV文件中找到 ${invalidPhotos.length} 条需要删除的记录`);
        
        if (!fs.existsSync(trainPrePath)) {
            console.error(`错误: 找不到目录 ${trainPrePath}`);
            return;
        }
        
        const categoryFolders = fs.readdirSync(trainPrePath, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);
        
        let totalDeleted = 0;
        let totalNotFound = 0;
        let totalErrors = 0;
        
        for (const category of categoryFolders) {
            const categoryPath = path.join(trainPrePath, category);
            console.log(`\n正在处理大类: ${category}`);
            
            const speciesFolders = fs.readdirSync(categoryPath, { withFileTypes: true })
                .filter(dirent => dirent.isDirectory())
                .map(dirent => dirent.name);
            
            const relevantPhotos = invalidPhotos.filter(p => speciesFolders.includes(p.taxonId));
            
            if (relevantPhotos.length === 0) {
                console.log(`  没有找到需要删除的照片`);
                continue;
            }
            
            console.log(`  需要删除 ${relevantPhotos.length} 张照片`);
            
            for (const { taxonId, photoId } of relevantPhotos) {
                const speciesPath = path.join(categoryPath, taxonId);
                const imagePattern = `processed_inaturalist_${photoId}.jpg`;
                const imagePath = path.join(speciesPath, imagePattern);
                
                if (fs.existsSync(imagePath)) {
                    try {
                        fs.unlinkSync(imagePath);
                        console.log(`  ✓ 删除成功: ${imagePattern} (taxon: ${taxonId})`);
                        totalDeleted++;
                    } catch (error) {
                        console.error(`  ✗ 删除失败: ${imagePattern} - ${error.message}`);
                        totalErrors++;
                    }
                } else {
                    console.log(`  ✗ 文件不存在: ${imagePattern} (taxon: ${taxonId})`);
                    totalNotFound++;
                }
            }
        }
        
        console.log(`\n=== 删除完成 ===`);
        console.log(`成功删除: ${totalDeleted} 张`);
        console.log(`文件不存在: ${totalNotFound} 张`);
        console.log(`删除失败: ${totalErrors} 张`);
        
    } catch (error) {
        console.error('执行过程中发生错误:', error);
    }
}

removeInvalidPhotos();