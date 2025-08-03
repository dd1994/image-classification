const axios = require('axios');
const fs = require('fs');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');

const API_KEY = '95cf54522c3c48c88d4ab800ec33eb0f';
const INPUT_FILE = 'species_area.csv';
const OUTPUT_FILE = 'species_area_with_distribution.csv';

// 延时函数
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// 处理物种数据
const processSpeciesData = async () => {
  const speciesList = [];

  // 读取原始CSV文件
  fs.createReadStream(INPUT_FILE)
    .pipe(csv())
    .on('data', (row) => {
      // 过滤掉属级（单单词拉丁名）
      if (row.taxonName.trim().includes(' ')) {
        speciesList.push({
          SpeciesID: row.SpeciesID,
          taxonName: row.taxonName,
          chineseName: row.chineseName
        });
      }
    })
    .on('end', async () => {
      console.log(`找到 ${speciesList.length} 个物种需要查询`);

      // 创建CSV写入器
      const csvWriter = createObjectCsvWriter({
        path: OUTPUT_FILE,
        header: [
          {id: 'SpeciesID', title: 'SpeciesID'},
          {id: 'taxonName', title: 'taxonName'},
          {id: 'chineseName', title: 'chineseName'},
          {id: 'distribution', title: 'distribution'}
        ]
      });

      const results = [];

      // 顺序处理每个物种
      for (let i = 0; i < speciesList.length; i++) {
        const species = speciesList[i];

        try {
          console.log(`正在查询 (${i+1}/${speciesList.length}): ${species.taxonName}`);

          const url = `http://www.sp2000.org.cn/api/v2/getSpeciesByScientificName?scientificName=${encodeURIComponent(species.taxonName)}&apiKey=${API_KEY}`;
          const response = await axios.get(url, { timeout: 15000 });

          if (response?.data?.code === 200 &&
              response?.data?.data?.species?.length > 0 &&
              response?.data?.data?.species?.[0]?.accepted_name_info) {

            const distribution = response.data.data.species[0].accepted_name_info.Distribution || '';

            // 提取中文省份（括号内的内容）
            let provinces = '';
            if (distribution) {
              const match = distribution.match(/\(([^)]+)\)/);
              provinces = match ? match[1].replace(/,/g, '、') : '无分布信息';
            } else {
              provinces = '无分布信息';
            }

            results.push({
              ...species,
              distribution: provinces
            });

            console.log(`  分布: ${provinces}`);
          } else {
            results.push({
              ...species,
              distribution: '未找到物种信息'
            });
            console.log('  未找到物种信息');
          }
        } catch (error) {
          console.error(`  查询失败: ${error.message}`);
          results.push({
            ...species,
            distribution: '查询失败'
          });
        }

        // 每次请求后等待3秒（最后一个除外）
        if (i < speciesList.length - 1) {
          console.log('等待3秒...');
          await delay(3000);
        }
      }

      // 写入结果到CSV
      await csvWriter.writeRecords(results);
      console.log(`\n已完成所有查询!`);
      console.log(`结果已保存到 ${OUTPUT_FILE}`);
    });
};

// 启动处理流程
processSpeciesData().catch(console.error);