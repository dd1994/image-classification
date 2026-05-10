## 提高大规模细粒度识别率相关论文汇总

### Bag of Tricks and a Strong Baseline for FGVC
pdf 下载地址：https://ceur-ws.org/Vol-3180/paper-182.pdf
github 地址： https://github.com/wujiekd/Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification-Mindspore

### Watch out Venomous Snake Species: A Solution to SnakeCLEF2023
pdf 下载地址：https://export.arxiv.org/pdf/2307.09748v1
github 地址：https://github.com/xiaoxsparraw/CLEF2023

### A Deep Learning based Solution to FungiCLEF2023
pdf 下载地址：https://ceur-ws.org/Vol-3497/paper-173.pdf
gtihub 地址：https://github.com/xiaoxsparraw/CLEF2023

### Metaformer Model with ArcFaceLoss and Contrastive Learning for SnakeCLEF2023 Fine-Grained Classification
pdf 下载地址：https://ceur-ws.org/Vol-3497/paper-180.pdf
github 地址：https://github.com/BAOfanTing/SnakeCLEF2023

### Entropy-guided Open-set Fine-grained Fungi Recognition
pdf 下载地址：https://ceur-ws.org/Vol-3497/paper-179.pdf
github 地址：https://ceur-ws.org/Vol-3497/paper-179.pdf


### OpenWGAN-GPforFine-Grained Open-Set Fungi Classification
pdf 下载地址：https://ceur-ws.org/Vol-3740/paper-195.pdf
gtihub 地址： https://github.com/Jack-Etheredge/fungiclef2024

### Generalizable Training Techniques for Fine-Grained Long-Tailed Image Recognition: Transferring Methods Optimized for FungiCLEF 2024 to SnakeCLEF 2024
PDF 下载地址：https://ceur-ws.org/Vol-3740/paper-194.pdf
github 地址：https://github.com/Jack-Etheredge/snakeclef2024


### WhenLarge Kernel Meets Vision Transformer: A Solution for SnakeCLEF & FungiCLEF
PDF 下载地址：https://ceur-ws.org/Vol-3180/paper-175.pdf
github 地址：https://github.com/sinbais/CLEF2022

### 1st Place Solution for FungiCLEF 2022 Competition: Fine-grained Open-set Fungi Recognition

PDF 下载地址：https://ceur-ws.org/Vol-3180/paper-178.pdf
github 地址：https://github.com/guoshengcv/fgvc9_fungiclef




要求：
`D:\fgvc-survey` 文件夹下包含了多篇提升大规模细粒度分类识别率的论文及其代码，我在训练一个支持国内 4 万种动植物分类的图像识别模型，代码在 `D:\image-classification`，希望从上述论文和代码中获取一些经验。请阅读论文和对应代码，总结论文和代码中用到识别率提升技巧, 总结结果以清晰已读的形式呈现。
* 这个任务会比较重，可以开启 sub agent 进行调研。
* 每篇调研结果对应写入到 `D:\image-classification\docs` 目录下的一个 markdown 文件，最后再进行一次汇总。
* 调研完成后，总结 `D:\image-classification` 项目还有哪些尝试改进的空间(包括未尝试的方法和超参数设置的不合理等)。
* 不仅要总结用到提升识别率的方法，还要附上方法对应的超参数，因为超参数也很重要。