# tianchi-purchase-redemption

本仓库用于“资金流入流出预测”项目，围绕天池资金流入流出预测赛题整理数据分析、特征工程和建模实验。

## 数据说明

原始数据不上传 GitHub。本地原始数据请放在：

```text
data/raw/
```

清洗后的中间数据、模型输出和提交文件也不应提交到仓库。

## 文档

项目文档位于 `docs/`：

- `docs/背景信息汇总.md`
- `docs/参考思路.md`

## Conda 环境

使用 `environment.yml` 创建项目环境：

```powershell
conda env create -f environment.yml
conda activate tianchi-purchase-redemption
python -m ipykernel install --user --name tianchi-purchase-redemption --display-name "Python (tianchi-purchase-redemption)"
```

之后在 Jupyter Notebook 或 JupyterLab 中选择 `Python (tianchi-purchase-redemption)` 内核。
