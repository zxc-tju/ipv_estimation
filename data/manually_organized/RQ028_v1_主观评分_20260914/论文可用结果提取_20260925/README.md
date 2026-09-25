# 主观评分论文结果提取包

先读 [分析结论与使用建议](分析结论与使用建议.md)，再看 [论文候选文字](论文候选文字.md) 与 [论文衔接位置](论文衔接位置.md)。核心数值在 `tables/paper_core_results.csv`；两张图各有PNG、PDF和SVG。

输入为上一级用户确认的最终版本。本包是探索性提取产物，原数据保持原样，论文未被修改。所有研究数据读取限定于上一级目录。

复算顺序：`python3 process/extract_results.py` → `python3 process/make_figures.py` → `python3 process/write_handoff.py`。只覆盖本提取包的生成文件。脚本不访问其他研究数据；独立核验记录与论文衔接说明是人工/独立代理产物，不由复算脚本生成。
