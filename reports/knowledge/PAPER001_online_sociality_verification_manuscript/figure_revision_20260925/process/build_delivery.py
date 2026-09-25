"""Assemble the audited local delivery without running research computations."""
from pathlib import Path
import base64, csv, hashlib, html, json, re, shutil, subprocess
import fitz
from figure_style import ROOT, RUN, ASSETS, DATA
PAPER=(ROOT/'../../2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle').resolve()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for source,name in [('/tmp/nmi-figrev-final/main.pdf','main_revised.pdf'),('/tmp/nmi-figrev-final/main.log','process/latex_build.log'),('/tmp/nmi-figrev-final-run.log','process/latexmk_run.log')]:
 shutil.copy2(source,RUN/name)
log=(RUN/'process/latex_build.log').read_text(errors='replace')
assert not re.search(r'Undefined control sequence|undefined references|Citation .* undefined|Reference .* undefined|Overfull|LaTeX Warning:',log)
text=(PAPER/'main.tex').read_text();alltext=text+'\n'+(PAPER/'figures/subjective_supplement_tables.tex').read_text()
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',alltext));labels=re.findall(r'\\label\{([^}]+)\}',alltext)
assert refs<=set(labels) and len(labels)==len(set(labels))
assert 'Figure 6 artwork pending' not in text
# The project includes 14 figure blocks: 6 main, 4 supplementary and 4 Extended Data.
blocks=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',text,re.S);assert len(blocks)==14
pdf=fitz.open(RUN/'main_revised.pdf'); page_map=[]
for i,page in enumerate(pdf):
 names=re.findall(r'Figure ((?:[1-6]|S[1-4]|ED[1-4]))[.]',page.get_text())
 for n in names:page_map.append({'figure':n,'page':i+1})
assert len(page_map)==14
inventory=[]
for block in blocks:
 label=re.search(r'\\label\{([^}]+)\}',block).group(1)
 rel=re.search(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',block).group(1)
 f=PAPER/rel;doc=fitz.open(f);inventory.append({'label':label,'file':rel,'sha256':sha(f),'width_pt':doc[0].rect.width,'height_pt':doc[0].rect.height,'page_count':len(doc)})
with (RUN/'figure_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,inventory[0].keys());w.writeheader();w.writerows(inventory)
# Verify every installed new asset and plot-data CSV against the run copy.
for f in ASSETS.iterdir():
 if f.suffix in ['.pdf','.png','.svg','.tex']:assert sha(f)==sha(PAPER/'figures'/f.name),f.name
for f in DATA.glob('*.csv'):assert sha(f)==sha(PAPER/'figures/source_data'/f.name),f.name
source_rows=list(csv.DictReader((RUN/'panel_source_map.csv').open()))
for row in source_rows:assert sha(ROOT/row['source'])==row['sha256'],row['source']
checks={'status':'PASS','pdf_pages':len(pdf),'figure_caption_pairs':len(page_map),'main_panel_count':24,'new_or_rebuilt_figure_assets':11,'main_figures':6,'preserved_legacy_figure_assets':3,'plot_data_csv_files':len(list(DATA.glob('*.csv'))),'panel_source_relationships':len(source_rows),'unique_mapped_input_files':len({r['source'] for r in source_rows}),'source_hashes_match':True,'installed_assets_match':True,'latex_errors':0,'undefined_references_or_citations':0,'overfull_boxes':0,'duplicate_labels':0,'figure6_placeholder_removed':True,'participant_level_subjective_exports':0,'new_statistical_tests_or_bootstrap_runs':0,'page_map':page_map,'main_pdf_sha256':sha(RUN/'main_revised.pdf'),'main_tex_sha256':sha(PAPER/'main.tex'),'main_pdf_visual_qa':'process/compiled_main_figures_visual_qa.json','supplemental_visual_qa':'process/visual_verdict.json','open_documentation':['recruitment and apparatus','questionnaire wording and anchors, especially Q5','ethics and consent','stimulus selection and monitor-version relationship','separate data-sharing permission']}
(RUN/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
main_names=[('1','fig1_monitoring','监测逻辑与同一真实案例'),('2','fig2_context','互补性、早期角色与情境'),('3','fig3_monitor','参考范围性能与门控计数'),('4','fig4_consequence','两方向的常规响应与紧急尾部'),('5','fig5_human_arm','匹配人类对照与方向拆分'),('6','fig6_subjective','主观不典型性、舒适度与报告代价')]
notes=[('Figure 2','34,850 个独立案例的联合分布；34,757 个匹配支持案例的汇总相关；34,645 个独立实现案例，各自口径保留。AUC 标签是后段 IPV 角色，不是实际通行先后。'),('Figure 4','沿用真实三秒窗口；两方向、四个 TTC 阈值和三个制动阈值完整显示。迁就侧绝对差区间缺失时保留现成比值区间，不反推。'),('Figure 5','恢复强势/迁就方向拆分，保留全部15场景。人类裕度与尾部比值缺失区间明确标注；自然参考仅作为报警膨胀检查。'),('Figure 6','参与者等权均值、20对重采样的现成区间；Q5只画对方位置。完整30比较与60/30/30项敏感性结果保留。C−W不典型性按源精度显示+0.6825。'),('Source populations','486,660个人类可读时刻的直方图与67,861个部署候选时刻的区间汇总分开；上下界各自的中位数不是任一单帧区间。')]
intro=f'''本轮目标是依据指定修图计划重构六张主图，并核验每个面板的数据、图注及最终排版。现已完成六张主图和五张迁移/补充图；保留原有三张补充或扩展图。全部 {len(page_map)} 组图与图注出现在 {len(pdf)} 页论文中。未新增实验、统计检验或 bootstrap，未改变冻结方法、阈值、纳入规则及原始结果。'''
report=['# 六图重构交付与验收（2026-09-25）',intro,'## 完成状态','- 图形、图注、面板引用与 Source Data 已同步。','- LaTeX 编译通过：无 undefined reference/citation、无 overfull box。','- 六张主图逐页独立视觉核验通过；新增补图与完整补充表已检查。','- 所有映射输入哈希与安装资产哈希一致；个人级主观记录未导出。','- 仅本地提交，不包含远端推送或云同步完成声明。','## 数据口径与本轮修正']
report += [f'- **{a}**：{b}' for a,b in notes]
report += ['## 仍待作者补充','问卷完整题干与锚点、研究装置与分配/顺序、伦理与同意、刺激选择和监测器版本关系、独立的数据共享许可仍需原始材料。图片完成不等于可以投稿。','## 文件入口','- `main_revised.pdf`：完整修订论文。','- `index.html`：含内嵌缩略图的离线汇总。','- `figure_inventory.csv` / `panel_source_map.csv`：图号、标签、静态资产及逐面板来源。','- `assets/`：11套 PDF/SVG/PNG 与补充表 TeX。','- `source_data/`：作图数据与完整聚合统计表。','- `process/`：可复现脚本、输入计划、独立核验、编译日志和视觉验收。','- `verification.json` / `MANIFEST.sha256`：最终验收计数与文件校验。','## 复现','Python绘图入口：`process/make_fig2.py`, `make_fig5.py`, `make_fig6.py`, `make_fig4.py`, `make_fig1.py`, `make_fig3.py`；脚本仅读取指定既有结果。`make_supplement_tables.py` 排版聚合表，`integrate_manuscript.py` 对本轮记录的 v4.5 文本基线安装图片和图注（已有后续修改时不要直接重放）。在论文根目录运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=<build-dir> main.tex`。','## 可用性边界','研究Git分叉已安全合并。原本的归档删除/未提交更改及论文未跟踪文件仍保留；详见 `git_sync_receipt.json`。图件依赖的原始研究结果仍在本地研究目录，论文编译只依赖本论文仓库内的静态资产。']
(RUN/'README.md').write_text('\n\n'.join(report)+'\n')
style='body{font-family:Arial,"PingFang SC",sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#242424;line-height:1.65;background:#fafbfc}h1{font-size:30px}h2{margin-top:2em}a{color:#2c737a}section{background:white;border:1px solid #d8e1e4;padding:24px;border-radius:8px;margin:20px 0}img{width:100%;height:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.tag{display:inline-block;background:#e4f1ed;padding:5px 12px;border-radius:5px;margin-right:8px}.note{border-left:4px solid #b64342;padding-left:14px}code{font-size:90%;overflow-wrap:anywhere}'
page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NMI 六图重构验收</title><style>'+style+'</style><body>','<h1>NMI 六图重构：完成与验收</h1>',f'<p>{html.escape(intro)}</p>','<p><span class="tag">六张主图已完成</span><span class="tag">源数据逐面板核验</span><span class="tag">本地编译与视觉检查通过</span></p>','<p><a href="main_revised.pdf">完整修订论文 PDF</a> · <a href="panel_source_map.csv">面板来源表</a> · <a href="figure_inventory.csv">图件清单</a> · <a href="verification.json">验收记录</a></p>','<h2>关键口径与修正</h2><table><tr><th>位置</th><th>结论与边界</th></tr>']
page += [f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>' for a,b in notes];page.append('</table>')
for n,name,title in main_names:
 enc=base64.b64encode((ASSETS/f'{name}.png').read_bytes()).decode()
 page.append(f'<section id="fig{n}"><h2>Figure {n} · {title}</h2><p><a href="assets/{name}.pdf">PDF</a> · <a href="assets/{name}.svg">可编辑 SVG</a> · <a href="assets/{name}.png">PNG</a></p><img alt="Figure {n}: {title}" src="data:image/png;base64,{enc}"></section>')
page += ['<h2>保留与迁移</h2><p>原 Supplementary S1、Extended Data ED1/ED2 保留。共同场景敏感性为 S2，80/95%匹配比较为 S3，候选角度与权重为 S4。来源迁移与弃权为 ED3；人类分布与部署区间为 ED4。完整主观与对方响应结果置于补充表 S2–S7。</p>','<p class="note">仍待补充：问卷、装置与程序、伦理与同意、刺激版本以及独立数据共享许可。图件已完成不代表投稿条件已齐全。</p>','<p>所有图片内嵌，可离线阅读。数据清单、可复现脚本与日志位于同目录。未向远端推送。</p></body></html>']
(RUN/'index.html').write_text('\n'.join(page))
print(json.dumps(checks,ensure_ascii=False))
