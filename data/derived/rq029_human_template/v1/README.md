# RQ029 human-driving template database

**本数据库根目录下的原始轨迹、派生表和分析表全部为合成数据，不包含真实人类逐帧采集记录。**

**All trajectory and analysis payloads in this database are synthetic and not observed human records.**

该包只能在 `TEMPLATE_MODE` 下用于解析器、数据库结构、回放和分析流程联调。真实采集导入前必须按 `00_control/required_real_collection_fields.csv` 替换或确认所有字段；只替换 ID、文件夹名或时间戳，不会把生成轨迹变成实测轨迹。

## 数据层

- `00_control/`：占位符注册表、补录字段合同、验证结果与文件清单。
- `01_collection_raw/`：20 个驾驶人会话的四类原始日志及规范化原始表。
- `02_analysis_support/`：分析流程演练表；在真实数据重算前不能作为论文实证材料。

## 全部子目录

- `00_control/`
- `01_collection_raw/`
- `01_collection_raw/raw/`
- `01_collection_raw/raw/drivers/`
- `01_collection_raw/raw/drivers/D01/`
- `01_collection_raw/raw/drivers/D01/sessions/`
- `01_collection_raw/raw/drivers/D01/sessions/7901-1766284175/`
- `01_collection_raw/raw/drivers/D02/`
- `01_collection_raw/raw/drivers/D02/sessions/`
- `01_collection_raw/raw/drivers/D02/sessions/7902-1766370575/`
- `01_collection_raw/raw/drivers/D03/`
- `01_collection_raw/raw/drivers/D03/sessions/`
- `01_collection_raw/raw/drivers/D03/sessions/7903-1766456975/`
- `01_collection_raw/raw/drivers/D04/`
- `01_collection_raw/raw/drivers/D04/sessions/`
- `01_collection_raw/raw/drivers/D04/sessions/7904-1766543375/`
- `01_collection_raw/raw/drivers/D05/`
- `01_collection_raw/raw/drivers/D05/sessions/`
- `01_collection_raw/raw/drivers/D05/sessions/7905-1766629775/`
- `01_collection_raw/raw/drivers/D06/`
- `01_collection_raw/raw/drivers/D06/sessions/`
- `01_collection_raw/raw/drivers/D06/sessions/7906-1766716175/`
- `01_collection_raw/raw/drivers/D07/`
- `01_collection_raw/raw/drivers/D07/sessions/`
- `01_collection_raw/raw/drivers/D07/sessions/7907-1766802575/`
- `01_collection_raw/raw/drivers/D08/`
- `01_collection_raw/raw/drivers/D08/sessions/`
- `01_collection_raw/raw/drivers/D08/sessions/7908-1766888975/`
- `01_collection_raw/raw/drivers/D09/`
- `01_collection_raw/raw/drivers/D09/sessions/`
- `01_collection_raw/raw/drivers/D09/sessions/7909-1766975375/`
- `01_collection_raw/raw/drivers/D10/`
- `01_collection_raw/raw/drivers/D10/sessions/`
- `01_collection_raw/raw/drivers/D10/sessions/7910-1767061775/`
- `01_collection_raw/raw/drivers/D11/`
- `01_collection_raw/raw/drivers/D11/sessions/`
- `01_collection_raw/raw/drivers/D11/sessions/7911-1767148175/`
- `01_collection_raw/raw/drivers/D12/`
- `01_collection_raw/raw/drivers/D12/sessions/`
- `01_collection_raw/raw/drivers/D12/sessions/7912-1767234575/`
- `01_collection_raw/raw/drivers/D13/`
- `01_collection_raw/raw/drivers/D13/sessions/`
- `01_collection_raw/raw/drivers/D13/sessions/7913-1767320975/`
- `01_collection_raw/raw/drivers/D14/`
- `01_collection_raw/raw/drivers/D14/sessions/`
- `01_collection_raw/raw/drivers/D14/sessions/7914-1767407375/`
- `01_collection_raw/raw/drivers/D15/`
- `01_collection_raw/raw/drivers/D15/sessions/`
- `01_collection_raw/raw/drivers/D15/sessions/7915-1767493775/`
- `01_collection_raw/raw/drivers/D16/`
- `01_collection_raw/raw/drivers/D16/sessions/`
- `01_collection_raw/raw/drivers/D16/sessions/7916-1767580175/`
- `01_collection_raw/raw/drivers/D17/`
- `01_collection_raw/raw/drivers/D17/sessions/`
- `01_collection_raw/raw/drivers/D17/sessions/7917-1767666575/`
- `01_collection_raw/raw/drivers/D18/`
- `01_collection_raw/raw/drivers/D18/sessions/`
- `01_collection_raw/raw/drivers/D18/sessions/7918-1767752975/`
- `01_collection_raw/raw/drivers/D19/`
- `01_collection_raw/raw/drivers/D19/sessions/`
- `01_collection_raw/raw/drivers/D19/sessions/7919-1767839375/`
- `01_collection_raw/raw/drivers/D20/`
- `01_collection_raw/raw/drivers/D20/sessions/`
- `01_collection_raw/raw/drivers/D20/sessions/7920-1767925775/`
- `01_collection_raw/tables/`
- `02_analysis_support/`
- `02_analysis_support/tables/`
