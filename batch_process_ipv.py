import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_case(json_path):
    try:
        json_path = Path(json_path)
        xlsx_path = json_path.parent / json_path.name.replace('_metadata.json', '_ipv_results.xlsx')
        
        if not xlsx_path.exists():
            return f"Missing XLSX for {json_path}"

        # 读取 XLSX
        df = pd.read_excel(xlsx_path)
        
        # 按照约定，前两个非 error 的 ipv_ 列分别是 primary 和 secondary
        ipv_cols = [c for c in df.columns if c.startswith('ipv_') and not c.endswith('_error')]
        if len(ipv_cols) < 2:
            return f"Invalid columns in {xlsx_path}: {df.columns.tolist()}"
        
        # 从第 5 行开始（索引 4）
        valid_data = df.iloc[4:]
        
        means = []
        for col in ipv_cols[:2]:
            err_col = col + '_error'
            # 筛选 error < 0.6 的行
            valid_vals = valid_data[valid_data[err_col] < 0.6][col]
            if not valid_vals.empty:
                means.append(float(valid_vals.mean()))
            else:
                means.append("unknown")

        # 读取并更新 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        # 更新 primary
        primary = data.get('pair', {}).get('primary', {})
        new_primary = OrderedDict()
        for k, v in primary.items():
            new_primary[k] = v
            if k == 'vehicle_id':
                new_primary['mean_ipv'] = means[0]
        data['pair']['primary'] = new_primary

        # 更新 secondary
        secondary = data.get('pair', {}).get('secondary', {})
        new_secondary = OrderedDict()
        for k, v in secondary.items():
            new_secondary[k] = v
            if k == 'vehicle_id':
                new_secondary['mean_ipv'] = means[1]
        data['pair']['secondary'] = new_secondary

        # 写回 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return "Success"
    except Exception as e:
        return f"Error processing {json_path}: {str(e)}"

def main():
    root_dir = Path('interhub_traj_lane/ipv_estimation_results')
    
    # 查找所有 metadata.json 文件
    json_files = list(root_dir.glob('**/data/*_metadata.json'))
    total_files = len(json_files)
    logger.info(f"Found {total_files} cases to process.")

    # 使用并行处理
    # 考虑到 Excel 读取是 IO 和 CPU 混合，使用 ProcessPoolExecutor
    # 这里的 max_workers 可以根据核心数调整，默认通常足够
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_case, str(f)): f for f in json_files}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "Success":
                success_count += 1
            else:
                error_count += 1
                logger.error(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total_files} (Success: {success_count}, Error: {error_count})")

    logger.info(f"Finished. Total: {total_files}, Success: {success_count}, Error: {error_count}")

if __name__ == "__main__":
    main()

