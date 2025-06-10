import os
import glob
import re
import yaml
import pandas as pd
from io import StringIO
import logging

# --- Configuration ---
# Adjust this if your HPO script used a different base output directory
HPO_OUTPUT_BASE_DIR = "18_2/stage2_lstm_output"
# Regex to find trial directories and extract trial number
TRIAL_DIR_PATTERN = r".*_trial_(\d+)$" # Matches directories ending with _trial_<number>
# REPORT_FILENAME = "overall_classification_report.txt" # Remove fixed filename
CONFIG_FILENAME_PATTERN = "effective_config_trial_{}.yaml"
# Default experiment name if config loading fails
DEFAULT_EXP_NAME = "casme2_lstm_stage2_crop_repeat_6class"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_f1_from_report(report_path):
    """Parses the macro average F1-score from the classification report file."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
            # Find the summary table part (usually after the main report)
            # A simple approach: find the line starting with 'macro avg'
            match = re.search(r"^\s*macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", report_content, re.MULTILINE)
            if match:
                macro_f1 = float(match.group(3)) # Group 3 should be the f1-score
                return macro_f1
            else:
                 # Fallback: Try parsing the table if the simple regex fails (more complex)
                 # This part might need adjustment based on exact report format
                 try:
                     # Attempt to read the table section into pandas
                     # Find where the table likely starts (header like precision recall f1-score support)
                     table_start = report_content.find("precision    recall  f1-score   support")
                     if table_start != -1:
                          table_text = report_content[table_start:]
                          # Use StringIO to treat the string as a file
                          report_io = StringIO(table_text)
                          df = pd.read_csv(report_io, sep='\s{2,}', engine='python', skiprows=[1]) # Skip divider line
                          df.columns = [col.strip() for col in df.columns] # Clean column names
                          df = df.set_index(df.columns[0]) # Set first column (class name) as index
                          if 'macro avg' in df.index:
                               return df.loc['macro avg', 'f1-score']
                 except Exception as pd_err:
                     logger.debug(f"Pandas parsing failed for {report_path}: {pd_err}")
                     pass # Ignore pandas errors, rely on regex or return None
                     
            logger.warning(f"无法在 {report_path} 中解析 'macro avg' f1-score。")
            return None
    except FileNotFoundError:
        logger.warning(f"报告文件未找到: {report_path}")
        return None
    except Exception as e:
        logger.error(f"读取或解析报告文件时出错 {report_path}: {e}")
        return None

def find_best_trial(base_dir):
    """Finds the trial with the highest macro F1 score in the output directory."""
    best_trial_info = {"trial_num": -1, "f1_score": -1.0, "dir_path": None}
    
    logger.info(f"正在扫描目录: {base_dir}")
    potential_dirs = []
    try:
        potential_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except FileNotFoundError:
        logger.error(f"基础输出目录未找到: {base_dir}")
        return None
        
    if not potential_dirs:
        logger.warning(f"在 {base_dir} 中未找到任何子目录。")
        return None

    found_trials = 0
    for dir_name in potential_dirs:
        match = re.match(TRIAL_DIR_PATTERN, dir_name)
        if match:
            trial_num = int(match.group(1))
            trial_dir_path = os.path.join(base_dir, dir_name)
            
            # --- Dynamically determine report filename --- 
            exp_name = DEFAULT_EXP_NAME # Start with default
            config_filename = CONFIG_FILENAME_PATTERN.format(trial_num)
            config_path = os.path.join(trial_dir_path, config_filename)
            try:
                with open(config_path, 'r', encoding='utf-8') as f_cfg:
                    trial_config = yaml.safe_load(f_cfg)
                    # Use experiment_name from the trial's config if available
                    exp_name = trial_config.get('experiment_name', DEFAULT_EXP_NAME)
                    logger.debug(f"Trial {trial_num}: Loaded exp_name '{exp_name}' from config.")
            except FileNotFoundError:
                logger.warning(f"Trial {trial_num}: 未找到配置文件 {config_path}，将使用默认 exp_name '{exp_name}' 构建报告路径。")
            except Exception as e_cfg:
                 logger.error(f"Trial {trial_num}: 加载配置文件 {config_path} 时出错: {e_cfg}，将使用默认 exp_name。")
                 
            # Construct the expected report filename using the (potentially loaded) exp_name
            report_filename = f"{exp_name}_overall_classification_report.txt"
            report_path = os.path.join(trial_dir_path, report_filename)
            # --- End dynamic filename determination ---
            
            logger.info(f"正在检查 Trial {trial_num} 在 {trial_dir_path} (寻找报告: {report_filename})")
            f1_score = parse_f1_from_report(report_path) # Use the dynamically constructed path
            
            if f1_score is not None:
                found_trials += 1
                logger.info(f"Trial {trial_num}: Macro F1 = {f1_score:.4f}")
                if f1_score > best_trial_info["f1_score"]:
                    logger.info(f"找到新的最佳 Trial: {trial_num} (F1: {f1_score:.4f})")
                    best_trial_info["trial_num"] = trial_num
                    best_trial_info["f1_score"] = f1_score
                    best_trial_info["dir_path"] = trial_dir_path
            else:
                 logger.warning(f"跳过 Trial {trial_num}，无法获取 F1 分数。")
                 
    if best_trial_info["trial_num"] == -1:
        logger.warning(f"在扫描了 {len(potential_dirs)} 个目录后，未找到任何有效的 Trial 结果。")
        return None
        
    logger.info(f"扫描完成。共找到 {found_trials} 个有效的 Trial 结果。")
    return best_trial_info

if __name__ == "__main__":
    logger.info("--- 开始查找最佳 HPO Trial --- ")
    best_trial = find_best_trial(HPO_OUTPUT_BASE_DIR)
    
    if best_trial:
        logger.info("--- 最佳 Trial 结果 --- ")
        logger.info(f"Trial 编号: {best_trial['trial_num']}")
        logger.info(f"Macro F1 分数: {best_trial['f1_score']:.4f}")
        logger.info(f"结果目录: {best_trial['dir_path']}")
        
        # Try to load and print the parameters used in the best trial
        config_filename = CONFIG_FILENAME_PATTERN.format(best_trial['trial_num'])
        config_path = os.path.join(best_trial['dir_path'], config_filename)
        logger.info(f"正在尝试加载最佳 Trial 的配置文件: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                best_config = yaml.safe_load(f)
                logger.info("最佳 Trial 使用的有效配置:")
                # Pretty print the relevant sections
                print("--- Training Config ---")
                print(yaml.dump({'training': best_config.get('training', {})}, default_flow_style=False))
                print("--- Model Config ---")
                print(yaml.dump({'model': best_config.get('model', {})}, default_flow_style=False))
                print("--- Balancing Config ---")
                print(yaml.dump({'balancing': best_config.get('balancing', {})}, default_flow_style=False))
        except FileNotFoundError:
            logger.warning(f"未找到最佳 Trial 的有效配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载或打印最佳 Trial 配置时出错: {e}")
            
    else:
        logger.info("未能找到最佳 Trial。请检查 HPO 输出目录和报告文件。")

    logger.info("--- 查找脚本结束 --- ") 