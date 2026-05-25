import os
import re
import yaml
import time
import ast

# ==============================================================================
#  1. 基础路径配置 (请确保这里指向你的 train.py)
# ==============================================================================
TRAIN_SCRIPT_PATH = 'train.py'  # 你的训练脚本文件名
DEFAULT_YAML_PATH = './models_yaml/anytext2_sd15.yaml' # 如果train.py里没找到，用这个默认值
# ==============================================================================

def parse_train_py(file_path):
    """
    静态分析 train.py 文件，提取变量，不执行代码
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    info = {
        'training_stage': None,
        'resume_path': 'Unknown',
        'config_path': None,
        'root_dir': './checkpoints', # 默认值
        'json_paths': [],
        'base_ckpt_source': 'Unknown'
    }

    # 1. 提取 TRAINING_STAGE
    stage_match = re.search(r'^TRAINING_STAGE\s*=\s*(\d+)', content, re.MULTILINE)
    if stage_match:
        info['training_stage'] = int(stage_match.group(1))

    # 2. 提取 config_path (YAML路径)
    config_match = re.search(r'config_path\s*=\s*[\'"](.*?)[\'"]', content)
    if config_match:
        info['config_path'] = config_match.group(1)

    # 3. 提取 root_dir (保存路径)
    root_match = re.search(r'root_dir\s*=\s*[\'"](.*?)[\'"]', content)
    if root_match:
        info['root_dir'] = root_match.group(1)

    # 4. 智能提取 resume_path (基于 Stage 判断)
    # 你的 train.py 逻辑是 if TRAINING_STAGE == 1 ... else ...
    if info['training_stage'] == 1:
        # 查找 stage 1 的路径
        match = re.search(r'if TRAINING_STAGE == 1:.*?resume_path\s*=\s*[\'"](.*?)[\'"]', content, re.DOTALL)
        if match:
            info['resume_path'] = match.group(1)
            info['base_ckpt_source'] = "Stage 1 Logic (Image Gen)"
    else:
        # 查找 stage 2 的路径 (通常在 else 里)
        # 这里用简化的逻辑：查找 else 块里的 resume_path
        match = re.search(r'else:\s*.*?resume_path\s*=\s*[\'"](.*?)[\'"]', content, re.DOTALL)
        if match:
            info['resume_path'] = match.group(1)
            info['base_ckpt_source'] = "Stage 2 Logic (Text-in-Image)"

    # 5. 提取 json_paths (数据集)
    # 因为 json_paths 是一个列表，可能跨多行，且包含注释，我们使用 AST 解析更安全
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            # 寻找 if __name__ == '__main__' 下面的 json_paths
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'json_paths':
                        # 提取列表中的字符串
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant): # Python 3.8+
                                    info['json_paths'].append(elt.value)
                                elif isinstance(elt, ast.Str): # Old Python
                                    info['json_paths'].append(elt.s)
                                elif isinstance(elt, ast.Call): # 处理 replace 等调用
                                    info['json_paths'].append("Dynamic Path (calculated in code)")
    except Exception as e:
        print(f"⚠️ 解析 json_paths 时遇到复杂结构，尝试正则提取... ({e})")
        # 如果 AST 失败，尝试正则兜底
        json_block = re.search(r'json_paths\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if json_block:
            lines = json_block.group(1).split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    clean_path = line.strip("', ")
                    if clean_path:
                        info['json_paths'].append(clean_path)

    return info

def parse_yaml_config(yaml_path):
    """
    解析 YAML 获取注入配置和书法开关
    """
    info = {
        'injection': 'Unknown',
        'use_calligraphy': 'Unknown'
    }
    
    if not os.path.exists(yaml_path):
        print(f"⚠️ 警告: 找不到 YAML 文件 {yaml_path}")
        return info

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        params = config.get('model', {}).get('params', {})
        
        # 1. 注入方式
        info['injection'] = params.get('context_injection_config', 'Not Set')
        
        # 2. 书法风格开关
        emb_conf = params.get('embedding_manager_config', {}).get('params', {})
        info['use_calligraphy'] = emb_conf.get('use_calligraphy_style', False)
        
    except Exception as e:
        print(f"❌ YAML 解析失败: {e}")
        
    return info

def main():
    print("🔍 正在读取当前环境配置...")
    
    # 1. 解析 train.py
    train_info = parse_train_py(TRAIN_SCRIPT_PATH)
    if not train_info:
        return

    # 2. 解析 YAML
    yaml_file = train_info.get('config_path') or DEFAULT_YAML_PATH
    yaml_info = parse_yaml_config(yaml_file)

    # 3. 准备日志内容
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_lines = []
    log_lines.append(f"========================================")
    log_lines.append(f"      AnyText2 训练环境快照")
    log_lines.append(f"========================================")
    log_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"来源脚本: {TRAIN_SCRIPT_PATH}")
    log_lines.append(f"")
    log_lines.append(f"[关键设置]")
    log_lines.append(f"----------------------------------------")
    log_lines.append(f"• 训练阶段 (Stage)     : {train_info['training_stage']}")
    log_lines.append(f"• 书法风格 (Calligraphy): {yaml_info['use_calligraphy']}")
    log_lines.append(f"• 基础模型 (Resume Ckpt): {train_info['resume_path']}")
    log_lines.append(f"  └─ 来源逻辑          : {train_info['base_ckpt_source']}")
    log_lines.append(f"")
    log_lines.append(f"[注入配置 (Injection)]")
    log_lines.append(f"----------------------------------------")
    inj = yaml_info['injection']
    if isinstance(inj, dict):
        for k, v in inj.items():
            log_lines.append(f"• {k:<10}: {v}")
    else:
        log_lines.append(f"• {inj}")
    log_lines.append(f"")
    log_lines.append(f"[数据集列表 (Datasets)]")
    log_lines.append(f"----------------------------------------")
    if not train_info['json_paths']:
        log_lines.append("⚠️ 未检测到有效的数据集路径 (请检查 train.py if __name__ 块)")
    for p in train_info['json_paths']:
        log_lines.append(f"• {p}")
    
    log_lines.append(f"")
    log_lines.append(f"[文件引用]")
    log_lines.append(f"• YAML Config: {yaml_file}")

    # 4. 保存日志
    save_dir = os.path.join(train_info['root_dir'], 'manual_logs')
    os.makedirs(save_dir, exist_ok=True)
    
    log_filename = f"config_check_{timestamp}_Stage{train_info['training_stage']}.txt"
    save_path = os.path.join(save_dir, log_filename)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
        
    # 5. 打印结果到屏幕
    print('\n'.join(log_lines))
    print(f"\n✅ 日志文件已生成: {save_path}")

if __name__ == "__main__":
    main()