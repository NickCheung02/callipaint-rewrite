'''
Transfer train json to test json format
'''
import json

def convert_poem_data(input_path, output_path, max_items=100):
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备新格式的数据
    new_data = {"data_list": []}
    
    # 获取前max_items条数据（如果数据不足max_items条，则全部保留）
    items_to_process = data["data_list"][:max_items]
    
    # 处理每个数据项
    for item in items_to_process:
        # 创建新的annotations列表
        new_annotations = []
        for ann in item["annotations"]:
            # 转换polygon坐标：浮点数 -> 整数（四舍五入）
            rounded_polygon = []
            for point in ann["polygon"]:
                # 四舍五入并转换为整数
                x = int(round(point[0]))
                y = int(round(point[1]))
                rounded_polygon.append([x, y])
            
            # 安全获取字段，使用默认值处理缺失字段
            text = ann.get("text", "")
            # 修正字段名拼写错误，并处理缺失情况
            valid_value = ann.get("vaild", True)  # 如果不存在，使用True作为默认值
            
            # 转换每个annotation
            new_ann = {
                "polygon": rounded_polygon,
                "text": text,
                "valid": valid_value,  # 修正拼写错误
                "pos": 0               # 新增固定位置字段
            }
            new_annotations.append(new_ann)
        
        # 安全获取其他字段
        img_name = item.get("img_name", "")
        caption = item.get("caption", "")
        
        # 构建新的数据项
        new_item = {
            "img_name": img_name,
            "caption": caption,
            "annotations": new_annotations
        }
        new_data["data_list"].append(new_item)
    
    # 写入新JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    # 打印处理结果
    print(f"成功转换 {len(new_data['data_list'])} 条数据 (原始数据共 {len(data['data_list'])} 条)")

# 使用示例
convert_poem_data(
    input_path="poem_data/new_data.json",
    output_path="poem_data/processed_data.json",
    max_items=100
)