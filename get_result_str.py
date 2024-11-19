import re
import sys

def extract_metrics(file_path):
    # 读取文件的最后一行
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()

    # 正则表达式模式
    # pattern = r'ndcg@(\d+)=([\d.]+) recall@(\d+)=([\d.]+) .*? mrr@(\d+)=([\d.]+)'
    pattern = r'ndcg@(\d+)=([\d.]+)'

    # 提取数据
    matches = re.findall(pattern, last_line)

    # 过滤并整理结果
    results = []
    for match in matches:
        idx = int(match[0])
        if idx in {10, 20, 50, 100}:
            # results.extend([match[1], match[3], match[5]])
            results.extend([match[1]])

    # 格式化结果
    result_str = ' '.join(results)
    return result_str

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = extract_metrics(file_path)
    print(result)
