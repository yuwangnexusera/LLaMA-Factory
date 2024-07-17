import pandas as pd
import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def process_csv_file(input_path, output_path, filter_department):
    # 检测文件编码
    encoding = detect_encoding(input_path)

    # 手动读取文件内容并处理编码错误
    with open(input_path, "r", encoding=encoding, errors="replace") as file:
        content = file.read()

    # 将文件内容转换为pandas的DataFrame
    from io import StringIO

    df = pd.read_csv(StringIO(content))

    # 查看数据
    print("Original DataFrame:")
    print(df.head())

    # 过滤出特定部门的数据
    filtered_df = df[df["department"] == filter_department]

    # 提取特定列
    filtered_columns = filtered_df[["title", "ask", "answer"]]
    for i, row in filtered_columns.iterrows():
        title = row["title"]
        ask = row["ask"]
        answer = row["answer"]
        print(f"Title: {title}")
        print(f"Ask: {ask}")
        print(f"Answer: {answer}")
        print()

    # 保存过滤后的数据


def main():
    input_path = "../data/turmo.csv"
    output_path = "data/QA/turmo.json"
    filter_department = "肺癌"

    process_csv_file(input_path, output_path, filter_department)


if __name__ == "__main__":
    main()
