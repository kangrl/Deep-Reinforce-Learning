'''
Author: kangrl
Email: kangrl@live.cn
Date: 2025-02-20 13:12:15
LastEditors: kangrl
LastEditTime: 2025-02-20 13:12:15
FilePath: \Deep-Reinforce-Learning\convert_ipynb_to_md.py
Copyright (C) 2025 by kangrl, All Rights Reserved.
Description:
'''

import os
import nbformat
from nbconvert import MarkdownExporter

def convert_ipynb_to_md(directory):
    # 创建Markdown导出器
    exporter = MarkdownExporter()

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 读取.ipynb文件
            with open(file_path, 'r', encoding='utf-8') as f:
                nb_content = f.read()
            # 将内容转换为Notebook对象
            notebook = nbformat.reads(nb_content, as_version=4)
            # 导出为Markdown
            (body, resources) = exporter.from_notebook_node(notebook)
            # 构建输出文件名
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_file_path = os.path.join(directory, md_filename)
            # 将Markdown内容写入文件
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(body)
            print(f"Converted {filename} to {md_filename}")

# 指定要转换的文件夹路径
directory_path = './02 Concepts/'
convert_ipynb_to_md(directory_path)
