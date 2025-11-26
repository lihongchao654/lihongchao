from PyPDF2 import PdfReader
import os

pdf_path = r"d:\新疆大学\文献项目\主余震易损性研究\主余震构造\基于地震动模拟的一致危险谱和条件均值谱生成及应用_朱瑞广.pdf"

if not os.path.exists(pdf_path):
    print(f"文件不存在: {pdf_path}")
else:
    print(f"开始提取PDF内容...\n")
    reader = PdfReader(pdf_path)
    print(f"总页数: {len(reader.pages)}\n")
    
    # 提取所有文本
    all_text = ""
    for i, page in enumerate(reader.pages[:10]):  # 先提取前10页
        text = page.extract_text()
        if text:
            all_text += f"--- 第 {i+1} 页 ---\n" + text + "\n"
    
    # 保存到文件
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    
    print("内容已保存到 pdf_content.txt")
    print(f"总字符数: {len(all_text)}")
    print(f"\n前2000个字符:\n")
    print(all_text[:2000])
