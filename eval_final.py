#!/usr/bin/env python3
"""最终评估: 全31页 final run vs 基线, 对照PDF文字层真值"""
import fitz, re
from collections import Counter

def norm_text(s):
    s = re.sub(r'\s+', '', s)
    table = {}
    for a, b in [('（','('),('）',')'),('：',':'),('；',';'),('，',','),('。','.')]:
        table[ord(a)] = b
    return s.translate(table)

def shingles(s, n=8):
    return set(s[i:i+n] for i in range(len(s)-n+1))

src = fitz.open('/Users/liwenqiang/Downloads/SAMR_NSA_Basic_Security_Requirements_GenAI_service.pdf')
gt = norm_text(''.join(p.get_text() for p in src))
src.close()
g = shingles(gt)

print("=== 全文档对比 (真值 %d字, %d shingles) ===" % (len(gt), len(g)))
for path, label in [('/tmp/ocr_baseline.out', '基线   '), ('/tmp/ocr_final.out', '优化后 ')]:
    ocr = norm_text(open(path).read())
    s = shingles(ocr)
    hit = sum(1 for x in s if x in g)
    cov = sum(1 for x in g if x in s)
    print(f"{label}: 输出{len(ocr)}字 | 命中率 {100*hit/len(s):5.1f}% | 真值覆盖率 {100*cov/len(g):5.1f}%")

# 关键缺失段验证: B.2.3.7.2 预期结果内容(基线完全丢失)
key_phrases = ['端侧具备端侧安全模块', '多次提醒和预警的机制', '上述预期结果均满足判定为符合',
               '生成式人工智能服务管理暂行办法', '网络安全标准实践指南']
print("\n=== 关键内容点验证 (页27-28早停丢失段 + 参考文献) ===")
base = norm_text(open('/tmp/ocr_baseline.out').read())
fin = norm_text(open('/tmp/ocr_final.out').read())
for kp in key_phrases:
    k = norm_text(kp)
    b = '✓' if k in base else '✗'
    f = '✓' if k in fin else '✗'
    print(f"  基线{b} 优化后{f}  {kp}")

# 循环垃圾复测
def norm_line(l):
    return re.sub(r'\s+', '', l)
print("\n=== 循环垃圾行(OCR≥3次且真值0次) ===")
for path, label in [('/tmp/ocr_baseline.out','基线'), ('/tmp/ocr_final.out','优化后')]:
    lines = [l for l in open(path).read().split('\n') if l.strip()]
    c = Counter(norm_line(l) for l in lines)
    gt_n = norm_text(gt)
    garbage = {l: n for l, n in c.items() if n >= 3 and gt_n.count(l) == 0 and len(l) >= 8}
    total_g = sum(garbage.values())
    print(f"{label}: {len(garbage)}种 / {total_g}行")
    for l, n in sorted(garbage.items(), key=lambda x: -x[1])[:3]:
        print(f"    {n}x {l[:48]}")
