#!/usr/bin/env python3
"""A/B 评估: 新旧清理管线输出对比 + 退化标记检查（结果输出到 stdout）"""
import fitz, re

def norm_text(s):
    s = re.sub(r'\s+', '', s)
    table = {}
    for a, b in [('（','('),('）',')'),('：',':'),('；',';'),('，',','),('。','.')]:
        table[ord(a)] = b
    return s.translate(table)

def shingles(s, n=8):
    return set(s[i:i+n] for i in range(len(s)-n+1))

# 1) 检查两次 run 的 stderr 事件
for log in ['/tmp/ocr_oldclean_2328.log', '/tmp/ocr_fixed_2328.log']:
    txt = open(log).read()
    events = [f"{kw}×{txt.count(kw)}" for kw in ['早停', '退化重试', '疑似重复', '内容循环']]
    print(f"{log}: {', '.join(events)}")

# 2) 输出 diff
a = open('/tmp/ocr_oldclean_2328.out','rb').read()
b = open('/tmp/ocr_fixed_2328.out','rb').read()
print(f"\nold-clean 输出 {len(a)}B, fixed 输出 {len(b)}B, 相同={a==b}")

# 3) 基线全文档 run 的事件数(对照)
base = open('/tmp/ocr_baseline.log').read()
print(f"基线全31页run: 早停×{base.count('早停')}, 退化重试×{base.count('退化重试')}")

# 4) 对6页真值评估两次输出
src = fitz.open('/tmp/pages_23_28.pdf')
gt = norm_text(''.join(p.get_text() for p in src))
src.close()
g = shingles(gt)
for path, label in [('/tmp/ocr_oldclean_2328.out','旧管线  '), ('/tmp/ocr_fixed_2328.out','新管线r5')]:
    ocr = norm_text(open(path).read())
    s = shingles(ocr)
    hit = sum(1 for x in s if x in g)
    cov = sum(1 for x in g if x in s)
    print(f"{label}: 输出{len(ocr)}字 | 命中率 {100*hit/len(s):.1f}% | 覆盖率 {100*cov/len(g):.1f}%")
