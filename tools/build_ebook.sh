#!/usr/bin/env bash
# build_ebook.sh — 从 docs/ 构建三格式电子书（epub 主版 + Kindle Paperwhite 3 专用 azw3/mobi）
#
# 用法:  ./tools/build_ebook.sh [输出目录]   （默认当前目录，即仓库根）
#
# 产物:
#   ds-ocr-docs.epub          — 主版（保留 emoji 与 light-dark() 配色，供 Apple Books/Calibre 等现代阅读器）
#   ds-ocr-docs.azw3          — Kindle 版（emoji 剥离/替换、CSS 降级、kindle_pw3 输出配置，USB 拷入即读）
#   ds-ocr-docs.mobi          — Kindle 版 KF8（与 azw3 同源同构，仅扩展名不同）
#
# 依赖: pandoc（构建 epub）、calibre 的 ebook-convert（转 azw3/mobi，
#       macOS 位于 /Applications/calibre.app/Contents/MacOS/ebook-convert）

set -euo pipefail
cd "$(dirname "$0")/.."                 # 仓库根

OUT="${1:-.}"
CALIBRE_EBOOK_CONVERT="${CALIBRE_EBOOK_CONVERT:-/Applications/calibre.app/Contents/MacOS/ebook-convert}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ── 章节顺序与旧版一致: README → 00…15 → 术语表 ─────────────────────────
MD_FILES=(
  docs/README.md
  docs/00-零基础起步篇.md
  docs/01-Transformer基础篇.md
  docs/02-MoE混合专家篇.md
  docs/03-推理引擎实战篇.md
  docs/04-完整推理流程篇.md
  docs/05-性能优化实战篇.md
  docs/06-视觉编码器篇.md
  docs/07-分词与词表篇.md
  docs/08-解码策略与后处理篇.md
  docs/09-量化与部署篇.md
  docs/10-论文导读与进阶路线篇.md
  docs/11-从零构建大模型：预训练篇.md
  docs/12-从零构建大模型：后训练与对齐篇.md
  docs/13-进阶专题篇.md
  docs/14-V4与DASH注意力篇.md
  docs/15-代码逐文件导读篇.md
  docs/00-术语表-Glossary.md
)
for f in "${MD_FILES[@]}"; do [ -f "$f" ] || { echo "缺文件: $f" >&2; exit 1; }; done

# ── 1. 主版 epub ────────────────────────────────────────────────────────
pandoc "${MD_FILES[@]}" -o "$OUT/ds-ocr-docs.epub" \
  --css tools/ebook-stylesheet.css \
  --toc-depth=1 \
  --metadata title="ds-ocr 学习文档" \
  --metadata subtitle="从入门到精通大模型推理（16 篇 + 术语表）" \
  --metadata author="vislee" \
  --metadata date="$(date +%F)" \
  --metadata lang="zh-CN" \
  --metadata rights="© vislee · 本电子书由仓库 docs/ 目录构建生成"

# ── 2. Kindle 版 epub：剥 emoji、light-dark() 降级、KF8 兼容 CSS ────────
# PW3 的 KF8 引擎不支持: light-dark() 颜色函数、table 的 display:block+横向
# 滚动、pre 的 overflow:auto（长代码行直接被裁掉）；内置字体也没有 emoji 字形。
mkdir "$TMP/src" && unzip -qo "$OUT/ds-ocr-docs.epub" -d "$TMP/src"
python3 - "$TMP/src" <<'PYEOF'
import glob, os, sys, re

src = sys.argv[1]
# 纯剥离: 1F000–1FFFF（emoji 主区）、FE0F 变体选择符、⚠（26A0，无安全等价）
STRIP = set([0xFE0F, 0x26A0]) | set(range(0x1F000, 0x20000))
# 替换为 GB2312 必有字形，零语义损失: ✅❌✓✔→√、✗❌→×、⭐→★
REPLACE = {0x2705: '√', 0x274C: '×', 0x2B50: '★', 0x2713: '√', 0x2714: '√', 0x2717: '×'}

def clean(s):
    return ''.join(REPLACE.get(ord(c), '' if ord(c) in STRIP else c) for c in s)

files = (glob.glob(f'{src}/EPUB/text/*.xhtml') + glob.glob(f'{src}/EPUB/*.xhtml') +
         glob.glob(f'{src}/EPUB/*.ncx') + glob.glob(f'{src}/EPUB/*.opf'))
for f in files:
    # 必须先读后写：open(f,'w') 一旦执行就会截断文件，边写边读会得到空内容
    s = open(f, encoding='utf-8').read()
    open(f, 'w', encoding='utf-8').write(clean(s))

css_file = f'{src}/EPUB/styles/stylesheet1.css'
css = open(css_file, encoding='utf-8').read()
# light-dark(L,D) 每处上方都已有浅色回退声明 → 整行删除即回退到浅色主题
css = '\n'.join(l for l in css.split('\n') if 'light-dark(' not in l)
css = css.replace('color-scheme: light dark;', '')
css = css.replace('display: block;', 'display: table;')   # 仅 table 规则命中
css = css.replace('.display.math {\n  display: table;', '.display.math {\n  display: block;')  # 数学块还原
css = css.replace('overflow-x: auto;', '')
css += '''
/* ===== Kindle Paperwhite 3 (KF8) 专用覆盖 ===== */
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
  overflow: visible;
  font-size: 70%;
}
pre code { overflow-wrap: break-word; word-wrap: break-word; }
table { display: table; table-layout: auto; width: 100%; font-size: 80%; }
th, td { word-break: break-word; overflow-wrap: break-word; }
'''
open(css_file, 'w', encoding='utf-8').write(css)
PYEOF
# 重打包: mimetype 必须第一个且不压缩（EPUB 规范）
( cd "$TMP/src" && rm -f "$TMP/ds-ocr-docs-kindle.epub" \
  && zip -q -X -0 "$TMP/ds-ocr-docs-kindle.epub" mimetype \
  && zip -q -X -9 -r "$TMP/ds-ocr-docs-kindle.epub" . -x mimetype )

# ── 3. 转 azw3 / mobi（PW3 输出配置 + 保留相对字号，禁用字体重缩放）──────
"$CALIBRE_EBOOK_CONVERT" "$TMP/ds-ocr-docs-kindle.epub" "$OUT/ds-ocr-docs.azw3" \
  --output-profile kindle_pw3 --disable-font-rescaling >/dev/null
"$CALIBRE_EBOOK_CONVERT" "$TMP/ds-ocr-docs-kindle.epub" "$OUT/ds-ocr-docs.mobi" \
  --output-profile kindle_pw3 --disable-font-rescaling --mobi-file-type new >/dev/null

ls -la "$OUT/ds-ocr-docs.epub" "$OUT/ds-ocr-docs.azw3" "$OUT/ds-ocr-docs.mobi"
