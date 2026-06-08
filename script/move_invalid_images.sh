#!/bin/bash
# 将 data/train 中文件名以 "." 结尾的文件移动到 data/invalidImg/
# 这些文件在 Windows 上无法通过标准 API 访问（Win32 会截去末尾的点），
# 导致 DataLoader 报 FileNotFoundError

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
TRAIN_DIR="$DATA_DIR/train"
INVALID_DIR="$DATA_DIR/invalidImg"

mkdir -p "$INVALID_DIR"

echo "正在扫描 $TRAIN_DIR 中以 '.' 结尾的文件..."

count=0
while IFS= read -r -d '' file; do
    basename=$(basename "$file")
    dest="$INVALID_DIR/$basename"

    # 如果目标已存在，追加编号避免覆盖
    if [ -e "$dest" ]; then
        base="${basename%.}"
        i=1
        while [ -e "$INVALID_DIR/${base}_${i}" ]; do
            i=$((i + 1))
        done
        dest="$INVALID_DIR/${base}_${i}"
    fi

    mv "$file" "$dest"
    echo "已移动: $file -> $dest"
    count=$((count + 1))
done < <(find "$TRAIN_DIR" -type f -name "*." -print0)

echo ""
echo "完成！共移动 $count 个文件到 $INVALID_DIR"
