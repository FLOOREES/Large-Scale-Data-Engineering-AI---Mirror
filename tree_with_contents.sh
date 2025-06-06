#!/usr/bin/env bash
echo "FILE TREE:"
tree -I "P1|.venv|__pycache__"

echo ""
echo "PYTHON FILES (.py) CONTENT:"
find . -type f -name "*.py" \
  -not -path "./P1/*" \
  -not -path "./.venv/*" \
  -not -path "*/__pycache__/*" | while read file; do
    echo "====== FILE: $file ======"
    cat "$file"
    echo ""
done
