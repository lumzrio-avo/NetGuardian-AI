#!/usr/bin/env bash
# ============================================================
# smoke_all.sh —— 统一口径改版后的一次性冒烟测试
#
# 用法（在 WSL 里）:
#   bash smoke_all.sh <数据根目录> [python解释器]
#
# 例:
#   bash smoke_all.sh /home/lumzrio/RCAEval/data/RE1
#   bash smoke_all.sh /home/lumzrio/RCAEval/data/RE1 /home/lumzrio/RCAEval/env/bin/python
#
# 数据根目录下应有 RE1-OB / RE1-SS / RE1-TT 三个子目录
#   （即 <数据根>/RE1-OB/<service>_<fault>/<case>/data.csv）
# ============================================================
set -u

DATA="${1:-}"
PY="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${TMPDIR:-/tmp}/smoke_pccode_$(date +%H%M%S)"

if [ -z "$DATA" ]; then
  echo "用法: bash smoke_all.sh <数据根目录> [python解释器]"
  echo "  例: bash smoke_all.sh /home/lumzrio/RCAEval/data/RE1"
  echo "  例: bash smoke_all.sh /home/lumzrio/RCAEval/data/RE1 ~/RCAEval/env/bin/python"
  exit 1
fi

# ---------- 解释器解析：显式传入 > 自动探测 ----------
probe() {  # 返回 0 表示该解释器依赖齐全
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c "import numpy,pandas,networkx,tqdm,causallearn,tigramite" >/dev/null 2>&1
}

if [ -z "$PY" ]; then
  echo "[探测] 未指定解释器，正在自动查找依赖齐全的 python ..."
  for cand in \
      "${VIRTUAL_ENV:-/nonexistent}/bin/python" \
      "$HOME/RCAEval/env/bin/python" \
      "$HOME/.venv/bin/python" \
      "$HERE/venv/bin/python" \
      "$HERE/.venv/bin/python" \
      "$(command -v python3 2>/dev/null)" \
      "$(command -v python 2>/dev/null)" ; do
    if probe "$cand"; then PY="$cand"; echo "[探测] ✅ 选用: $PY"; break; fi
    [ -n "$cand" ] && [ "$cand" != "/nonexistent/bin/python" ] && echo "       ✗ $cand"
  done
fi

if [ -z "$PY" ] || ! probe "$PY"; then
  echo
  echo "!! 没找到依赖齐全的 python。请手动指定解释器:"
  echo "     bash smoke_all.sh <数据根> ~/RCAEval/env/bin/python"
  echo
  echo "   先看看有哪些环境可用:"
  for cand in "$HOME/RCAEval/env/bin/python" "$HOME/.venv/bin/python" \
              "$HERE/venv/bin/python" /usr/bin/python3; do
    [ -e "$cand" ] || continue
    printf "     %-45s " "$cand"
    if "$cand" -c "import numpy,pandas,networkx,tqdm,causallearn,tigramite" >/dev/null 2>&1; then
      echo "✅ 依赖齐全"
    else
      echo "✗ 缺包 -- $("$cand" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
    fi
  done
  echo
  echo "   装包（官方 RCAEval 指定 causal-learn==0.1.3.3）:"
  echo "     <解释器> -m pip install causal-learn==0.1.3.3 tigramite networkx tqdm"
  echo
  echo "   ⚠️ 注意: 直接敲 \$PY 是没用的 —— 那个变量只在脚本内部存在。"
  exit 2
fi

echo "============================================================"
echo "数据根 : $DATA"
echo "解释器 : $PY"
echo "输出   : $OUT"
echo "============================================================"

# ---------- 0) 环境自检 ----------
echo
echo "[0] 依赖检查"
$PY - <<'PYEOF'
import importlib, sys
need = ['numpy', 'pandas', 'networkx', 'tqdm', 'causallearn', 'tigramite']
miss = []
for m in need:
    try:
        importlib.import_module(m)
        print('  OK  ', m)
    except Exception as e:
        print('  MISS', m)
        miss.append(m)
if miss:
    print('\n缺包，安装（官方 RCAEval 指定 causal-learn==0.1.3.3）:')
    print('  %s -m pip install causal-learn==0.1.3.3 tigramite networkx tqdm' % sys.executable)
    sys.exit(2)
PYEOF
if [ $? -ne 0 ]; then echo "!! 依赖不全，先装包"; exit 2; fi

# ---------- 1) 逐个冒烟 ----------
run() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "[$name] $*"
  echo "============================================================"
  ( cd "$HERE" && $PY "$@" ) 2>&1 | tail -25
}

run "PC · RE1-OB"   run_rca_full_ob.py --algo pc    --data-root "$DATA/RE1-OB" --output "$OUT/ob-pc"    --test
run "PC · RE1-SS"   run_pc_enhanced_ss.py           --data-root "$DATA/RE1-SS" --output "$OUT/ss-pc"    --test
run "PC · RE1-TT"   run_pc_tt.py                    --data-root "$DATA/RE1-TT" --output "$OUT/tt-pc"    --test
run "PCMCI · RE1-OB" run_rca_full_ob.py --algo pcmci --data-root "$DATA/RE1-OB" --output "$OUT/ob-pcmci" --test
run "PCMCI · RE1-SS" run_pcmci.py                   --data-root "$DATA/RE1-SS" --output "$OUT/ss-pcmci" --test
run "PCMCI · RE1-TT" run_pcmci.py                   --data-root "$DATA/RE1-TT" --output "$OUT/tt-pcmci" --test

# ---------- 2) 核对日志关键标记 ----------
echo
echo "============================================================"
echo "核对：窗口 / alpha / 变量数 / 是否退化成全零图"
echo "============================================================"

check_dir() {
  local label="$1" dir="$2"
  echo
  echo "---- $label ($dir) ----"
  if [ ! -d "$dir" ]; then echo "  !! 输出目录不存在（该配置可能启动失败）"; return; fi

  # 窗口
  local w full normal
  full=$(grep -rh "使用全窗口数据" "$dir" 2>/dev/null | tail -1)
  normal=$(grep -rh "使用正常窗口数据" "$dir" 2>/dev/null | tail -1)
  echo "  窗口        : ${full:-${normal:-（未打印）}}"

  # alpha（PC 的日志里有 "运行 PC 算法: alpha=..."；PCMCI 看启动横幅）
  echo "  alpha/参数  : $(grep -rhoE "alpha=[0-9.]+" "$dir" 2>/dev/null | sort -u | tr '\n' ' ')"

  # 变量数
  echo "  变量数      : $(grep -rhoE "变量数=[0-9]+|有效变量数: [0-9]+|统一筛选后: [0-9]+" "$dir" 2>/dev/null | tail -1)"

  # 边数 / 全零图
  echo "  边数        : $(grep -rhoE "edges=[0-9]+|边=[0-9]+" "$dir" 2>/dev/null | sort -u | tr '\n' ' ')"
  if grep -rqE "edges=0\b|边=0\b" "$dir" 2>/dev/null; then
    echo "  ⚠️ 警告      : 出现 edges=0（全零图），需要排查"
  else
    echo "  ✅ 未见全零图"
  fi

  # 成功/失败 + 报错（最关键 —— 有 ERROR 就是跑挂了）
  local ok_cnt fail_cnt
  ok_cnt=$(grep -rhoE "成功:\s*[0-9]+" "$dir" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  fail_cnt=$(grep -rhoE "失败:\s*[0-9]+" "$dir" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  echo "  成功/失败   : ${ok_cnt:-?} / ${fail_cnt:-?}"
  if [ -n "${fail_cnt:-}" ] && [ "$fail_cnt" != "0" ]; then
    echo "  ❌ 有失败案例！错误摘录:"
    grep -rh "ERROR\|NameError\|Traceback" "$dir"/.. 2>/dev/null | head -5 | sed 's/^/       /'
    grep -rhoE "(NameError|TypeError|ValueError|KeyError|AttributeError|ImportError)[^\n]*" \
      "$dir"/.. 2>/dev/null | sort -u | head -5 | sed 's/^/       /'
  fi
}

check_dir "PC · RE1-OB"    "$OUT/ob-pc"
check_dir "PC · RE1-SS"    "$OUT/ss-pc"
check_dir "PC · RE1-TT"    "$OUT/tt-pc"
check_dir "PCMCI · RE1-OB" "$OUT/ob-pcmci"
check_dir "PCMCI · RE1-SS" "$OUT/ss-pcmci"
check_dir "PCMCI · RE1-TT" "$OUT/tt-pcmci"

echo
echo "============================================================"
echo "期望值速查"
echo "  窗口  : 全部应为「使用全窗口数据」"
echo "  alpha : PC 一律 0.05；PCMCI 一律 0.2"
echo "  变量数: 同一套筛选规则，OB ~44-47 / SS ~40 / TT ~40 量级"
echo "  全零图: 不应出现"
echo "============================================================"
echo "冒烟输出目录: $OUT"
