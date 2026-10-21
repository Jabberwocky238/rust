#!/usr/bin/env bash
# 端到端：riscvc 仓库的 selfcheck（core、compiler_builtins、运行时 rvrt 全由本后端编）⇒ riscvc as / ld ⇒ 模拟器跑到 ebreak。
# a0 = 失败的检查个数，0 才过（判据同 riscvc 的 tests/rust_e2e.rs，那边走的是 rustc 出 LLVM IR 的老路）。
# 用法：[MARCH=…] tools/selfcheck.sh [O0|O2]（缺省 O2）
#   MARCH（缺省 rv32imf_zicsr）：rv32imaf_zicsr 开 A（原子操作那组检查才编）；带 d（rv32imafd_zicsr）⇒ 照内置目标写一份 +d、ilp32d 的 target JSON
# 环境：RISCVC_SRC（riscvc 源码，要它的 tests/rust；缺省与 rust 检出同级的 riscvc）、RISCVC（riscvc 命令；缺省 PATH 里的，没有就 cargo install）、KEEP=1 留中间产物
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${RISCVC_SRC:-$HERE/../../../riscvc}
LV=${1:-O2}
MARCH=${MARCH:-rv32imf_zicsr}
TARGET=riscv32imafc-unknown-none-elf
RISCVC=${RISCVC:-$(command -v riscvc || true)}
[ -n "$RISCVC" ] || { cargo +1.98.1 install -q riscvc --version =0.2.5; RISCVC=$(command -v riscvc); }
(cd "$HERE" && cargo build -q --release)
(cd "$HERE/tools/run-elf" && cargo build -q --release)
BE=$HERE/target/release/librustc_codegen_riscvc.so
WORK=$(mktemp -d)
trap "[ -n \"${KEEP:-}\" ] && echo \"中间产物：$WORK\" || rm -rf $WORK" EXIT
prof=$([ "$LV" = O2 ] && echo release || echo debug)
# 内置目标 riscv32imafc 里 MARCH 没有的（a、c）关掉
letters=$(echo "${MARCH#rv32}" | cut -d_ -f1)
OFF=$(for e in a c; do case $letters in *$e*) ;; *) printf -- "-%s," $e ;; esac; done)
SPEC=$TARGET
case $letters in
  *d*)
    # rustc 没有带 D 的 riscv32 内置目标：照 riscv32imafc 的规格加 +d、换 ilp32d（名字要带 -none：cargo 据此给 compiler_builtins 开 mem）
    TARGET=riscv32imafdc-unknown-none-elf
    SPEC=$WORK/$TARGET.json
    RUSTC_BOOTSTRAP=1 rustc +1.98.1 -Z unstable-options --print target-spec-json --target riscv32imafc-unknown-none-elf \
      | sed -e 's/"features": "+m,+a,+c,+f"/"features": "+m,+a,+c,+f,+d"/' -e 's/"llvm-abiname": "ilp32f"/"llvm-abiname": "ilp32d"/' > "$SPEC"
    grep -q '+d"' "$SPEC" && grep -q ilp32d "$SPEC" || { echo "改不出带 D 的 target JSON" >&2; exit 1; }
    ;;
esac
build() {
  RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zcodegen-backend=$BE -Ctarget-feature=${OFF%,} -Ccodegen-units=1" \
    cargo +1.98.1 build -q --manifest-path "$1" --target-dir "$2" --target "$SPEC" \
      $([ "$prof" = release ] && echo --release) -Zbuild-std=core $([ "$SPEC" = "$TARGET" ] || echo -Zjson-target-spec)
}
build "$SRC/tests/rust/examples/selfcheck/Cargo.toml" "$WORK/t"
build "$SRC/tests/rust/rt/Cargo.toml" "$WORK/rt"
mkdir -p "$WORK/o"
# staticlib 里已经有 core / compiler_builtins / 程序本身的 .o；运行时只要它自己那份 rlib 里的 .o
(cd "$WORK/o" && ar x "$WORK/t/$TARGET/$prof/libselfcheck.a" && ar x "$(ls "$WORK/rt/$TARGET/$prof/deps/"librvrt-*.rlib)" && rm -f lib.rmeta)
cat > "$WORK/crt.s" <<CRT
    .section .text.start, "ax"
    .globl _start
_start:
    lui sp, %hi(__stack_top)
    addi sp, sp, %lo(__stack_top)
    call check
    ebreak
CRT
"$RISCVC" as --march $MARCH "$WORK/crt.s" -o "$WORK/crt.o"
cat > "$WORK/link.ld" <<'LD'
ENTRY(_start)
SECTIONS {
  . = 0x10000;
  .text : { *(.text.start) *(.text*) }
  .rodata : { *(.rodata*) *(.srodata*) }
  .data : { *(.data*) *(.sdata*) }
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
  . = ALIGN(16);
  . = . + 0x10000;
  __stack_top = .;
}
LD
"$RISCVC" ld -T "$WORK/link.ld" "$WORK/crt.o" "$WORK/o/"*.o -o "$WORK/app.elf"
"$HERE/tools/run-elf/target/release/run-elf" "$WORK/app.elf" $MARCH
