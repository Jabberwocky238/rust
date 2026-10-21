# rustc_codegen_riscvc

rustc 的代码生成后端：MIR ⇒ rustc_codegen_ssa ⇒ [riscvc](https://crates.io/crates/riscvc) 的内部 IR ⇒ RV32 汇编 ⇒ 目标文件。⛔ 不经 LLVM。

## 构建

后端 dylib 与 rustc 的内部 API 绑死，只能用 **rustc 1.98.1**（本分支的基线）编、也只能被它加载。`rust-toolchain.toml` 已钉版本与组件
（`rustc-dev`、`rust-src`），`.cargo/config.toml` 用 `RUSTC_BOOTSTRAP=1` 打开 `rustc_private`。

```bash
cargo build --release            # ⇒ target/release/librustc_codegen_riscvc.so
```

## 使用

```bash
RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zcodegen-backend=$PWD/target/release/librustc_codegen_riscvc.so -Ctarget-feature=-a,-c" \
  cargo +1.98.1 build --target riscv32imafc-unknown-none-elf -Zbuild-std=core --release
```

- 目标：`riscv32imafc-unknown-none-elf` 叠上 `-Ctarget-feature`，得出 riscvc 的目标串（缺省规格 `+m,+a,+c,+f`，`-a,-c` ⇒ `rv32imf_zicsr`）。
- 产物：每个 CGU 一个 `.o`（riscvc 汇编器出的 ELF32）；rlib / staticlib 由 rustc 照常打包。可执行文件的链接仍走 rustc 的缺省链接器，
  裸机程序建议出 staticlib、再用 `riscvc ld` 链。
- `--emit=asm` 出 riscvc 生成的汇编；`--emit=llvm-ir` 出 riscvc IR 的文本（`Debug` 形式，看问题用）。
- 开了优化（`-Copt-level` ≥ 1）才跑 riscvc 的中端（mem2reg、gvn、sccp、dce）。

## 范围

| 支持 | 不支持 |
| --- | --- |
| 标量、标量对、内存形态的布局；FnAbi 的 Direct / Pair / Cast / Indirect（sret、byval） | SIMD 向量、可伸缩向量 |
| 原子操作（有 A 出 amo / LR-SC，没有 A 由 riscvc 调 `__atomic_*`） | 展开（`panic = unwind`）：只支持 `panic = abort` |
| 位运算、饱和、溢出、f32 / f64 的数学 intrinsic；其余有回退实现的交回 rustc | 内联汇编（只认空模板）、`global_asm!`、naked 函数 |
| 分配器 shim | LTO、调试信息、`va_arg`、f16 / f128 |

## 验证

```bash
tools/selfcheck.sh O2                          # 也可 O0；缺省 MARCH=rv32imf_zicsr
MARCH=rv32imaf_zicsr tools/selfcheck.sh O2     # 开 A：原子操作那组检查才编
MARCH=rv32imafd_zicsr tools/selfcheck.sh O2    # 开 D：+d、ilp32d 的 target JSON（脚本现写）
```

riscvc 仓库的 `tests/rust/examples/selfcheck`（自检程序）连同 core、compiler_builtins、运行时 `rvrt` 全由本后端编，
`riscvc ld` 链接后在 riscvc 的指令级模拟器里从 `_start` 跑到 `ebreak`：a0 = 失败的检查个数，0 才过。
要 riscvc 源码（`RISCVC_SRC`，缺省与 rust 检出同级的 `../riscvc`）与 riscvc 命令（没有就 `cargo install riscvc --version =0.2.5`）。
