//! 目标特性、中端 pass 与写目标文件。
use std::fmt::Write as _;

use rustc_codegen_ssa::back::write::{CodegenContext, EmitObj, ModuleConfig, SharedEmitter};
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxt;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputType};
use rustc_span::Symbol;

use crate::RvModule;

/// riscvc 认得的单字母扩展，按 ISA 串的规范次序。
const LETTERS: [char; 5] = ['m', 'a', 'f', 'd', 'c'];

/// 打开的目标特性：目标规格的 `features`（`+m,+a,+c,+f`）再叠上 `-Ctarget-feature`（`-a,-c`），后写的算数。
pub struct Features(Vec<String>);

impl Features {
    pub fn of(sess: &Session) -> Self {
        let mut on: Vec<String> = Vec::new();
        for f in sess.target.features.split(',').chain(sess.opts.cg.target_feature.split(',')) {
            let (add, name) = match f.as_bytes().first() {
                Some(b'+') => (true, &f[1..]),
                Some(b'-') => (false, &f[1..]),
                _ => continue,
            };
            on.retain(|x| x != name);
            if add {
                on.push(String::from(name));
            }
        }
        Self(on)
    }

    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|x| x == name)
    }

    /// riscvc 的目标串（`feature::Target::parse` 的写法），如 `rv32imf_zicsr`。F / D 要 Zicsr（fcsr）。
    pub fn march(&self) -> String {
        let mut s = String::from("rv32i");
        s.extend(LETTERS.iter().filter(|c| self.has(c.encode_utf8(&mut [0; 4]))));
        if self.has("f") || self.has("d") || self.has("zicsr") {
            s.push_str("_zicsr");
        }
        s
    }

    /// 给 `cfg(target_feature = …)` 的特性名。
    pub fn symbols(&self) -> Vec<Symbol> {
        let mut v: Vec<Symbol> = self.0.iter().map(|s| Symbol::intern(s)).collect();
        if self.has("f") || self.has("d") {
            v.push(Symbol::intern("zicsr"));
        }
        v
    }
}

impl RvModule {
    fn target(&self) -> Result<riscvc::feature::Target<'static>, String> {
        riscvc::feature::Target::parse(&self.march)
    }
}

/// 中端：开了优化就跑 riscvc 缺省那串（mem2reg 把 alloca 提成 SSA 值、gvn、sccp、dce）。
pub fn optimize(shared_emitter: &SharedEmitter, module: &mut ModuleCodegen<RvModule>, config: &ModuleConfig) {
    if config.opt_level.is_none_or(|l| l == OptLevel::No) {
        return;
    }
    let reg = riscvc::plugin::Registry::default();
    if let Err(e) = reg.run(&mut module.module_llvm.module, &riscvc::opt::DEFAULT_PIPELINE) {
        let dcx = DiagCtxt::new(Box::new(shared_emitter.clone()));
        dcx.handle().fatal(format!("riscvc 中端（{}）：{e}", module.name));
    }
}

/// 代码生成 ⇒ 汇编文本 ⇒ 目标文件；`--emit=asm` 另写 `.s`，`--emit=llvm-ir` 写 riscvc IR 的文本（看问题用）。
pub fn codegen(cgcx: &CodegenContext, prof: &SelfProfilerRef, shared_emitter: &SharedEmitter, module: ModuleCodegen<RvModule>, config: &ModuleConfig) -> CompiledModule {
    let _timer = prof.generic_activity_with_arg("riscvc_module_codegen", &*module.name);
    let dcx = DiagCtxt::new(Box::new(shared_emitter.clone()));
    let dcx = dcx.handle();
    let out = &cgcx.output_filenames;
    let rv = &module.module_llvm;
    let write = |kind: OutputType, data: &[u8]| {
        let path = out.temp_path_for_cgu(kind, &module.name);
        if let Err(e) = std::fs::write(&path, data) {
            dcx.fatal(format!("写 {}：{e}", path.display()));
        }
    };
    if config.emit_ir {
        write(OutputType::LlvmAssembly, rv.dump().as_bytes());
    }
    let target = rv.target().unwrap_or_else(|e| dcx.fatal(format!("目标 {}：{e}", rv.march)));
    let asm = rv
        .module
        .compile_with(&riscvc::codegen::Options { target, ..Default::default() })
        .unwrap_or_else(|e| dcx.fatal(format!("riscvc 代码生成（{}）：{e}", module.name)));
    if config.emit_asm {
        write(OutputType::Assembly, asm.as_bytes());
    }
    if let EmitObj::ObjectCode(_) = config.emit_obj {
        let target = rv.target().unwrap_or_else(|e| dcx.fatal(e));
        let obj = riscvc::asm::Assembler::new()
            .options(riscvc::asm::Options { attributes: None, relax: true, ..Default::default() })
            .target(target)
            .assemble(&asm)
            .unwrap_or_else(|e| dcx.fatal(format!("riscvc 汇编（{}）：{e}", module.name)));
        write(OutputType::Object, &obj);
    }
    module.into_compiled_module(config.emit_obj != EmitObj::None, false, false, config.emit_asm, config.emit_ir, out)
}

impl RvModule {
    /// riscvc IR 的文本（`Debug` 形式，看问题用）。
    fn dump(&self) -> String {
        let mut s = String::new();
        for g in &self.module.globals {
            let _ = writeln!(s, "{g:?}");
        }
        for f in &self.module.functions {
            let _ = writeln!(s, "{f:#?}");
        }
        s
    }
}
