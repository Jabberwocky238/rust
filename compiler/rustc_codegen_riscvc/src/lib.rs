//! rustc 的代码生成后端：MIR ⇒（rustc_codegen_ssa）⇒ riscvc 的内部 IR ⇒ RV32 汇编 ⇒ 目标文件。⛔ 不经 LLVM。
//!
//! ```text
//! context   每个 CGU 一份的上下文：riscvc `ir::Module` + 实例 / 静态量 / 常量的表；类型、常量、声明
//! builder   rustc_codegen_ssa 的 Builder：一条 MIR 语句 ⇒ 若干 `ir::Op`
//! abi       FnAbi ⇒ 函数类型与形参属性
//! intrinsic 后端自己落的 intrinsic（其余交回 rustc 的回退实现）
//! write     中端 pass、代码生成、汇编 ⇒ `.o`
//! ```
//!
//! 用法：`rustc -Zcodegen-backend=<本 dylib> --target riscv32imafc-unknown-none-elf -Ctarget-feature=-a,-c …`
#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;

// 链上 librustc_driver（rustc 的其余 crate 都在里面）
extern crate rustc_driver;

mod abi;
mod builder;
mod context;
mod intrinsic;
mod write;

use std::any::Any;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_codegen_ssa::back::lto::ThinModule;
use rustc_codegen_ssa::back::write::{CodegenContext, FatLtoInput, ModuleConfig, OngoingCodegen, SharedEmitter, TargetMachineFactoryFn, ThinLtoInput};
use rustc_codegen_ssa::traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods};
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleCodegen, TargetConfig};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph::{WorkProduct, WorkProductMap};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames};
use rustc_span::{Symbol, sym};

/// 后端本体。`march` 在 `init` 时按目标与 `-Ctarget-feature` 定下，之后每个 CGU 都带一份。
#[derive(Clone, Default)]
pub struct RiscvcBackend {
    march: Arc<OnceLock<String>>,
}

/// 一个 CGU 的产物：riscvc 的模块 + 它要按哪个目标编。
pub struct RvModule {
    pub module: riscvc::ir::Module,
    pub march: String,
}

impl RiscvcBackend {
    fn march(&self) -> String {
        self.march.get().cloned().expect("init 之前没有目标")
    }
}

impl CodegenBackend for RiscvcBackend {
    fn name(&self) -> &'static str {
        "riscvc"
    }

    fn init(&self, sess: &Session) {
        self.march.get_or_init(|| write::Features::of(sess).march());
    }

    fn target_config(&self, sess: &Session) -> TargetConfig {
        let features = write::Features::of(sess).symbols();
        // ⛔ f16 / f128：riscvc 只有软浮点库调用的 f32 / f64，报「不可靠」让 core 不生成依赖它们的代码
        TargetConfig {
            target_features: features.clone(),
            unstable_target_features: features,
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn thin_lto_supported(&self) -> bool {
        false
    }

    fn fallback_intrinsics(&self) -> Vec<Symbol> {
        vec![sym::type_id_eq]
    }

    fn target_cpu(&self, sess: &Session) -> String {
        sess.opts.cg.target_cpu.clone().unwrap_or_else(|| sess.target.cpu.to_string())
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(self.clone(), tcx))
    }

    fn join_codegen(&self, ongoing: Box<dyn Any>, sess: &Session, _outputs: &OutputFilenames, crate_info: &CrateInfo) -> (CompiledModules, WorkProductMap) {
        ongoing.downcast::<OngoingCodegen<RiscvcBackend>>().expect("不是 riscvc 后端的 OngoingCodegen").join(sess, crate_info)
    }
}

impl ExtraBackendMethods for RiscvcBackend {
    type Module = RvModule;

    fn codegen_allocator(&self, tcx: TyCtxt<'_>, _module_name: &str, methods: &[AllocatorMethod]) -> RvModule {
        RvModule { module: context::CodegenCx::allocator_shim(tcx, methods), march: self.march() }
    }

    fn compile_codegen_unit(&self, tcx: TyCtxt<'_>, cgu_name: Symbol) -> (ModuleCodegen<RvModule>, u64) {
        context::CodegenCx::compile_cgu(tcx, cgu_name, self.march())
    }
}

impl WriteBackendMethods for RiscvcBackend {
    type Module = RvModule;
    type TargetMachine = ();
    type ModuleBuffer = std::convert::Infallible;
    type ThinData = ();

    fn target_machine_factory(&self, _sess: &Session, _opt_level: OptLevel, _features: &[String]) -> TargetMachineFactoryFn<Self> {
        Arc::new(|_, _| ())
    }

    fn optimize_and_codegen_fat_lto(
        sess: &Session,
        _cgcx: &CodegenContext,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule {
        sess.dcx().fatal("riscvc 后端不支持 LTO")
    }

    fn run_thin_lto(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        dcx: DiagCtxtHandle<'_>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<ThinLtoInput<Self>>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        dcx.fatal("riscvc 后端不支持 thin LTO")
    }

    fn optimize(_cgcx: &CodegenContext, _prof: &SelfProfilerRef, shared_emitter: &SharedEmitter, module: &mut ModuleCodegen<RvModule>, config: &ModuleConfig) {
        write::optimize(shared_emitter, module, config);
    }

    fn optimize_and_codegen_thin(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        _thin: ThinModule<Self>,
    ) -> CompiledModule {
        unreachable!("thin LTO 已在 run_thin_lto 拒掉")
    }

    fn codegen(cgcx: &CodegenContext, prof: &SelfProfilerRef, shared_emitter: &SharedEmitter, module: ModuleCodegen<RvModule>, config: &ModuleConfig) -> CompiledModule {
        write::codegen(cgcx, prof, shared_emitter, module, config)
    }

    fn serialize_module(_module: RvModule, _is_thin: bool) -> Self::ModuleBuffer {
        unreachable!("不支持 LTO，⛔ 序列化模块")
    }
}

/// rustc 按 `-Zcodegen-backend=<路径>` 加载本 dylib 后调用的入口。
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(RiscvcBackend::default())
}
