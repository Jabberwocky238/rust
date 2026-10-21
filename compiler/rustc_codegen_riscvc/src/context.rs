//! 每个 CGU 一份的代码生成上下文：riscvc 的 `ir::Module`，加上 rustc 的实例 / 静态量 / 常量到它里面的下标的表。
//!
//! ⭐ 值（[`Val`]）分两种：函数里的 SSA 值，和驻留在上下文里的常量 / 符号地址。后者在 rustc 眼里是「常量」，
//! 可以跨函数、跨块用，也能当全局的初值；落到某个函数里时才由 builder 在当前块出一条 `Op::Const`。
use std::cell::RefCell;
use std::time::Instant;

use riscvc::ir::{ArgAttrs, BlockId, Block, Const, Function, Global, Inst, Kind, Linkage as RvLinkage, Module, Op, Param, Target, Ty, Types, ValId};
use rustc_abi::{self as abi, AddressSpace, Align, BackendRepr, Float, HasDataLayout, Primitive, Size, TargetDataLayout, WrappingRange};
use rustc_ast::expand::allocator::{AllocatorMethod, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE, default_fn_name, global_fn_name};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::attrs::Linkage;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::dep_graph;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::{Allocation, ConstAllocation, GlobalAlloc, Pointer, Scalar as InterpScalar, read_target_uint};
use rustc_middle::mono::{CodegenUnit, Visibility};
use rustc_middle::ty::layout::{FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{self, ExistentialTraitRef, Instance, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_abi::{Reg, RegKind};
use rustc_target::callconv::{CastTarget, FnAbi};
use rustc_target::spec::{HasTargetSpec, HasX86AbiOpt, X86Abi};

use crate::RvModule;
use crate::builder::Builder;

/// 函数下标（`Module::functions`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fid(pub u32);

/// 后端的值。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Val {
    /// 某个函数里的 SSA 值。
    Local(Fid, ValId),
    /// 驻留常量（[`CodegenCx::cst`]）。
    Const(u32),
    /// 函数的地址。
    Func(Fid),
    /// 全局变量的地址（`Module::globals` 下标）。
    Global(u32),
}

/// 基本块：所在函数 + 块下标。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blk {
    pub f: Fid,
    pub b: BlockId,
}

/// 驻留常量表：`(类型, 常量)` ⇒ 下标。
#[derive(Default)]
struct Consts {
    list: Vec<(Ty, Const)>,
    map: FxHashMap<(Ty, Const), u32>,
}

pub struct CodegenCx<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub cgu: &'tcx CodegenUnit<'tcx>,
    pub module: RefCell<Module>,
    consts: RefCell<Consts>,
    /// 符号名 ⇒ 函数 / 全局（同名只声明一次）。
    fns: RefCell<FxHashMap<String, Fid>>,
    globals: RefCell<FxHashMap<String, u32>>,
    instances: RefCell<FxHashMap<Instance<'tcx>, Fid>>,
    statics: RefCell<FxHashMap<DefId, u32>>,
    /// 匿名常量全局：（初值常量, 可写）⇒ 全局（`static_addr_of` 去重）。
    anon: RefCell<FxHashMap<(u32, bool), u32>>,
    vtables: RefCell<FxHashMap<(ty::Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), Val>>,
    /// 各函数入口块开头要插的指令（alloca 与它的计数常量）：收尾时一次插进去，mem2reg 只提升入口块里的 alloca。
    pub prelude: RefCell<FxHashMap<Fid, Vec<Inst>>>,
}

impl<'tcx> CodegenCx<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, cgu: &'tcx CodegenUnit<'tcx>) -> Self {
        Self {
            tcx,
            cgu,
            module: RefCell::default(),
            consts: RefCell::default(),
            fns: RefCell::default(),
            globals: RefCell::default(),
            instances: RefCell::default(),
            statics: RefCell::default(),
            anon: RefCell::default(),
            vtables: RefCell::default(),
            prelude: RefCell::default(),
        }
    }

    /// 编一个 CGU（同 cg_llvm：挂在依赖图上，增量编译据此判断要不要重编）。回模块与耗时（纳秒，给调度器配平）。
    pub fn compile_cgu(tcx: TyCtxt<'tcx>, cgu_name: Symbol, march: String) -> (ModuleCodegen<RvModule>, u64) {
        let start = Instant::now();
        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        let (module, _) = tcx.dep_graph.with_task(dep_node, tcx, || Self::cgu_module(tcx, cgu_name, march), Some(dep_graph::hash_result));
        (module, u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }

    fn cgu_module(tcx: TyCtxt<'tcx>, cgu_name: Symbol, march: String) -> ModuleCodegen<RvModule> {
        let cgu = tcx.codegen_unit(cgu_name);
        let mut cx = CodegenCx::new(tcx, cgu);
        let items = cgu.items_in_deterministic_order(tcx);
        for &(item, data) in &items {
            item.predefine::<Builder<'_, '_>>(&mut cx, cgu_name.as_str(), data.linkage, data.visibility);
        }
        for &(item, data) in &items {
            item.define::<Builder<'_, '_>>(&mut cx, cgu_name.as_str(), data);
        }
        maybe_create_entry_wrapper::<Builder<'_, '_>>(&cx, cx.cgu);
        ModuleCodegen::new_regular(cgu_name.to_string(), RvModule { module: cx.finish(), march })
    }

    /// 收尾：各函数的前导插到入口块开头。
    fn finish(self) -> Module {
        let mut m = self.module.into_inner();
        for (f, insts) in self.prelude.into_inner() {
            if let Some(entry) = m.functions[f.0 as usize].blocks.first_mut() {
                entry.insts.splice(0..0, insts);
            }
        }
        m
    }

    // ---- 类型 ----

    pub fn ty(&self, k: Kind) -> Ty {
        self.module.borrow_mut().types.get(k)
    }

    pub fn kind(&self, t: Ty) -> Kind {
        self.module.borrow().types.kind(t).clone()
    }

    /// 函数类型拆开：（返回, 形参, 可变参数）。
    pub fn sig(&self, fty: Ty) -> (Ty, Vec<Ty>, bool) {
        match self.kind(fty) {
            Kind::Func(r, p, v) => (r, p, v),
            k => bug!("riscvc：{k:?} 不是函数类型"),
        }
    }

    pub fn scalar_ty(&self, s: abi::Scalar) -> Ty {
        match s.primitive() {
            Primitive::Int(i, _) => self.ty(Kind::Int(u32::try_from(i.size().bits()).expect("整数位宽"))),
            Primitive::Float(Float::F16) => self.ty(Kind::Half),
            Primitive::Float(Float::F32) => Types::FLOAT,
            Primitive::Float(Float::F64) => Types::DOUBLE,
            Primitive::Float(Float::F128) => self.ty(Kind::Fp128),
            Primitive::Pointer(_) => Types::PTR,
        }
    }

    /// 字节数组 `[n x i8]`：内存形态的布局一律用它（字段投影走字节偏移，⛔ 要结构体类型）。
    pub fn bytes_ty(&self, n: u64) -> Ty {
        self.ty(Kind::Array(n, Types::I8))
    }

    // ---- 常量 ----

    /// 驻留一个常量。
    pub fn cst(&self, ty: Ty, c: Const) -> Val {
        let mut t = self.consts.borrow_mut();
        if let Some(&i) = t.map.get(&(ty, c.clone())) {
            return Val::Const(i);
        }
        let i = u32::try_from(t.list.len()).expect("常量太多");
        t.list.push((ty, c.clone()));
        t.map.insert((ty, c), i);
        Val::Const(i)
    }

    /// 整数常量，⭐ 按位宽截断后再驻留：riscvc 的指令选择直接用 `Const::Int` 的原值（i1 的「全 1」不截就成了 `not`）。
    pub fn int(&self, ty: Ty, v: u128) -> Val {
        let v = match self.module.borrow().types.bits(ty) {
            Some(bits) if bits < 128 => v & ((1u128 << bits) - 1),
            _ => v,
        };
        self.cst(ty, Const::Int(v))
    }

    /// 常量形态的值 ⇒（类型, 常量）；符号地址 ⇒ `Const::Global(名字, 0)`。⛔ 函数里的 SSA 值。
    pub fn konst(&self, v: Val) -> (Ty, Const) {
        match v {
            Val::Const(i) => self.consts.borrow().list[i as usize].clone(),
            Val::Func(f) => (Types::PTR, Const::Global(self.module.borrow().functions[f.0 as usize].name.clone(), 0)),
            Val::Global(g) => (Types::PTR, Const::Global(self.module.borrow().globals[g as usize].name.clone(), 0)),
            Val::Local(..) => bug!("riscvc：要常量，拿到的是 SSA 值 {v:?}"),
        }
    }

    // ---- 声明 ----

    /// 按名字声明函数（已有就回已有的）。`attrs` 与 `fty` 的形参一一对应。
    pub fn declare(&self, name: &str, fty: Ty, ret_attrs: ArgAttrs, attrs: Vec<ArgAttrs>) -> Fid {
        if let Some(&f) = self.fns.borrow().get(name) {
            return f;
        }
        let (ret, params, varargs) = self.sig(fty);
        let mut attrs = attrs;
        attrs.resize(params.len(), ArgAttrs::default());
        let mut m = self.module.borrow_mut();
        let f = Fid(u32::try_from(m.functions.len()).expect("函数太多"));
        m.functions.push(Function {
            name: String::from(name),
            linkage: RvLinkage::External,
            hidden: false,
            ret,
            ret_attrs,
            params: params.into_iter().zip(attrs).map(|(ty, attrs)| Param { ty, attrs }).collect(),
            varargs,
            section: None,
            align: None,
            stack_probe: None,
            blocks: Vec::new(),
            vals: Vec::new(),
        });
        self.fns.borrow_mut().insert(String::from(name), f);
        f
    }

    pub fn declare_fn(&self, name: &str, fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>) -> Fid {
        let (fty, ret, attrs) = self.fn_sig(fn_abi);
        self.declare(name, fty, ret, attrs)
    }

    /// 按名字定义 / 声明全局（已有就回已有的）。
    pub fn global(&self, name: &str, ty: Ty) -> u32 {
        if let Some(&g) = self.globals.borrow().get(name) {
            return g;
        }
        let mut m = self.module.borrow_mut();
        let g = u32::try_from(m.globals.len()).expect("全局太多");
        m.globals.push(Global {
            name: String::from(name),
            linkage: RvLinkage::External,
            hidden: false,
            ty,
            init: None,
            constant: false,
            align: None,
            section: None,
            thread_local: false,
        });
        self.globals.borrow_mut().insert(String::from(name), g);
        g
    }

    /// 匿名全局（私有，名字 `.Lanon.N` 不进符号表），按（初值, 可写）去重。
    pub fn anon(&self, init: Val, align: Align, mutable: bool) -> Val {
        // 只装一个指针的内存，初值是符号地址（Func / Global）：先驻留成常量再查表
        let init = match init {
            Val::Local(..) => bug!("riscvc：匿名全局的初值不是常量"),
            Val::Const(_) => init,
            v => {
                let (ty, c) = self.konst(v);
                self.cst(ty, c)
            }
        };
        let Val::Const(id) = init else { unreachable!() };
        if let Some(&g) = self.anon.borrow().get(&(id, mutable)) {
            return Val::Global(g);
        }
        let (ty, c) = self.konst(init);
        let name = format!(".Lanon.{}", self.module.borrow().globals.len());
        let g = self.global(&name, ty);
        {
            let mut m = self.module.borrow_mut();
            let x = &mut m.globals[g as usize];
            (x.linkage, x.init, x.constant, x.align) = (RvLinkage::Private, Some(c), !mutable, Some(align.bytes()));
        }
        self.anon.borrow_mut().insert((id, mutable), g);
        Val::Global(g)
    }

    /// 编译期内存 ⇒ 常量：字节段（`[n x i8]`）与指针（重定位）交替，多段时拼成紧排结构体。
    pub fn const_alloc(&self, alloc: &Allocation) -> Val {
        let dl = self.data_layout();
        let psize = usize::try_from(dl.pointer_size().bytes()).expect("指针宽度");
        let mut parts: Vec<Val> = Vec::new();
        let bytes = |parts: &mut Vec<Val>, r: std::ops::Range<usize>| {
            if !r.is_empty() {
                let b = alloc.inspect_with_uninit_and_ptr_outside_interpreter(r);
                parts.push(self.cst(self.bytes_ty(b.len() as u64), Const::Bytes(b.to_vec())));
            }
        };
        let mut next = 0;
        for &(offset, prov) in alloc.provenance().ptrs().iter() {
            let offset = usize::try_from(offset.bytes()).expect("偏移");
            bytes(&mut parts, next..offset);
            let raw = read_target_uint(dl.endian, alloc.inspect_with_uninit_and_ptr_outside_interpreter(offset..offset + psize)).expect("读重定位指针");
            let space = self.tcx.global_alloc(prov.alloc_id()).address_space(self);
            let scalar = abi::Scalar::Initialized { value: Primitive::Pointer(space), valid_range: WrappingRange::full(dl.pointer_size()) };
            let ptr = InterpScalar::from_pointer(Pointer::new(prov, Size::from_bytes(u64::try_from(raw).expect("指针值"))), &self.tcx);
            parts.push(self.scalar_to_backend(ptr, scalar, Types::PTR));
            next = offset + psize;
        }
        bytes(&mut parts, next..alloc.len());
        match parts[..] {
            [] => self.cst(self.bytes_ty(0), Const::Bytes(Vec::new())),
            [one] => one,
            _ => self.const_struct(&parts, true),
        }
    }

    pub fn get_static_global(&self, def_id: DefId) -> u32 {
        if let Some(&g) = self.statics.borrow().get(&def_id) {
            return g;
        }
        let instance = Instance::mono(self.tcx, def_id);
        let ty = self.layout_of(instance.ty(self.tcx, self.typing_env()));
        let name = self.tcx.symbol_name(instance).name;
        let g = self.global(name, self.bytes_ty(ty.size.bytes()));
        self.module.borrow_mut().globals[g as usize].thread_local = self.tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
        self.statics.borrow_mut().insert(def_id, g);
        g
    }

    /// 分配器 shim：`__rust_alloc` 等转发到 `__rdl_alloc` / 用户 `#[global_allocator]` 的实现（同 cg_gcc）。
    pub fn allocator_shim(tcx: TyCtxt<'tcx>, methods: &[AllocatorMethod]) -> Module {
        let mut m = Module::default();
        for method in methods {
            let params: Vec<Ty> = method
                .inputs
                .iter()
                .flat_map(|i| match i.ty {
                    AllocatorTy::Layout => vec![Types::I32, Types::I32],
                    AllocatorTy::Ptr => vec![Types::PTR],
                    AllocatorTy::Usize => vec![Types::I32],
                    AllocatorTy::Never | AllocatorTy::ResultPtr | AllocatorTy::Unit => bug!("分配器的形参不对"),
                })
                .collect();
            let ret = match method.output {
                AllocatorTy::ResultPtr => Types::PTR,
                AllocatorTy::Never | AllocatorTy::Unit => Types::VOID,
                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => bug!("分配器的返回不对"),
            };
            let from = mangle_internal_symbol(tcx, &global_fn_name(method.name));
            let to = mangle_internal_symbol(tcx, &default_fn_name(method.name));
            m.wrapper(from, Some(to), &params, ret);
        }
        m.wrapper(mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE), None, &[], Types::VOID);
        m
    }
}

/// 直接在模块上拼转发函数（分配器 shim 用：没有 rustc 的实例，不走 builder）。
trait Wrapper {
    fn wrapper(&mut self, from: String, to: Option<String>, params: &[Ty], ret: Ty);
}

impl Wrapper for Module {
    fn wrapper(&mut self, from: String, to: Option<String>, params: &[Ty], ret: Ty) {
        let decl = |name: String, hidden: bool, blocks: Vec<Block>, vals: Vec<Ty>| Function {
            name,
            linkage: RvLinkage::External,
            hidden,
            ret,
            ret_attrs: ArgAttrs::default(),
            params: params.iter().map(|&ty| Param { ty, attrs: ArgAttrs::default() }).collect(),
            varargs: false,
            section: None,
            align: None,
            stack_probe: None,
            blocks,
            vals,
        };
        let n = u32::try_from(params.len()).expect("形参");
        let args: Vec<ValId> = (0..n).map(ValId).collect();
        let mut vals = params.to_vec();
        let mut insts = Vec::new();
        let mut result = None;
        if let Some(to) = to {
            let fty = self.types.get(Kind::Func(ret, params.to_vec(), false));
            self.functions.push(decl(to.clone(), true, Vec::new(), Vec::new()));
            if ret != Types::VOID {
                result = Some(ValId(n));
                vals.push(ret);
            }
            let call = Op::Call { fty, callee: riscvc::ir::Callee::Direct(to), args: args.iter().map(|&a| (a, ArgAttrs::default())).collect(), ret: ArgAttrs::default() };
            insts.push(Inst { result, op: call });
        }
        insts.push(Inst { result: None, op: Op::Ret(result) });
        self.functions.push(decl(from, false, vec![Block { params: args, insts }], vals));
    }
}

impl<'tcx> BackendTypes for CodegenCx<'tcx> {
    type Value = Val;
    type Function = Fid;
    type BasicBlock = Blk;
    type Type = Ty;
    type FunctionSignature = Ty;
    type Funclet = ();
    type DIScope = ();
    type DILocation = ();
    type DIVariable = ();
}

impl<'tcx> HasTyCtxt<'tcx> for CodegenCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl HasDataLayout for CodegenCx<'_> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl HasTargetSpec for CodegenCx<'_> {
    fn target_spec(&self) -> &rustc_target::spec::Target {
        &self.tcx.sess.target
    }
}

impl HasX86AbiOpt for CodegenCx<'_> {
    fn x86_abi_opt(&self) -> X86Abi {
        X86Abi { regparm: None, reg_struct_return: false }
    }
}

impl<'tcx> HasTypingEnv<'tcx> for CodegenCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: ty::Ty<'tcx>) -> ! {
        self.tcx.dcx().span_fatal(span, format!("riscvc：{ty} 的布局：{err}"))
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    fn handle_fn_abi_err(&self, err: FnAbiError<'tcx>, span: Span, req: FnAbiRequest<'tcx>) -> ! {
        match req {
            FnAbiRequest::OfFnPtr { sig, extra_args } => span_bug!(span, "fn_abi_of_fn_ptr({sig}, {extra_args:?})：{err:?}"),
            FnAbiRequest::OfInstance { instance, extra_args } => span_bug!(span, "fn_abi_of_instance({instance}, {extra_args:?})：{err:?}"),
        }
    }
}

impl<'tcx> BaseTypeCodegenMethods for CodegenCx<'tcx> {
    fn type_i8(&self) -> Ty {
        Types::I8
    }
    fn type_i16(&self) -> Ty {
        Types::I16
    }
    fn type_i32(&self) -> Ty {
        Types::I32
    }
    fn type_i64(&self) -> Ty {
        Types::I64
    }
    fn type_i128(&self) -> Ty {
        self.ty(Kind::Int(128))
    }
    fn type_isize(&self) -> Ty {
        Types::I32
    }
    fn type_f16(&self) -> Ty {
        self.ty(Kind::Half)
    }
    fn type_f32(&self) -> Ty {
        Types::FLOAT
    }
    fn type_f64(&self) -> Ty {
        Types::DOUBLE
    }
    fn type_f128(&self) -> Ty {
        self.ty(Kind::Fp128)
    }
    fn type_array(&self, ty: Ty, len: u64) -> Ty {
        self.ty(Kind::Array(len, ty))
    }
    fn type_func(&self, args: &[Ty], ret: Ty) -> Ty {
        self.ty(Kind::Func(ret, args.to_vec(), false))
    }
    fn type_kind(&self, ty: Ty) -> TypeKind {
        match self.kind(ty) {
            Kind::Void => TypeKind::Void,
            Kind::Int(_) => TypeKind::Integer,
            Kind::Half => TypeKind::Half,
            Kind::BFloat => TypeKind::BFloat,
            Kind::Float => TypeKind::Float,
            Kind::Double => TypeKind::Double,
            Kind::Fp128 => TypeKind::FP128,
            Kind::Ptr => TypeKind::Pointer,
            Kind::Label => TypeKind::Label,
            Kind::Metadata => TypeKind::Metadata,
            Kind::Token => TypeKind::Token,
            Kind::Array(..) => TypeKind::Array,
            Kind::Vector(..) => TypeKind::Vector,
            Kind::Struct(..) | Kind::Opaque(_) => TypeKind::Struct,
            Kind::Func(..) => TypeKind::Function,
        }
    }
    fn type_ptr(&self) -> Ty {
        Types::PTR
    }
    fn type_ptr_ext(&self, _space: AddressSpace) -> Ty {
        Types::PTR
    }
    fn element_type(&self, ty: Ty) -> Ty {
        match self.kind(ty) {
            Kind::Array(_, e) | Kind::Vector(_, e) => e,
            k => bug!("riscvc：{k:?} 没有元素类型"),
        }
    }
    fn vector_length(&self, ty: Ty) -> usize {
        match self.kind(ty) {
            Kind::Vector(n, _) => n as usize,
            k => bug!("riscvc：{k:?} 不是向量"),
        }
    }
    fn float_width(&self, ty: Ty) -> usize {
        match self.kind(ty) {
            Kind::Half | Kind::BFloat => 16,
            Kind::Float => 32,
            Kind::Double => 64,
            Kind::Fp128 => 128,
            k => bug!("riscvc：{k:?} 不是浮点"),
        }
    }
    fn int_width(&self, ty: Ty) -> u64 {
        match self.kind(ty) {
            Kind::Int(n) => u64::from(n),
            k => bug!("riscvc：{k:?} 不是整数"),
        }
    }
    fn val_ty(&self, v: Val) -> Ty {
        match v {
            Val::Local(f, id) => self.module.borrow().functions[f.0 as usize].vals[id.0 as usize],
            Val::Const(i) => self.consts.borrow().list[i as usize].0,
            Val::Func(_) | Val::Global(_) => Types::PTR,
        }
    }
}

impl<'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Ty {
        match layout.backend_repr {
            BackendRepr::Scalar(s) => self.scalar_ty(s),
            BackendRepr::ScalarPair(a, b) => {
                let fields = vec![self.scalar_ty(a), self.scalar_ty(b)];
                self.ty(Kind::Struct(fields, false))
            }
            BackendRepr::SimdVector { element, count } => {
                let e = self.scalar_ty(element);
                self.ty(Kind::Vector(u32::try_from(count).expect("向量长度"), e))
            }
            _ => self.bytes_ty(layout.size.bytes()),
        }
    }

    fn cast_backend_type(&self, cast: &CastTarget) -> Ty {
        let unit = self.reg_backend_type(&cast.rest.unit);
        let count = if cast.rest.total == Size::ZERO { 0 } else { cast.rest.total.bytes().div_ceil(cast.rest.unit.size.bytes()) };
        if cast.prefix.is_empty() {
            if count == 1 && (!cast.rest.is_consecutive || cast.rest.unit != Reg::i128()) {
                return unit;
            }
            return self.type_array(unit, count);
        }
        let fields: Vec<Ty> = cast.prefix.iter().map(|r| self.reg_backend_type(r)).chain((0..count).map(|_| unit)).collect();
        self.ty(Kind::Struct(fields, false))
    }

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>) -> Ty {
        self.fn_sig(fn_abi).0
    }

    fn fn_ptr_backend_type(&self, _fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>) -> Ty {
        Types::PTR
    }

    fn reg_backend_type(&self, reg: &Reg) -> Ty {
        let bits = reg.size.bits();
        match reg.kind {
            RegKind::Integer => self.ty(Kind::Int(u32::try_from(bits).expect("寄存器宽度"))),
            RegKind::Float => match bits {
                16 => self.ty(Kind::Half),
                32 => Types::FLOAT,
                64 => Types::DOUBLE,
                128 => self.ty(Kind::Fp128),
                _ => bug!("riscvc：{bits} 位的浮点寄存器"),
            },
            RegKind::Vector { .. } => self.ty(Kind::Vector(u32::try_from(reg.size.bytes()).expect("向量宽度"), Types::I8)),
        }
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Ty {
        match layout.backend_repr {
            BackendRepr::Scalar(s) if s.is_bool() => Types::I1,
            BackendRepr::ScalarPair(..) => {
                let fields = vec![self.scalar_pair_element_backend_type(layout, 0, true), self.scalar_pair_element_backend_type(layout, 1, true)];
                self.ty(Kind::Struct(fields, false))
            }
            _ => self.backend_type(layout),
        }
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(layout.backend_repr, BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. })
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(layout.backend_repr, BackendRepr::ScalarPair(..))
    }

    fn scalar_pair_element_backend_type(&self, layout: TyAndLayout<'tcx>, index: usize, immediate: bool) -> Ty {
        let BackendRepr::ScalarPair(a, b) = layout.backend_repr else { bug!("riscvc：{layout:?} 不是标量对") };
        let s = [a, b][index];
        if immediate && s.is_bool() { Types::I1 } else { self.scalar_ty(s) }
    }
}

impl<'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'tcx> {}

impl<'tcx> ConstCodegenMethods for CodegenCx<'tcx> {
    fn const_null(&self, t: Ty) -> Val {
        self.cst(t, Const::Zero)
    }
    fn const_undef(&self, t: Ty) -> Val {
        self.cst(t, Const::Undef)
    }
    fn const_poison(&self, t: Ty) -> Val {
        self.cst(t, Const::Undef)
    }
    fn const_bool(&self, val: bool) -> Val {
        self.int(Types::I1, u128::from(val))
    }
    fn const_i8(&self, i: i8) -> Val {
        self.const_int(Types::I8, i64::from(i))
    }
    fn const_i16(&self, i: i16) -> Val {
        self.const_int(Types::I16, i64::from(i))
    }
    fn const_i32(&self, i: i32) -> Val {
        self.const_int(Types::I32, i64::from(i))
    }
    fn const_i64(&self, i: i64) -> Val {
        self.const_int(Types::I64, i)
    }
    fn const_int(&self, t: Ty, i: i64) -> Val {
        // 位型按位宽截断（Const::Int 的约定）：先符号扩展到 128 位
        let bits = self.int_width(t);
        let v = u128::from_ne_bytes(i128::from(i).to_ne_bytes());
        self.int(t, if bits >= 128 { v } else { v & ((1u128 << bits) - 1) })
    }
    fn const_u8(&self, i: u8) -> Val {
        self.int(Types::I8, u128::from(i))
    }
    fn const_u32(&self, i: u32) -> Val {
        self.int(Types::I32, u128::from(i))
    }
    fn const_u64(&self, i: u64) -> Val {
        self.int(Types::I64, u128::from(i))
    }
    fn const_u128(&self, i: u128) -> Val {
        self.int(self.type_i128(), i)
    }
    fn const_usize(&self, i: u64) -> Val {
        self.int(Types::I32, u128::from(i))
    }
    fn const_uint(&self, t: Ty, i: u64) -> Val {
        self.int(t, u128::from(i))
    }
    fn const_uint_big(&self, t: Ty, u: u128) -> Val {
        self.int(t, u)
    }
    fn const_real(&self, t: Ty, val: f64) -> Val {
        #[allow(clippy::cast_possible_truncation, reason = "f32 常量本来就是舍入后的值")]
        let bits = match self.kind(t) {
            Kind::Float => u64::from((val as f32).to_bits()),
            Kind::Double => val.to_bits(),
            k => bug!("riscvc：{k:?} 的浮点常量"),
        };
        self.cst(t, Const::Float(bits))
    }
    fn const_str(&self, s: &str) -> (Val, Val) {
        let bytes = self.cst(self.bytes_ty(s.len() as u64), Const::Bytes(s.as_bytes().to_vec()));
        (self.anon(bytes, Align::ONE, false), self.const_usize(s.len() as u64))
    }
    fn const_struct(&self, elts: &[Val], packed: bool) -> Val {
        let (tys, cs): (Vec<Ty>, Vec<Const>) = elts.iter().map(|&v| self.konst(v)).unzip();
        let ty = self.ty(Kind::Struct(tys, packed));
        self.cst(ty, Const::Agg(cs))
    }
    fn const_vector(&self, elts: &[Val]) -> Val {
        let (tys, cs): (Vec<Ty>, Vec<Const>) = elts.iter().map(|&v| self.konst(v)).unzip();
        let ty = self.ty(Kind::Vector(u32::try_from(elts.len()).expect("向量长度"), tys[0]));
        self.cst(ty, Const::Agg(cs))
    }
    fn const_to_opt_uint(&self, v: Val) -> Option<u64> {
        self.const_to_opt_u128(v, false).and_then(|x| u64::try_from(x).ok())
    }
    fn const_to_opt_u128(&self, v: Val, sign_ext: bool) -> Option<u128> {
        let Val::Const(_) = v else { return None };
        let (ty, Const::Int(x)) = self.konst(v) else { return None };
        let bits = self.module.borrow().types.bits(ty)?;
        if !sign_ext || bits >= 128 || x >> (bits - 1) & 1 == 0 {
            return Some(x);
        }
        Some(x | (u128::MAX << bits))
    }
    fn scalar_to_backend_with_pac(&self, cv: InterpScalar, layout: abi::Scalar, llty: Ty, _pac: Option<PacMetadata>) -> Val {
        match cv {
            InterpScalar::Int(int) => {
                let data = int.to_bits(layout.size(self));
                match self.kind(llty) {
                    Kind::Float | Kind::Double | Kind::Half | Kind::Fp128 => self.cst(llty, Const::Float(u64::try_from(data).expect("浮点位型"))),
                    _ => self.int(llty, data),
                }
            }
            InterpScalar::Ptr(ptr, _) => {
                let (prov, offset) = ptr.prov_and_relative_offset();
                let base = match self.tcx.global_alloc(prov.alloc_id()) {
                    GlobalAlloc::Memory(alloc) => {
                        // 零长：指针只要对齐、非空，⛔ 占全局
                        if alloc.inner().len() == 0 {
                            return self.int(llty, u128::from(alloc.inner().align.bytes().wrapping_add(offset.bytes())));
                        }
                        let init = self.const_alloc(alloc.inner());
                        self.anon(init, alloc.inner().align, alloc.inner().mutability.is_mut())
                    }
                    GlobalAlloc::Function { instance, .. } => self.get_fn_addr(instance, None),
                    GlobalAlloc::VTable(ty, dyn_ty) => {
                        let id = self.tcx.vtable_allocation((ty, dyn_ty.principal().map(|p| self.tcx.instantiate_bound_regions_with_erased(p))));
                        self.static_addr_of(self.tcx.global_alloc(id).unwrap_memory(), None)
                    }
                    GlobalAlloc::TypeId { .. } => return self.int(llty, u128::from(offset.bytes())),
                    GlobalAlloc::Static(def_id) => Val::Global(self.get_static_global(def_id)),
                };
                let v = self.const_ptr_byte_offset(base, offset);
                if llty == Types::PTR {
                    return v;
                }
                let (_, c) = self.konst(v);
                self.cst(llty, c)
            }
        }
    }
    fn const_ptr_byte_offset(&self, val: Val, offset: Size) -> Val {
        if offset == Size::ZERO {
            return val;
        }
        match self.konst(val) {
            (ty, Const::Global(name, off)) => self.cst(ty, Const::Global(name, off + i64::try_from(offset.bytes()).expect("偏移"))),
            (_, c) => bug!("riscvc：给常量 {c:?} 加偏移"),
        }
    }
}

impl<'tcx> StaticCodegenMethods for CodegenCx<'tcx> {
    fn static_addr_of(&self, alloc: ConstAllocation<'_>, _kind: Option<&str>) -> Val {
        let init = self.const_alloc(alloc.inner());
        self.anon(init, alloc.inner().align, false)
    }

    fn codegen_static(&mut self, def_id: DefId) {
        let Ok(alloc) = self.tcx.eval_static_initializer(def_id) else { return };
        let alloc = alloc.inner();
        let init = self.const_alloc(alloc);
        let (ty, c) = self.konst(init);
        let g = self.get_static_global(def_id);
        let attrs = self.tcx.codegen_fn_attrs(def_id);
        let mut m = self.module.borrow_mut();
        let x = &mut m.globals[g as usize];
        x.ty = ty;
        x.init = Some(c);
        x.constant = !alloc.mutability.is_mut() && !self.tcx.is_mutable_static(def_id);
        x.align = Some(alloc.align.bytes());
        x.section = attrs.link_section.map(|s| s.to_string());
    }
}

impl<'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn vtables(&self) -> &RefCell<FxHashMap<(ty::Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), Val>> {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> Fid {
        if let Some(&f) = self.instances.borrow().get(&instance) {
            return f;
        }
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let f = self.declare_fn(self.tcx.symbol_name(instance).name, fn_abi);
        self.instances.borrow_mut().insert(instance, f);
        f
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>, _pac: Option<PacMetadata>) -> Val {
        Val::Func(self.get_fn(instance))
    }

    fn eh_personality(&self) -> Fid {
        let fty = self.type_func(&[], Types::I32);
        self.declare("rust_eh_personality", fty, ArgAttrs::default(), Vec::new())
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, _llfn: Fid) {}

    fn apply_target_cpu_attr(&self, _llfn: Fid) {}

    fn declare_c_main(&self, fn_type: Ty) -> Option<Fid> {
        if self.fns.borrow().contains_key("main") {
            return None;
        }
        Some(self.declare("main", fn_type, ArgAttrs::default(), Vec::new()))
    }

    fn intrinsic_call_expects_place_always(&self, _name: Symbol) -> bool {
        false
    }
}

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn predefine_static(&mut self, def_id: DefId, linkage: Linkage, visibility: Visibility, symbol_name: &str) {
        let instance = Instance::mono(self.tcx, def_id);
        let DefKind::Static { nested, .. } = self.tcx.def_kind(def_id) else { bug!("riscvc：{def_id:?} 不是 static") };
        let ty = if nested { self.tcx.types.unit } else { instance.ty(self.tcx, self.typing_env()) };
        let size = self.layout_of(ty).size.bytes();
        let g = self.global(symbol_name, self.bytes_ty(size));
        {
            let mut m = self.module.borrow_mut();
            let x = &mut m.globals[g as usize];
            x.linkage = linkage.into_rv();
            x.hidden = visibility == Visibility::Hidden;
            x.thread_local = self.tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
        }
        self.statics.borrow_mut().insert(def_id, g);
    }

    fn predefine_fn(&mut self, instance: Instance<'tcx>, linkage: Linkage, visibility: Visibility, symbol_name: &str) {
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let f = self.declare_fn(symbol_name, fn_abi);
        let attrs = self.tcx.codegen_instance_attrs(instance.def);
        // compiler_builtins 的非内部符号一律 hidden（同 cg_llvm：别让它们从 dylib 漏出去）
        let hidden = visibility == Visibility::Hidden || (linkage != Linkage::Internal && self.tcx.is_compiler_builtins(LOCAL_CRATE));
        {
            let mut m = self.module.borrow_mut();
            let x = &mut m.functions[f.0 as usize];
            x.linkage = linkage.into_rv();
            x.hidden = hidden;
            x.section = attrs.link_section.map(|s| s.to_string());
            x.align = attrs.alignment.map(|a| a.bytes());
        }
        self.instances.borrow_mut().insert(instance, f);
    }
}

/// rustc 的链接属性 ⇒ riscvc 的。
trait IntoRv {
    fn into_rv(self) -> RvLinkage;
}

impl IntoRv for Linkage {
    fn into_rv(self) -> RvLinkage {
        match self {
            Linkage::External => RvLinkage::External,
            Linkage::AvailableExternally => RvLinkage::Available,
            Linkage::LinkOnceAny | Linkage::LinkOnceODR => RvLinkage::LinkOnce,
            Linkage::WeakAny | Linkage::WeakODR => RvLinkage::Weak,
            Linkage::Internal => RvLinkage::Internal,
            Linkage::ExternalWeak => RvLinkage::ExternWeak,
            Linkage::Common => RvLinkage::Common,
        }
    }
}

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(&self, _ty: ty::Ty<'tcx>, _trait_ref: Option<ExistentialTraitRef<'tcx>>, _vtable: Val) {}
}

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn codegen_global_asm(&mut self, _template: &[rustc_ast::InlineAsmTemplatePiece], _operands: &[GlobalAsmOperandRef<'tcx>], _options: rustc_ast::InlineAsmOptions, line_spans: &[Span]) {
        let span = line_spans.first().copied().unwrap_or_default();
        self.tcx.dcx().span_fatal(span, "riscvc 后端不支持 global_asm! / naked 函数")
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        self.tcx.symbol_name(instance).name.to_string()
    }
}

impl Blk {
    /// 跳转目标（不带块参数：rustc_codegen_ssa 不出 phi，跨块的值都经内存）。
    pub fn target(self) -> Target {
        Target { block: self.b, args: Vec::new() }
    }
}
