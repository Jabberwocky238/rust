//! rustc_codegen_ssa 的 Builder：在 riscvc 函数的某个块末尾追加 `ir::Op`。
use std::ops::Deref;

use riscvc::ir::{BinOp, Block, BlockId, CastOp, Callee, Const, FPred, IPred, Index, Inst, Intr, Kind, Op, RmwOp, Ty, Types, ValId};
use rustc_abi::{Align, BackendRepr, HasDataLayout, Scalar, Size, TargetDataLayout, WrappingRange};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{self, AtomicOrdering, Instance, TyCtxt};
use rustc_span::Span;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, HasX86AbiOpt, X86Abi};

use crate::context::{Blk, CodegenCx, Fid, Val};

pub struct Builder<'a, 'tcx> {
    pub cx: &'a CodegenCx<'tcx>,
    pub blk: Blk,
}

impl<'a, 'tcx> Deref for Builder<'a, 'tcx> {
    type Target = CodegenCx<'tcx>;
    fn deref(&self) -> &CodegenCx<'tcx> {
        self.cx
    }
}

/// rustc 的枚举 ⇒ riscvc 的（一一对应的几张表）。
trait Rv<T> {
    fn rv(self) -> T;
}

macro_rules! rv_map {
    ($($from:ident => $to:ident { $($a:ident => $b:ident),* $(,)? })*) => {
        $(impl Rv<$to> for $from {
            fn rv(self) -> $to {
                match self { $($from::$a => $to::$b),* }
            }
        })*
    };
}

rv_map! {
    IntPredicate => IPred { IntEQ => Eq, IntNE => Ne, IntUGT => Ugt, IntUGE => Uge, IntULT => Ult, IntULE => Ule, IntSGT => Sgt, IntSGE => Sge, IntSLT => Slt, IntSLE => Sle }
    RealPredicate => FPred {
        RealPredicateFalse => False, RealOEQ => Oeq, RealOGT => Ogt, RealOGE => Oge, RealOLT => Olt, RealOLE => Ole, RealONE => One, RealORD => Ord,
        RealUNO => Uno, RealUEQ => Ueq, RealUGT => Ugt, RealUGE => Uge, RealULT => Ult, RealULE => Ule, RealUNE => Une, RealPredicateTrue => True,
    }
    AtomicRmwBinOp => RmwOp {
        AtomicXchg => Xchg, AtomicAdd => Add, AtomicSub => Sub, AtomicAnd => And, AtomicNand => Nand, AtomicOr => Or, AtomicXor => Xor,
        AtomicMax => Max, AtomicMin => Min, AtomicUMax => UMax, AtomicUMin => UMin,
    }
}

/// 内存序 ⇒ C 的编号（riscvc 的约定：0 relaxed、2 acquire、3 release、4 acq_rel、5 seq_cst）。
impl Rv<u8> for AtomicOrdering {
    fn rv(self) -> u8 {
        match self {
            AtomicOrdering::Relaxed => 0,
            AtomicOrdering::Acquire => 2,
            AtomicOrdering::Release => 3,
            AtomicOrdering::AcqRel => 4,
            AtomicOrdering::SeqCst => 5,
        }
    }
}

impl<'tcx> CodegenCx<'tcx> {
    /// 给函数追加一个块；第一个块是入口，它的参数就是函数形参。
    pub fn append(&self, f: Fid) -> Blk {
        let mut m = self.module.borrow_mut();
        let func = &mut m.functions[f.0 as usize];
        let mut block = Block::default();
        if func.blocks.is_empty() {
            let tys: Vec<Ty> = func.params.iter().map(|p| p.ty).collect();
            for ty in tys {
                block.params.push(ValId(u32::try_from(func.vals.len()).expect("值太多")));
                func.vals.push(ty);
            }
        }
        func.blocks.push(block);
        Blk { f, b: BlockId(u32::try_from(func.blocks.len() - 1).expect("块太多")) }
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// 当前函数里的操作数：常量 / 符号地址在当前块落成一条 `Op::Const`。
    pub fn get(&mut self, v: Val) -> ValId {
        match v {
            Val::Local(f, id) => {
                debug_assert_eq!(f, self.blk.f, "跨函数用 SSA 值");
                id
            }
            _ => {
                let (ty, c) = self.cx.konst(v);
                self.push(Some(ty), Op::Const(c)).expect("有结果")
            }
        }
    }

    fn push(&mut self, ty: Option<Ty>, op: Op) -> Option<ValId> {
        let mut m = self.cx.module.borrow_mut();
        let f = &mut m.functions[self.blk.f.0 as usize];
        let result = ty.map(|ty| {
            f.vals.push(ty);
            ValId(u32::try_from(f.vals.len() - 1).expect("值太多"))
        });
        f.blocks[self.blk.b.0 as usize].insts.push(Inst { result, op });
        result
    }

    /// 出一条有结果的指令。
    pub fn emit(&mut self, ty: Ty, op: Op) -> Val {
        Val::Local(self.blk.f, self.push(Some(ty), op).expect("有结果"))
    }

    /// 出一条没有结果的指令。
    pub fn emit0(&mut self, op: Op) {
        self.push(None, op);
    }

    /// `void` 的「结果」（rustc 要一个值占位）。
    pub fn none(&self) -> Val {
        self.cx.cst(Types::VOID, Const::Undef)
    }

    fn bin(&mut self, op: BinOp, a: Val, b: Val) -> Val {
        let ty = self.val_ty(a);
        let (a, b) = (self.get(a), self.get(b));
        self.emit(ty, Op::Bin(op, a, b))
    }

    fn cast(&mut self, op: CastOp, v: Val, ty: Ty) -> Val {
        if self.val_ty(v) == ty {
            return v;
        }
        let v = self.get(v);
        self.emit(ty, Op::Cast(op, v))
    }

    pub fn intr(&mut self, ty: Ty, i: Intr, args: &[Val]) -> Val {
        let args = args.iter().map(|&a| self.get(a)).collect();
        if ty == Types::VOID {
            self.emit0(Op::Intr(i, args));
            return self.none();
        }
        self.emit(ty, Op::Intr(i, args))
    }

    /// 聚合值第 `idx` 个成员的类型。
    fn member(&self, agg: Ty, idx: u64) -> Ty {
        match self.kind(agg) {
            Kind::Struct(fields, _) => fields[idx as usize],
            Kind::Array(_, e) | Kind::Vector(_, e) => e,
            k => bug!("riscvc：{k:?} 不是聚合"),
        }
    }

    /// `{T, i1}`：带溢出 / 比较交换的结果。
    fn with_flag(&mut self, op: Op, ty: Ty) -> (Val, Val) {
        let pair = self.ty(Kind::Struct(vec![ty, Types::I1], false));
        let r = self.emit(pair, op);
        (self.extract_value(r, 0), self.extract_value(r, 1))
    }

    /// 饱和的浮点 ⇒ 整数（同 riscvc 前端对 `fptosi.sat` 的展开）：NaN ⇒ 0，越界 ⇒ 饱和值。
    fn fpto_sat(&mut self, signed: bool, x: Val, dest: Ty) -> Val {
        let (w, xt) = (self.int_width(dest), self.val_ty(x));
        let single = match self.kind(xt) {
            Kind::Float => true,
            Kind::Double => false,
            k => bug!("riscvc：{k:?} 的饱和转换"),
        };
        let w = i32::try_from(w).expect("位宽");
        let (lo, hi) = if signed { (-(2f64.powi(w - 1)), 2f64.powi(w - 1)) } else { (0.0, 2f64.powi(w)) };
        let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
        let (min, max) = if signed { ((mask >> 1) + 1, mask >> 1) } else { (0, mask) };
        #[allow(clippy::cast_possible_truncation, reason = "界都是 2 的幂，float 精确")]
        let bits = |v: f64| if single { u64::from((v as f32).to_bits()) } else { v.to_bits() };
        let (flo, fhi) = (self.cst(xt, Const::Float(bits(lo))), self.cst(xt, Const::Float(bits(hi))));
        let (cmin, cmax, zero) = (self.int(dest, min), self.int(dest, max), self.int(dest, 0));
        let t = if signed { self.fptosi(x, dest) } else { self.fptoui(x, dest) };
        let under = self.fcmp(RealPredicate::RealOLT, x, flo);
        let t = self.select(under, cmin, t);
        let over = self.fcmp(RealPredicate::RealOGE, x, fhi);
        let t = self.select(over, cmax, t);
        let nan = self.fcmp(RealPredicate::RealUNO, x, x);
        self.select(nan, zero, t)
    }
}

/// 一组二元运算：方法名 ⇒ `BinOp`。
macro_rules! binops {
    ($($name:ident => $op:ident),* $(,)?) => {
        $(fn $name(&mut self, lhs: Val, rhs: Val) -> Val { self.bin(BinOp::$op, lhs, rhs) })*
    };
}

/// 一组转换：方法名 ⇒ `CastOp`。
macro_rules! casts {
    ($($name:ident => $op:ident),* $(,)?) => {
        $(fn $name(&mut self, val: Val, dest_ty: Ty) -> Val { self.cast(CastOp::$op, val, dest_ty) })*
    };
}

impl<'a, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'tcx> {
    type CodegenCx = CodegenCx<'tcx>;

    fn build(cx: &'a CodegenCx<'tcx>, llbb: Blk) -> Self {
        Self { cx, blk: llbb }
    }

    fn cx(&self) -> &CodegenCx<'tcx> {
        self.cx
    }

    fn llbb(&self) -> Blk {
        self.blk
    }

    fn set_span(&mut self, _span: Span) {}

    fn append_block(cx: &'a CodegenCx<'tcx>, llfn: Fid, _name: &str) -> Blk {
        cx.append(llfn)
    }

    fn append_sibling_block(&mut self, _name: &str) -> Blk {
        self.cx.append(self.blk.f)
    }

    fn switch_to_block(&mut self, llbb: Blk) {
        self.blk = llbb;
    }

    fn ret_void(&mut self) {
        self.emit0(Op::Ret(None));
    }

    fn ret(&mut self, v: Val) {
        let v = self.get(v);
        self.emit0(Op::Ret(Some(v)));
    }

    fn br(&mut self, dest: Blk) {
        self.emit0(Op::Br(dest.target()));
    }

    fn cond_br(&mut self, cond: Val, then_llbb: Blk, else_llbb: Blk) {
        let c = self.get(cond);
        self.emit0(Op::CondBr(c, then_llbb.target(), else_llbb.target()));
    }

    fn switch(&mut self, v: Val, else_llbb: Blk, cases: impl ExactSizeIterator<Item = (u128, Blk)>) {
        let v = self.get(v);
        let cases = cases.map(|(k, b)| (k, b.target())).collect();
        self.emit0(Op::Switch(v, else_llbb.target(), cases));
    }

    fn invoke(
        &mut self,
        llty: Ty,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, ty::Ty<'tcx>>>,
        llfn: Val,
        args: &[Val],
        then: Blk,
        _catch: Blk,
        funclet: Option<&()>,
        instance: Option<Instance<'tcx>>,
    ) -> Val {
        // panic = abort：不会展开，invoke 就是 call + 跳到正常出口
        let r = self.call(llty, fn_attrs, fn_abi, llfn, args, funclet, instance);
        self.br(then);
        r
    }

    fn unreachable(&mut self) {
        self.emit0(Op::Unreachable);
    }

    binops! {
        add => Add, sub => Sub, mul => Mul, udiv => UDiv, exactudiv => UDiv, sdiv => SDiv, exactsdiv => SDiv, urem => URem, srem => SRem,
        shl => Shl, lshr => LShr, ashr => AShr, and => And, or => Or, xor => Xor,
        fadd => FAdd, fadd_fast => FAdd, fadd_algebraic => FAdd, fsub => FSub, fsub_fast => FSub, fsub_algebraic => FSub,
        fmul => FMul, fmul_fast => FMul, fmul_algebraic => FMul, fdiv => FDiv, fdiv_fast => FDiv, fdiv_algebraic => FDiv,
        frem => FRem, frem_fast => FRem, frem_algebraic => FRem,
    }

    fn neg(&mut self, v: Val) -> Val {
        let zero = self.int(self.val_ty(v), 0);
        self.sub(zero, v)
    }

    fn fneg(&mut self, v: Val) -> Val {
        let ty = self.val_ty(v);
        let v = self.get(v);
        self.emit(ty, Op::FNeg(v))
    }

    fn not(&mut self, v: Val) -> Val {
        // 全 1（Const::Int 按位宽截断）
        let ones = self.int(self.val_ty(v), u128::MAX);
        self.xor(v, ones)
    }

    fn checked_binop(&mut self, oop: OverflowOp, ty: ty::Ty<'tcx>, lhs: Val, rhs: Val) -> (Val, Val) {
        let i = match (oop, ty.is_signed()) {
            (OverflowOp::Add, true) => Intr::SAddOv,
            (OverflowOp::Add, false) => Intr::UAddOv,
            (OverflowOp::Sub, true) => Intr::SSubOv,
            (OverflowOp::Sub, false) => Intr::USubOv,
            (OverflowOp::Mul, true) => Intr::SMulOv,
            (OverflowOp::Mul, false) => Intr::UMulOv,
        };
        let t = self.val_ty(lhs);
        let (a, b) = (self.get(lhs), self.get(rhs));
        self.with_flag(Op::Intr(i, vec![a, b]), t)
    }

    fn from_immediate(&mut self, val: Val) -> Val {
        if self.val_ty(val) == Types::I1 { self.zext(val, Types::I8) } else { val }
    }

    fn to_immediate_scalar(&mut self, val: Val, scalar: Scalar) -> Val {
        if scalar.is_bool() { self.trunc(val, Types::I1) } else { val }
    }

    fn alloca(&mut self, size: Size, align: Align) -> Val {
        // 进前导（入口块开头），⛔ 出在当前位置：循环里的 alloca 会反复长栈，mem2reg 也只认入口块的
        let ty = self.bytes_ty(size.bytes());
        let (one, p) = {
            let mut m = self.cx.module.borrow_mut();
            let f = &mut m.functions[self.blk.f.0 as usize];
            f.vals.extend([Types::I32, Types::PTR]);
            let n = u32::try_from(f.vals.len()).expect("值太多");
            (ValId(n - 2), ValId(n - 1))
        };
        let mut pre = self.cx.prelude.borrow_mut();
        let insts = pre.entry(self.blk.f).or_default();
        insts.push(Inst { result: Some(one), op: Op::Const(Const::Int(1)) });
        insts.push(Inst { result: Some(p), op: Op::Alloca { ty, count: one, align: align.bytes() } });
        Val::Local(self.blk.f, p)
    }

    fn alloca_with_ty(&mut self, layout: TyAndLayout<'tcx>) -> Val {
        self.alloca(layout.size, layout.align.abi)
    }

    fn load(&mut self, ty: Ty, ptr: Val, align: Align) -> Val {
        let ptr = self.get(ptr);
        self.emit(ty, Op::Load { ptr, align: align.bytes(), volatile: false, atomic: None })
    }

    fn volatile_load(&mut self, ty: Ty, ptr: Val, align: Align) -> Val {
        let ptr = self.get(ptr);
        self.emit(ty, Op::Load { ptr, align: align.bytes(), volatile: true, atomic: None })
    }

    fn atomic_load(&mut self, ty: Ty, ptr: Val, order: AtomicOrdering, size: Size) -> Val {
        let ptr = self.get(ptr);
        self.emit(ty, Op::Load { ptr, align: size.bytes(), volatile: false, atomic: Some(order.rv()) })
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, Val>) -> OperandRef<'tcx, Val> {
        assert_eq!(place.val.llextra.is_some(), place.layout.is_unsized());
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }
        let val = if place.val.llextra.is_some() {
            OperandValue::Ref(place.val)
        } else if self.is_backend_immediate(place.layout) {
            let v = self.load(self.backend_type(place.layout), place.val.llval, place.val.align);
            OperandValue::Immediate(match place.layout.backend_repr {
                BackendRepr::Scalar(s) => self.to_immediate_scalar(v, s),
                _ => v,
            })
        } else if let BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let off = a.size(self).align_to(b.default_align(self).abi);
            let part = |bx: &mut Self, i: usize, s: Scalar, align: Align| {
                let ptr = if i == 0 { place.val.llval } else { bx.inbounds_ptradd(place.val.llval, bx.const_usize(off.bytes())) };
                let ty = bx.scalar_pair_element_backend_type(place.layout, i, false);
                let v = bx.load(ty, ptr, align);
                bx.to_immediate_scalar(v, s)
            };
            let x = part(self, 0, a, place.val.align);
            let y = part(self, 1, b, place.val.align.restrict_for_offset(off));
            OperandValue::Pair(x, y)
        } else {
            OperandValue::Ref(place.val)
        };
        OperandRef { val, layout: place.layout, move_annotation: None }
    }

    fn write_operand_repeatedly(&mut self, elem: OperandRef<'tcx, Val>, count: u64, dest: PlaceRef<'tcx, Val>) {
        // 当前指针放在一个 alloca 里（没有 phi；mem2reg 会把它提成块参数）
        let zero = self.const_usize(0);
        let n = self.const_usize(count);
        let start = dest.project_index(self, zero).val.llval;
        let end = dest.project_index(self, n).val.llval;
        let psize = self.data_layout().pointer_size();
        let palign = self.data_layout().pointer_align().abi;
        let cur = self.alloca(psize, palign);
        self.store(start, cur, palign);
        let (header, body, next) = (self.append_sibling_block("repeat_header"), self.append_sibling_block("repeat_body"), self.append_sibling_block("repeat_next"));
        self.br(header);
        self.switch_to_block(header);
        let p = self.load(Types::PTR, cur, palign);
        let more = self.icmp(IntPredicate::IntNE, p, end);
        self.cond_br(more, body, next);
        self.switch_to_block(body);
        let align = dest.val.align.restrict_for_offset(dest.layout.field(self.cx, 0).size);
        elem.val.store(self, PlaceRef::new_sized_aligned(p, elem.layout, align));
        let step = self.const_usize(elem.layout.size.bytes());
        let p2 = self.inbounds_ptradd(p, step);
        self.store(p2, cur, palign);
        self.br(header);
        self.switch_to_block(next);
    }

    fn range_metadata(&mut self, _load: Val, _range: WrappingRange) {}

    fn nonnull_metadata(&mut self, _load: Val) {}

    fn store(&mut self, val: Val, ptr: Val, align: Align) -> Val {
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(&mut self, val: Val, ptr: Val, align: Align, flags: MemFlags) -> Val {
        let align = if flags.contains(MemFlags::UNALIGNED) { 1 } else { align.bytes() };
        let (val, ptr) = (self.get(val), self.get(ptr));
        self.emit0(Op::Store { val, ptr, align, volatile: flags.contains(MemFlags::VOLATILE), atomic: None });
        self.none()
    }

    fn atomic_store(&mut self, val: Val, ptr: Val, order: AtomicOrdering, size: Size) {
        let (val, ptr) = (self.get(val), self.get(ptr));
        self.emit0(Op::Store { val, ptr, align: size.bytes(), volatile: false, atomic: Some(order.rv()) });
    }

    fn gep(&mut self, ty: Ty, ptr: Val, indices: &[Val]) -> Val {
        let base = self.get(ptr);
        let indices = indices
            .iter()
            .map(|&i| match self.const_to_opt_u128(i, true) {
                Some(c) => Index::Const(i64::try_from(i128::from_ne_bytes(c.to_ne_bytes())).expect("GEP 下标")),
                None => Index::Dyn(self.get(i)),
            })
            .collect();
        self.emit(Types::PTR, Op::Gep { src: ty, base, indices })
    }

    fn inbounds_gep(&mut self, ty: Ty, ptr: Val, indices: &[Val]) -> Val {
        self.gep(ty, ptr, indices)
    }

    casts! {
        trunc => Trunc, sext => SExt, zext => ZExt, fptoui => FpToUi, fptosi => FpToSi, uitofp => UiToFp, sitofp => SiToFp,
        fptrunc => FpTrunc, fpext => FpExt, ptrtoint => PtrToInt, inttoptr => IntToPtr, bitcast => Bitcast,
    }

    fn fptoui_sat(&mut self, val: Val, dest_ty: Ty) -> Val {
        self.fpto_sat(false, val, dest_ty)
    }

    fn fptosi_sat(&mut self, val: Val, dest_ty: Ty) -> Val {
        self.fpto_sat(true, val, dest_ty)
    }

    fn intcast(&mut self, val: Val, dest_ty: Ty, is_signed: bool) -> Val {
        let (from, to) = (self.int_width(self.val_ty(val)), self.int_width(dest_ty));
        match from.cmp(&to) {
            std::cmp::Ordering::Equal => val,
            std::cmp::Ordering::Greater => self.trunc(val, dest_ty),
            std::cmp::Ordering::Less if is_signed => self.sext(val, dest_ty),
            std::cmp::Ordering::Less => self.zext(val, dest_ty),
        }
    }

    fn pointercast(&mut self, val: Val, _dest_ty: Ty) -> Val {
        val
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Val, rhs: Val) -> Val {
        let (a, b) = (self.get(lhs), self.get(rhs));
        self.emit(Types::I1, Op::ICmp(op.rv(), a, b))
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: Val, rhs: Val) -> Val {
        let (a, b) = (self.get(lhs), self.get(rhs));
        self.emit(Types::I1, Op::FCmp(op.rv(), a, b))
    }

    fn memcpy(&mut self, dst: Val, dst_align: Align, src: Val, src_align: Align, size: Val, _flags: MemFlags, _tt: Option<rustc_ast::expand::typetree::FncTree>) {
        self.intr(Types::VOID, Intr::Memcpy { dst: dst_align.bytes(), src: src_align.bytes() }, &[dst, src, size]);
    }

    fn memmove(&mut self, dst: Val, dst_align: Align, src: Val, src_align: Align, size: Val, _flags: MemFlags) {
        self.intr(Types::VOID, Intr::Memmove { dst: dst_align.bytes(), src: src_align.bytes() }, &[dst, src, size]);
    }

    fn memset(&mut self, ptr: Val, fill_byte: Val, size: Val, align: Align, _flags: MemFlags) {
        self.intr(Types::VOID, Intr::Memset { dst: align.bytes() }, &[ptr, fill_byte, size]);
    }

    fn vscale(&mut self, _ty: Ty) -> Val {
        bug!("riscvc：不支持可伸缩向量")
    }

    fn select(&mut self, cond: Val, then_val: Val, else_val: Val) -> Val {
        let ty = self.val_ty(then_val);
        let (c, a, b) = (self.get(cond), self.get(then_val), self.get(else_val));
        self.emit(ty, Op::Select(c, a, b))
    }

    fn va_arg(&mut self, _list: Val, _ty: Ty) -> Val {
        bug!("riscvc：不支持 va_arg")
    }

    fn extract_element(&mut self, _vec: Val, _idx: Val) -> Val {
        bug!("riscvc：不支持向量")
    }

    fn vector_splat(&mut self, _num_elts: usize, _elt: Val) -> Val {
        bug!("riscvc：不支持向量")
    }

    fn extract_value(&mut self, agg: Val, idx: u64) -> Val {
        let ty = self.member(self.val_ty(agg), idx);
        // 常量聚合直接取（rustc 常把常量对拆开用）
        if let Val::Const(_) = agg {
            match self.konst(agg).1 {
                Const::Agg(xs) => return self.cst(ty, xs[idx as usize].clone()),
                Const::Undef => return self.const_undef(ty),
                Const::Zero => return self.const_null(ty),
                _ => {}
            }
        }
        let a = self.get(agg);
        self.emit(ty, Op::ExtractValue(a, vec![u32::try_from(idx).expect("下标")]))
    }

    fn insert_value(&mut self, agg: Val, elt: Val, idx: u64) -> Val {
        let ty = self.val_ty(agg);
        let (a, e) = (self.get(agg), self.get(elt));
        self.emit(ty, Op::InsertValue(a, e, vec![u32::try_from(idx).expect("下标")]))
    }

    fn set_personality_fn(&mut self, _personality: Fid) {}

    fn cleanup_landing_pad(&mut self, _pers_fn: Fid) -> (Val, Val) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn filter_landing_pad(&mut self, _pers_fn: Fid) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn resume(&mut self, _exn0: Val, _exn1: Val) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn cleanup_pad(&mut self, _parent: Option<Val>, _args: &[Val]) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn cleanup_ret(&mut self, _funclet: &(), _unwind: Option<Blk>) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn catch_pad(&mut self, _parent: Val, _args: &[Val]) {
        bug!("riscvc：只支持 panic = abort")
    }

    fn catch_switch(&mut self, _parent: Option<Val>, _unwind: Option<Blk>, _handlers: &[Blk]) -> Val {
        bug!("riscvc：只支持 panic = abort")
    }

    fn get_funclet_cleanuppad(&self, _funclet: &()) -> Val {
        bug!("riscvc：只支持 panic = abort")
    }

    fn atomic_cmpxchg(&mut self, dst: Val, cmp: Val, src: Val, order: AtomicOrdering, failure_order: AtomicOrdering, _weak: bool) -> (Val, Val) {
        let ty = self.val_ty(src);
        let (p, c, s) = (self.get(dst), self.get(cmp), self.get(src));
        self.with_flag(Op::CmpXchg(p, c, s, order.rv(), failure_order.rv()), ty)
    }

    fn atomic_rmw(&mut self, op: AtomicRmwBinOp, dst: Val, src: Val, order: AtomicOrdering, _ret_ptr: bool) -> Val {
        let ty = self.val_ty(src);
        let (p, s) = (self.get(dst), self.get(src));
        self.emit(ty, Op::AtomicRmw(op.rv(), p, s, order.rv()))
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        let order = match scope {
            SynchronizationScope::SingleThread => None,
            SynchronizationScope::CrossThread => Some(order.rv()),
        };
        self.emit0(Op::Fence(order));
    }

    fn set_invariant_load(&mut self, _load: Val) {}

    fn lifetime_start(&mut self, _ptr: Val, _size: Size) {}

    fn lifetime_end(&mut self, _ptr: Val, _size: Size) {}

    fn call(
        &mut self,
        llty: Ty,
        _caller_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, ty::Ty<'tcx>>>,
        fn_val: Val,
        args: &[Val],
        _funclet: Option<&()>,
        _callee_instance: Option<Instance<'tcx>>,
    ) -> Val {
        let (ret, _, _) = self.sig(llty);
        let (ret_attrs, attrs) = fn_abi.map(|a| self.fn_sig(a)).map_or_else(Default::default, |(_, r, a)| (r, a));
        let callee = match fn_val {
            Val::Func(f) => Callee::Direct(self.module.borrow().functions[f.0 as usize].name.clone()),
            v => Callee::Indirect(self.get(v)),
        };
        let args = args.iter().enumerate().map(|(i, &a)| (self.get(a), attrs.get(i).cloned().unwrap_or_default())).collect();
        let op = Op::Call { fty: llty, callee, args, ret: ret_attrs };
        if ret == Types::VOID {
            self.emit0(op);
            return self.none();
        }
        self.emit(ret, op)
    }

    fn tail_call(
        &mut self,
        llty: Ty,
        caller_attrs: Option<&CodegenFnAttrs>,
        fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>,
        llfn: Val,
        args: &[Val],
        funclet: Option<&()>,
        callee_instance: Option<Instance<'tcx>>,
    ) {
        let r = self.call(llty, caller_attrs, Some(fn_abi), llfn, args, funclet, callee_instance);
        if self.sig(llty).0 == Types::VOID { self.ret_void() } else { self.ret(r) }
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, _llret: Val) {}
}

impl<'a, 'tcx> BackendTypes for Builder<'a, 'tcx> {
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

impl<'a, 'tcx> HasTyCtxt<'tcx> for Builder<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl HasDataLayout for Builder<'_, '_> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.cx.data_layout()
    }
}

impl HasTargetSpec for Builder<'_, '_> {
    fn target_spec(&self) -> &rustc_target::spec::Target {
        self.cx.target_spec()
    }
}

impl HasX86AbiOpt for Builder<'_, '_> {
    fn x86_abi_opt(&self) -> X86Abi {
        self.cx.x86_abi_opt()
    }
}

impl<'a, 'tcx> HasTypingEnv<'tcx> for Builder<'a, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl<'a, 'tcx> LayoutOfHelpers<'tcx> for Builder<'a, 'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: ty::Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'a, 'tcx> FnAbiOfHelpers<'tcx> for Builder<'a, 'tcx> {
    fn handle_fn_abi_err(&self, err: FnAbiError<'tcx>, span: Span, req: FnAbiRequest<'tcx>) -> ! {
        self.cx.handle_fn_abi_err(err, span, req)
    }
}

impl<'a, 'tcx> AbiBuilderMethods for Builder<'a, 'tcx> {
    fn get_param(&mut self, index: usize) -> Val {
        let m = self.cx.module.borrow();
        Val::Local(self.blk.f, m.functions[self.blk.f.0 as usize].blocks[0].params[index])
    }
}

impl<'a, 'tcx> StaticBuilderMethods for Builder<'a, 'tcx> {
    fn get_static(&mut self, def_id: rustc_hir::def_id::DefId) -> Val {
        Val::Global(self.cx.get_static_global(def_id))
    }
}

impl<'a, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn add_coverage(&mut self, _instance: Instance<'tcx>, _kind: &rustc_middle::mir::coverage::CoverageKind) {}
}

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[rustc_ast::InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        _options: rustc_ast::InlineAsmOptions,
        line_spans: &[Span],
        _instance: Instance<'_>,
        dest: Option<Blk>,
        _catch_funclet: Option<(Blk, Option<&()>)>,
    ) {
        // 空模板、没有输出（编译器屏障一类）：riscvc 不跨它重排访存，什么都不用出
        let empty = template.iter().all(|p| matches!(p, rustc_ast::InlineAsmTemplatePiece::String(s) if s.trim().is_empty()));
        let outputs = operands.iter().any(|o| matches!(o, InlineAsmOperandRef::Out { .. } | InlineAsmOperandRef::InOut { .. }));
        if !empty || outputs {
            self.tcx.dcx().span_fatal(line_spans.first().copied().unwrap_or_default(), "riscvc 后端不支持内联汇编（只认空模板）");
        }
        if let Some(dest) = dest {
            self.br(dest);
        }
    }
}

impl<'a, 'tcx> DebugInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn dbg_scope_fn(&mut self, _instance: Instance<'tcx>, _fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>, _llfn: Option<Fid>) {}
    fn dbg_create_lexical_block(&mut self, _pos: rustc_span::BytePos, _parent: ()) {}
    fn dbg_location_clone_with_discriminator(&mut self, _loc: (), _discriminator: u32) -> Option<()> {
        None
    }
    fn dbg_loc(&mut self, _scope: (), _inlined_at: Option<()>, _span: Span) {}
    fn extend_scope_to_file(&mut self, _scope: (), _file: &rustc_span::SourceFile) {}
    fn create_dbg_var(&mut self, _name: rustc_span::Symbol, _ty: ty::Ty<'tcx>, _scope: (), _kind: rustc_codegen_ssa::mir::debuginfo::VariableKind, _span: Span) {}
    fn dbg_var_addr(&mut self, _var: (), _loc: (), _alloca: Val, _direct: Size, _indirect: &[Size], _fragment: &Option<std::ops::Range<Size>>) {}
    fn dbg_var_value(&mut self, _var: (), _loc: (), _value: Val, _direct: Size, _indirect: &[Size], _fragment: &Option<std::ops::Range<Size>>) {}
    fn set_dbg_loc(&mut self, _loc: ()) {}
    fn clear_dbg_loc(&mut self) {}
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {}
    fn set_var_name(&mut self, _value: Val, _name: &str) {}
}
