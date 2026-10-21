//! 后端自己落的 intrinsic。rustc_codegen_ssa 已经处理了与后端无关的那些（size_of、copy、原子操作…）；
//! 这里认得的落成 riscvc 的 `Intr` / 库调用，其余有回退实现的交回 rustc（`IntrinsicResult::Fallback`）。
use riscvc::ir::{ArgAttrs, Callee, Intr, Kind, Op, Types};
use rustc_codegen_ssa::RetagInfo;
use rustc_codegen_ssa::mir::IntrinsicResult;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceValue;
use rustc_codegen_ssa::common::IntPredicate;
use rustc_codegen_ssa::traits::*;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::Instance;
use rustc_span::Span;

use crate::builder::Builder;
use crate::context::Val;

/// 浮点 intrinsic 的名字 ⇒ riscvc 的 `Intr`：f32 / f64 两个版本同一个落法（f16 / f128 走回退）。
trait FloatIntr: Sized {
    fn float(name: &str) -> Option<Self>;
}

impl FloatIntr for Intr {
    fn float(name: &str) -> Option<Self> {
        // `sqrtf32` / `sqrtf64`；泛型的（`fabs<T>`）没有后缀，类型由调用处查
        let base = name.strip_suffix("f32").or_else(|| name.strip_suffix("f64")).unwrap_or(name);
        let base = base.strip_suffix('_').unwrap_or(base);
        Some(match base {
            "sqrt" => Intr::Sqrt,
            "fabs" => Intr::Fabs,
            "copysign" => Intr::Copysign,
            "fma" => Intr::Fma,
            "fmuladd" => Intr::FMulAdd,
            "minnum" => Intr::MinNum,
            "maxnum" => Intr::MaxNum,
            "round_ties_even" => Intr::Libm(String::from("roundeven")),
            "floor" | "ceil" | "trunc" | "round" | "pow" | "exp" | "exp2" | "log" | "log2" | "log10" | "sin" | "cos" | "tan" | "powi" => Intr::Libm(String::from(base)),
            _ => return None,
        })
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// 位运算类：在实参的类型上算，结果按 intrinsic 的返回类型截 / 扩（ctlz 等总回 u32）。
    fn bit_intr(&mut self, i: Intr, x: Val, result: TyAndLayout<'tcx>) -> Val {
        let t = self.val_ty(x);
        let r = self.intr(t, i, &[x]);
        let rt = self.immediate_backend_type(result);
        self.intcast(r, rt, false)
    }

    /// `memcmp(a, b, n)`。
    fn memcmp(&mut self, a: Val, b: Val, n: Val) -> Val {
        let fty = self.type_func(&[Types::PTR, Types::PTR, Types::I32], Types::I32);
        let f = self.declare("memcmp", fty, ArgAttrs::default(), Vec::new());
        self.call(fty, None, None, Val::Func(f), &[a, b, n], None, None)
    }
}

impl<'a, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        args: &[OperandRef<'tcx, Val>],
        result: TyAndLayout<'tcx>,
        _result_place: Option<PlaceValue<Val>>,
        span: Span,
    ) -> IntrinsicResult<'tcx, Val> {
        let name = self.tcx.item_name(instance.def_id());
        let name = name.as_str();
        let imm = |i: usize| args[i].immediate();
        let v = match name {
            "ctlz" | "ctlz_nonzero" => self.bit_intr(Intr::Ctlz, imm(0), result),
            "cttz" | "cttz_nonzero" => self.bit_intr(Intr::Cttz, imm(0), result),
            "ctpop" => self.bit_intr(Intr::Ctpop, imm(0), result),
            "bswap" => self.bit_intr(Intr::Bswap, imm(0), result),
            "bitreverse" => self.bit_intr(Intr::Bitreverse, imm(0), result),
            "rotate_left" | "rotate_right" => {
                let (x, t) = (imm(0), self.val_ty(imm(0)));
                let n = self.intcast(imm(1), t, false);
                self.intr(t, if name == "rotate_left" { Intr::Fshl } else { Intr::Fshr }, &[x, x, n])
            }
            "saturating_add" | "saturating_sub" => {
                let signed = args[0].layout.ty.is_signed();
                let i = match (name, signed) {
                    ("saturating_add", true) => Intr::SAddSat,
                    ("saturating_add", false) => Intr::UAddSat,
                    (_, true) => Intr::SSubSat,
                    (_, false) => Intr::USubSat,
                };
                let t = self.val_ty(imm(0));
                self.intr(t, i, &[imm(0), imm(1)])
            }
            "ptr_mask" => self.intr(Types::PTR, Intr::PtrMask, &[imm(0), imm(1)]),
            "is_val_statically_known" => self.const_bool(false),
            "black_box" => return IntrinsicResult::Operand(args[0].val),
            "prefetch_read_data" | "prefetch_write_data" | "prefetch_read_instruction" | "prefetch_write_instruction" => {
                return IntrinsicResult::Operand(OperandValue::ZeroSized);
            }
            "volatile_load" | "unaligned_volatile_load" if self.is_backend_immediate(result) => {
                let align = if name == "volatile_load" { result.align.abi } else { rustc_abi::Align::ONE };
                let v = self.volatile_load(self.backend_type(result), imm(0), align);
                match result.backend_repr {
                    rustc_abi::BackendRepr::Scalar(s) => self.to_immediate_scalar(v, s),
                    _ => v,
                }
            }
            "compare_bytes" => self.memcmp(imm(0), imm(1), imm(2)),
            "raw_eq" => {
                let size = self.layout_of(instance.args.type_at(0)).size.bytes();
                if size == 0 {
                    self.const_bool(true)
                } else {
                    let n = self.const_usize(size);
                    let r = self.memcmp(imm(0), imm(1), n);
                    let zero = self.const_i32(0);
                    self.icmp(IntPredicate::IntEQ, r, zero)
                }
            }
            // panic = abort：只调 try_fn(data)，永远「没有 panic」
            "catch_unwind" => {
                let fty = self.type_func(&[Types::PTR], Types::VOID);
                let (f, data) = (self.get(imm(0)), self.get(imm(1)));
                self.emit0(Op::Call { fty, callee: Callee::Indirect(f), args: vec![(data, ArgAttrs::default())], ret: ArgAttrs::default() });
                self.const_i32(0)
            }
            // 只有 f32 / f64 落成 riscvc 的指令 / libm 调用（f16 / f128 走回退或报错）
            _ => match Intr::float(name).filter(|_| matches!(self.kind(self.immediate_backend_type(result)), Kind::Float | Kind::Double)) {
                Some(i) => {
                    let rt = self.immediate_backend_type(result);
                    let xs: Vec<Val> = args.iter().map(|a| a.immediate()).collect();
                    self.intr(rt, i, &xs)
                }
                None => {
                    if self.tcx.intrinsic(instance.def_id()).is_some_and(|d| d.must_be_overridden) {
                        return IntrinsicResult::Err(self.tcx.dcx().span_err(span, format!("riscvc 后端还不支持 intrinsic `{name}`")));
                    }
                    // ⚠ 交回普通 Item 实例（同收集器推的那个）：intrinsic 种类的实例算出的是本地副本的符号名，
                    // 而 share-generics 下上游 crate 导出的是它自己的那份
                    return IntrinsicResult::Fallback(Instance::new_raw(instance.def_id(), instance.args));
                }
            },
        };
        IntrinsicResult::Operand(OperandValue::Immediate(v))
    }

    fn codegen_llvm_intrinsic_call(&mut self, instance: Instance<'tcx>, _args: &[OperandRef<'tcx, Val>], _is_cleanup: bool) -> Val {
        let name = self.tcx.symbol_name(instance).name;
        self.tcx.dcx().fatal(format!("riscvc 后端不支持 LLVM intrinsic `{name}`"))
    }

    fn abort(&mut self) {
        self.intr(Types::VOID, Intr::Trap, &[]);
    }

    fn assume(&mut self, _val: Val) {}

    fn expect(&mut self, cond: Val, _expected: bool) -> Val {
        cond
    }

    fn type_checked_load(&mut self, llvtable: Val, vtable_byte_offset: u64, _typeid: &[u8]) -> Val {
        let off = self.const_usize(vtable_byte_offset);
        let p = self.inbounds_ptradd(llvtable, off);
        let align = self.tcx.data_layout.pointer_align().abi;
        self.load(Types::PTR, p, align)
    }

    fn va_start(&mut self, val: Val) {
        self.intr(Types::VOID, Intr::VaStart, &[val]);
    }

    fn retag_mem(&mut self, _place: Val, _info: &RetagInfo<Val>) {}

    fn retag_reg(&mut self, ptr: Val, _info: &RetagInfo<Val>) -> Val {
        ptr
    }
}
