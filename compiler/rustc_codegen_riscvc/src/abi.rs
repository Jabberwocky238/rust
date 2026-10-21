//! FnAbi ⇒ riscvc 的函数类型与形参属性（形参的拆法同 cg_llvm：Pair 拆两个、Indirect 传地址、Cast 按寄存器类型）。
use std::cmp;

use riscvc::ir::{ArgAttrs, Kind, Ty, Types};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::*;
use rustc_middle::bug;
use rustc_middle::ty::{self, layout::LayoutOf};
use rustc_target::callconv::{ArgAbi, ArgAttributes, ArgExtension, FnAbi, PassMode};

use crate::builder::Builder;
use crate::context::{CodegenCx, Val};

/// rustc 的实参属性 ⇒ riscvc 的（只取影响代码生成的：符号 / 零扩展、对齐）。
trait IntoRv {
    fn rv(&self) -> ArgAttrs;
}

impl IntoRv for ArgAttributes {
    fn rv(&self) -> ArgAttrs {
        ArgAttrs {
            signext: self.arg_ext == ArgExtension::Sext,
            zeroext: self.arg_ext == ArgExtension::Zext,
            align: self.pointee_align.map(|a| a.bytes()),
            ..ArgAttrs::default()
        }
    }
}

impl<'tcx> CodegenCx<'tcx> {
    /// （函数类型, 返回值属性, 各形参属性）。形参属性与函数类型里的形参一一对应。
    pub fn fn_sig(&self, fn_abi: &FnAbi<'tcx, ty::Ty<'tcx>>) -> (Ty, ArgAttrs, Vec<ArgAttrs>) {
        let args = if fn_abi.c_variadic { &fn_abi.args[..fn_abi.fixed_count as usize] } else { &fn_abi.args[..] };
        let mut tys = Vec::with_capacity(args.len() + 1);
        let mut attrs = Vec::with_capacity(args.len() + 1);
        let mut ret_attrs = ArgAttrs::default();
        let ret = match &fn_abi.ret.mode {
            PassMode::Ignore => Types::VOID,
            PassMode::Direct(a) => {
                ret_attrs = a.rv();
                self.immediate_backend_type(fn_abi.ret.layout)
            }
            PassMode::Pair(..) => self.immediate_backend_type(fn_abi.ret.layout),
            PassMode::Cast { cast, .. } => self.cast_backend_type(cast),
            PassMode::Indirect { attrs: a, .. } => {
                tys.push(Types::PTR);
                attrs.push(ArgAttrs { sret: true, ..a.rv() });
                Types::VOID
            }
        };
        for arg in args {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(a) => {
                    tys.push(self.immediate_backend_type(arg.layout));
                    attrs.push(a.rv());
                }
                PassMode::Pair(a, b) => {
                    tys.extend([self.scalar_pair_element_backend_type(arg.layout, 0, true), self.scalar_pair_element_backend_type(arg.layout, 1, true)]);
                    attrs.extend([a.rv(), b.rv()]);
                }
                // 不定长（切片 / dyn）按值传：数据指针 + 元数据
                PassMode::Indirect { meta_attrs: Some(_), .. } => {
                    let ptr = self.layout_of(ty::Ty::new_mut_ptr(self.tcx, arg.layout.ty));
                    tys.extend([self.scalar_pair_element_backend_type(ptr, 0, true), self.scalar_pair_element_backend_type(ptr, 1, true)]);
                    attrs.extend([ArgAttrs::default(), ArgAttrs::default()]);
                }
                PassMode::Indirect { attrs: a, meta_attrs: None, on_stack } => {
                    tys.push(Types::PTR);
                    attrs.push(ArgAttrs { byval: on_stack.then(|| self.backend_type(arg.layout)), ..a.rv() });
                }
                PassMode::Cast { cast, pad_i32 } => {
                    if *pad_i32 {
                        tys.push(Types::I32);
                        attrs.push(ArgAttrs::default());
                    }
                    tys.push(self.cast_backend_type(cast));
                    attrs.push(ArgAttrs::default());
                }
            }
        }
        (self.ty(Kind::Func(ret, tys, fn_abi.c_variadic)), ret_attrs, attrs)
    }
}

impl<'a, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn store_fn_arg(&mut self, arg_abi: &ArgAbi<'tcx, ty::Ty<'tcx>>, idx: &mut usize, dst: PlaceRef<'tcx, Val>) {
        let mut next = |bx: &mut Self| {
            let v = bx.get_param(*idx);
            *idx += 1;
            v
        };
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                let (a, b) = (next(self), next(self));
                OperandValue::Pair(a, b).store(self, dst);
            }
            PassMode::Indirect { meta_attrs: Some(_), .. } => bug!("riscvc：不定长的 ArgAbi 不能存"),
            PassMode::Direct(_) | PassMode::Indirect { meta_attrs: None, .. } | PassMode::Cast { .. } => {
                let v = next(self);
                self.store_arg(arg_abi, v, dst);
            }
        }
    }

    fn store_arg(&mut self, arg_abi: &ArgAbi<'tcx, ty::Ty<'tcx>>, val: Val, dst: PlaceRef<'tcx, Val>) {
        match &arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Indirect { attrs, meta_attrs: None, .. } => {
                let align = attrs.pointee_align.unwrap_or(arg_abi.layout.align.abi);
                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(self, dst);
            }
            PassMode::Indirect { meta_attrs: Some(_), .. } => bug!("riscvc：不定长的 ArgAbi 不能存"),
            // 按寄存器类型传进来的：先落到一块按 cast 对齐的暂存，再拷实际大小
            PassMode::Cast { cast, .. } => {
                let (size, align) = (cast.size(self), cast.align(self));
                let copy = cmp::min(cast.unaligned_size(self).bytes(), arg_abi.layout.size.bytes());
                let scratch = self.alloca(size, align);
                rustc_codegen_ssa::mir::store_cast(self, cast, val, scratch, align);
                let n = self.const_usize(copy);
                self.memcpy(dst.val.llval, arg_abi.layout.align.abi, scratch, align, n, MemFlags::empty(), None);
            }
            PassMode::Pair(..) | PassMode::Direct(_) => {
                OperandRef::from_immediate_or_packed_pair(self, val, arg_abi.layout).val.store(self, dst);
            }
        }
    }
}
