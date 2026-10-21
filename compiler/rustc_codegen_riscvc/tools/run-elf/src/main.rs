//! `run-elf <app.elf> <march>`：模拟器从 `0x10000`（selfcheck.sh 链接脚本的入口）跑到停下。
//! 停在 `ebreak`（启动代码的结尾）⇒ 打印 a0、按 a0 退出；别的停法 ⇒ 打印停法与 pc，退出码 255。
use riscvc::sim::{Cpu, Stop};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, elf, march] = &args[..] else {
        eprintln!("用法：run-elf <app.elf> <march>");
        std::process::exit(2);
    };
    let elf = std::fs::read(elf).expect("读 ELF");
    let mut cpu = Cpu { target: riscvc::feature::Target::parse(march).expect("目标"), ..Cpu::default() };
    cpu.load(&elf).expect("装载");
    match cpu.run(0x10000, 0x0fff_fff0, 200_000_000) {
        Stop::Trap(0x0010_0073) => {
            println!("a0 = {}（{} 步）", cpu.x[10], cpu.steps);
            std::process::exit(i32::try_from(cpu.x[10].min(254)).unwrap_or(254));
        }
        stop => {
            println!("没跑到 ebreak：{stop:?}，pc = {:#x}（{} 步）", cpu.pc, cpu.steps);
            std::process::exit(255);
        }
    }
}
