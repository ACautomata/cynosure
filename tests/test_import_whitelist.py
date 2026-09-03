"""零依赖固化的静态检查（spec 测试面 #7）：对源码 import 语句做包白名单
断言（仅 MONAI/torch/标准库等），本包不 import NV-Generate-CTMR。

属静态架构检查、CLI 行为测试的唯一例外：NV-Generate-CTMR 不在运行环境内，
运行时 import 探测无意义，故用 ast 静态扫描。"""

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cynosure"
ALLOWED_THIRD_PARTY = frozenset({"monai", "torch", "numpy", "pydantic", "einops"})
WHITELIST = frozenset(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY | {"cynosure"}


def top_level_modules(path: Path) -> set[str]:
    """ast 解析单个源文件的顶层 import 包名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def all_source_modules() -> set[str]:
    return {
        module
        for source in sorted(SRC_ROOT.rglob("*.py"))
        for module in top_level_modules(source)
    }


class TestZeroDependency:
    def test_all_source_imports_within_whitelist(self) -> None:
        violations = all_source_modules() - WHITELIST
        assert violations == set(), f"白名单外的 import: {sorted(violations)}"

    def test_nv_generate_ctmr_is_rejected_by_construction(self, tmp_path: Path) -> None:
        """负面用例：基座仓库的任何 import 形态都落在白名单之外。"""
        for statement in (
            "import ctmr",
            "from ctmr.infrastructure import diff_model_infer",
            "import nv_generate_ctmr",
        ):
            probe = tmp_path / "probe.py"
            probe.write_text(statement, encoding="utf-8")
            imported = top_level_modules(probe)
            assert imported - WHITELIST, f"{statement} 应被白名单拒绝"

    def test_no_relative_imports(self) -> None:
        """包内一律绝对导入（cynosure.xxx），不留相对导入。"""
        offenders = []
        for source in sorted(SRC_ROOT.rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    offenders.append(f"{source.relative_to(SRC_ROOT)}:{node.lineno}")
        assert offenders == [], f"相对导入: {offenders}"

    def test_whitelist_does_not_silently_grow(self) -> None:
        """第三方白名单变更必须是显式决定：钉死清单本体。"""
        assert ALLOWED_THIRD_PARTY == {"monai", "torch", "numpy", "pydantic", "einops"}
