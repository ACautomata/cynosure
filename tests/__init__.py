"""tests 包标记。

测试模块以包路径互导（``from tests.conftest import ...`` /
``from tests.test_train_loop import ...``）：本文件使 ``tests`` 成为常规
包，pytest prepend 导入模式据此把仓库根插入 ``sys.path``——裸 ``pytest``
（仓库根目录、console script 入口）下这些导入才成立。删除本文件会让
 collection 阶段全部 ``ModuleNotFoundError: No module named 'tests'``。
"""
