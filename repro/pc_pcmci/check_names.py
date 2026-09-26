# -*- coding: utf-8 -*-
"""粗略的未定义名检查：找 Name(Load) 但全文件从未被绑定的标识符

用法: python _undef_check.py <文件...>
用途: 改动后做一次静态兜底（本地无 causallearn，跑不起来完整流程）
"""
import ast
import builtins
import sys


def bound_names(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            a = getattr(n, 'args', None)
            if a:
                for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                    names.add(arg.arg)
                if a.vararg:
                    names.add(a.vararg.arg)
                if a.kwarg:
                    names.add(a.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            a = n.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                names.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                names.add((al.asname or al.name).split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
    return names


def check(path):
    src = open(path, encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [('SYNTAX', str(e), 0)]
    b = bound_names(tree) | set(dir(builtins)) | {'__file__', '__name__', '__doc__', 'self', 'cls'}
    return [(n.id, n.lineno) for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in b]


total = 0
for path in sys.argv[1:]:
    res = check(path)
    if res:
        total += len(res)
        print('== %s' % path)
        seen = {}
        for name, line in res:
            seen.setdefault(name, []).append(line)
        for name, lines in seen.items():
            print('   %-30s 行 %s' % (name, lines[:8]))
print('\n可疑未定义名合计: %d' % total)
