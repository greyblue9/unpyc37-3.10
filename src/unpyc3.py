#!/usr/bin/env python3

"""
Decompiler for Python3.7.
Decompile a module or a function using the decompile() function

>>> from unpyc3 import decompile
>>> def foo(x, y, z=3, *args):
...    global g
...    for i, j in zip(x, y):
...        if z == i + j or args[i] == j:
...            g = i, j
...            return
...    
>>> print(decompile(foo))

def foo(x, y, z=3, *args):
    global g
    for i, j in zip(x, y):
        if z == i + j or args[i] == j:
            g = i, j
            return
>>>
"""
from __future__ import annotations

__package__ = "unpyc"
import sys
import traceback
from pathlib import Path
from types import ModuleType
nspkg = ModuleType(__package__)
nspkg.__package__ = __package__
nspkg.__name__ = __package__
nspkg.__path__ = [
    Path(__file__).parent.absolute().as_posix()
]
sys.modules[__package__] = nspkg

sys.modules[f"[__package__].{__name__}"] = __import__(__name__)
from typing import Union, Iterable, Any, List
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import opcode

for name in ("opcodes",):
    path = (
        Path(__file__).parent / f"{name}.py"
    ).absolute().as_posix()
    loader = SourceFileLoader(
        f"{__package__}.{name}",
        path,
    )
    module = module_from_spec(
        spec_from_loader(
            f"{__package__}.{name}",
            loader=loader,
            origin=path,
            is_package=False
        )
    )
    loader.exec_module(module)
    globals().update({name: module})

opnames = {}
opmap = {}

for opc, opname in enumerate(opcode.opname):
    opmap[opname] = opc
    opnames[opc] = opname

for opname, code in opmap.items():
    if not opname.isidentifier():
        continue
    # sys.stderr.write(f'Adding opcode {opname=} -> {code=}\x0a')
    exec(f'{opname} = {code}')
    # sys.stderr.flush()

def missing_var(name):
    if name not in globals():
        globals()[name] = -1

missing_var("END_FINALLY")
missing_var("BREAK_LOOP")

# __all__ = ['decompile']



def set_trace(trace_function):
    global current_trace
    current_trace = trace_function if trace_function else _trace


def get_trace():
    global current_trace
    return None if current_trace == _trace else current_trace


def trace(*args):
    global current_trace
    if current_trace:
        current_trace(*args)


def _trace(*args):
    pass


current_trace = _trace

IS_NOT_310 = sys.version_info < (3, 10)

# TODO:
# - Support for keyword-only arguments
# - Handle assert statements better
# - (Partly done) Nice spacing between function/class declarations

import dis
from array import array
from opcode import opname, opmap, HAVE_ARGUMENT, cmp_op
import inspect

import struct
import sys

# Masks for code object's co_flag attribute
VARARGS = 4
VARKEYWORDS = 8

# Put opcode names in the global namespace
for name, val in opmap.items():
    globals()[name] = val

# These opcodes will generate a statement. This is used in the first
# pass (in Code.find_else) to find which POP_JUMP_IF_* instructions
# are jumps to the else clause of an if statement
stmt_opcodes = {1, 130, 137, 138, 143, 80, 83, 86, 87, 88, 89, 90, 91, 95, 96, 97, 98, 108, 109, 119, 120, 121, 122, 125, 126}

# Conditional branching opcode that make up if statements and and/or
# expressions
pop_jump_if_opcodes = (POP_JUMP_IF_TRUE, POP_JUMP_IF_FALSE)

# These opcodes indicate that a pop_jump_if_x to the address just
# after them is an else-jump
else_jump_opcodes = (
    JUMP_FORWARD,
    RETURN_VALUE,
    JUMP_ABSOLUTE,
    RAISE_VARARGS,
    POP_TOP,
)
if "SETUP_LOOP" in globals():
    else_jump_opcodes += (
        SETUP_LOOP,
    )
# These opcodes indicate for loop rather than while loop
for_jump_opcodes = (GET_ITER, FOR_ITER, GET_ANEXT)


def read_code(stream):
    # This helper is needed in order for the PEP 302 emulation to
    # correctly handle compiled files
    # Note: stream must be opened in "rb" mode
    import marshal

    if sys.version_info < (3, 4):
        import imp

        runtime_magic = imp.get_magic()
    else:
        import importlib.util

        runtime_magic = importlib.util.MAGIC_NUMBER

    magic = stream.read(4)
    if magic != runtime_magic:
        print("*** Warning: file has wrong magic number ***")

    flags = 0
    if sys.version_info >= (3, 7):
        flags = struct.unpack('i', stream.read(4))[0]

    if flags & 1:
        stream.read(4)
        stream.read(4)
    else:
        stream.read(4)  # Skip timestamp
        if sys.version_info >= (3, 3):
            stream.read(4)  # Skip rawsize
            return marshal.load(stream)


def dec_module(path) -> Suite:
    if path.endswith(".py"):
        if sys.version_info < (3, 6):
            import imp

            path = imp.cache_from_source(path)
        else:
            import importlib.util

            path = importlib.util.cache_from_source(path)
    elif not path.endswith(".pyc") and not path.endswith(".pyo"):
        raise ValueError("path must point to a .py or .pyc file")
    with open(path, "rb") as stream:
        code_obj = read_code(stream)
        code = Code(code_obj)
        return code.get_suite(
            include_declarations=False, look_for_docstring=True
        )


def decompile(obj) -> Union[Suite, PyStatement]:
    """
    Decompile obj if it is a module object, a function or a
    code object. If obj is a string, it is assumed to be the path
    to a python module.
    """
    if isinstance(obj, str):
        return dec_module(obj)
    if inspect.iscode(obj):
        code = Code(obj)
        return code.get_suite()
    if inspect.isfunction(obj):
        code = Code(obj.__code__)
        defaults = obj.__defaults__
        kwdefaults = obj.__kwdefaults__
        return DefStatement(
            code, defaults, kwdefaults, obj.__closure__
        )
    elif inspect.ismodule(obj):
        return dec_module(obj.__file__)
    else:
        msg = "Object must be string, module, function or code object"
        raise TypeError(msg)


class Indent:
    def __init__(self, indent_level=0, indent_step=4):
        self.level = indent_level
        self.step = indent_step

    def write(self, pattern, *args, **kwargs):
        if args or kwargs:
            pattern = pattern.format(*args, **kwargs)
        return self.indent(pattern)

    def __add__(self, indent_increase):
        return type(self)(self.level + indent_increase, self.step)


class IndentPrint(Indent):
    def indent(self, string):
        print(" " * self.step * self.level + string)


class IndentString(Indent):
    def __init__(self, indent_level=0, indent_step=4, lines=None):
        Indent.__init__(self, indent_level, indent_step)
        if lines is None:
            self.lines = []
        else:
            self.lines = lines

    def __add__(self, indent_increase):
        return type(self)(
            self.level + indent_increase, self.step, self.lines
        )

    def sep(self):
        if not self.lines or self.lines[-1]:
            self.lines.append("")

    def indent(self, string):
        self.lines.append(" " * self.step * self.level + string)

    def __str__(self):
        return "\n".join(self.lines)


class Stack:
    def __init__(self):
        self._stack = []
        self._counts = {}

    def __bool__(self):
        return bool(self._stack)

    def __len__(self):
        return len(self._stack)

    def __contains__(self, val):
        return self.get_count(val) > 0

    def get_count(self, obj):
        return self._counts.get(id(obj), 0)

    def set_count(self, obj, val):
        if val:
            self._counts[id(obj)] = val
        else:
            del self._counts[id(obj)]

    def pop1(self):
        val = None
        if self._stack:
            val = self._stack.pop()
        else:
            raise Exception('Empty stack popped!')
        self.set_count(val, self.get_count(val) - 1)
        return val

    def pop(self, count=None):
        if count is None:
            val = self.pop1()
            return val
        else:
            vals = [self.pop1() for i in range(count)]
            vals.reverse()
            return vals

    def push(self, *args):
        for val in args:
            self.set_count(val, self.get_count(val) + 1)
            self._stack.append(val)

    def peek(self, count=None):
        if count is None:
            return self._stack[-1]
        else:
            return self._stack[-count:]

    def __getitem__(self, x):
        res = Stack()
        res._stack = self._stack[x]
        _counts = {}
        for val in res._stack:
            _counts[id(val)] = _counts.get(id(val), 0) + 1
        res._counts = _counts
        return res


def code_walker(code):
    l = len(code)
    code = array('B', code)
    oparg = 0
    i = 0
    extended_arg = 0

    while i < l:
        op = code[i]
        offset = 1
        if sys.version_info >= (3, 6):
            oparg = code[i + offset]
            offset += 1
        elif op >= HAVE_ARGUMENT:
            oparg = (
                code[i + offset]
                + code[i + offset + 1] * 256
                + extended_arg
            )
            extended_arg = 0
            offset += 2
        if op == EXTENDED_ARG:
            if sys.version_info >= (3, 6):
                op = code[i + offset]
                offset += 1
                oparg <<= 8
                oparg |= code[i + offset]
                offset += 1
            else:
                extended_arg = oparg * 65536
        yield i, (op, oparg)
        i += offset


class CodeFlags(object):
    def __init__(self, cf):
        self.flags = cf

    @property
    def optimized(self):
        return self.flags & 0x1

    @property
    def new_local(self):
        return self.flags & 0x2

    @property
    def varargs(self):
        return self.flags & 0x4

    @property
    def varkwargs(self):
        return self.flags & 0x8

    @property
    def nested(self):
        return self.flags & 0x10

    @property
    def generator(self):
        return self.flags & 0x20

    @property
    def no_free(self):
        return self.flags & 0x40

    @property
    def coroutine(self):
        return self.flags & 0x80

    @property
    def iterable_coroutine(self):
        return self.flags & 0x100

    @property
    def async_generator(self):
        return self.flags & 0x200


class Code:
    def __init__(self, code_obj, parent=None):
        self.code_obj = code_obj
        self.parent = parent
        self.derefnames = [
            PyName(v)
            for v in code_obj.co_cellvars + code_obj.co_freevars
        ]
        self.consts = list(map(PyConst, code_obj.co_consts))
        self.names = list(map(PyName, code_obj.co_names))
        self.varnames = list(map(PyName, code_obj.co_varnames))
        self.instr_seq = list(code_walker(code_obj.co_code))
        self.instr_list = self.instr_seq
        self.instr_map = {
            addr: i for i, (addr, _) in enumerate(self.instr_seq)
        }
        self.name = code_obj.co_name
        self.globals = []
        self.nonlocals = []
        self.jump_targets = []
        self.find_else()
        self.find_jumps()
        trace('================================================')
        trace(self.code_obj)
        trace('================================================')
        for addr in self:
            trace(str(addr))
            if (
                addr.opcode in stmt_opcodes
                or addr.opcode in pop_jump_if_opcodes
            ):
                trace(' ')
        trace('================================================')
        self.flags: CodeFlags = CodeFlags(code_obj.co_flags)

    def __getitem__(self, instr_index):
        if 0 <= instr_index < len(self.instr_seq):
            return Address(self, instr_index)

    def __iter__(self):
        for i in range(len(self.instr_seq)):
            yield Address(self, i)

    def show(self):
        for addr in self:
            print(addr)

    def address(self, addr):
        if addr in self.instr_map:
            tgt_addr = self.instr_map[addr]
            ret = self[tgt_addr]
            return ret
        if addr in self.instr_list:
            tgt_addr = self.instr_list[addr]
            ret = self[tgt_addr]
            return ret
        return self[addr]

    def iscellvar(self, i):
        return i < len(self.code_obj.co_cellvars)

    def find_jumps(self):
        for addr in self:
            opcode, arg = addr
            jt = addr.jump()
            if jt:
                self.jump_targets.append(jt)

    def find_else(self):
        jumps = {}
        last_jump = None
        for addr in self:
            opcode, arg = addr
            if opcode in pop_jump_if_opcodes:
                # 3.10 needs a doubled arg (e.g. 14) but any
                # version lower does not (e.g. 28)
                jump_addr = self.address(arg * (2 - IS_NOT_310) - 2)
                if (
                    jump_addr.opcode in else_jump_opcodes
                    or jump_addr.opcode is FOR_ITER
                ):
                    last_jump = addr
                    jumps[jump_addr] = addr
            elif opcode == JUMP_ABSOLUTE:
                # This case is to deal with some nested ifs such as:
                # if a:
                # if b:
                #         f()
                #     elif c:
                #         g()
                jump_addr = self.address(arg)
                if jump_addr in jumps:
                    jumps[addr] = jumps[jump_addr]
            elif opcode == JUMP_FORWARD:
                jump_addr = addr[1] + arg
                if jump_addr in jumps:
                    jumps[addr] = jumps[jump_addr]
            elif opcode in stmt_opcodes and last_jump is not None:
                # This opcode will generate a statement, so it means
                # that the last POP_JUMP_IF_x was an else-jump
                jumps[addr] = last_jump
        self.else_jumps = set(jumps.values())

    def get_suite(
        self, include_declarations=True, look_for_docstring=False
    ) -> Suite:
        dec = SuiteDecompiler(self[0])
        dec.run()
        first_stmt = dec.suite and dec.suite[0]
        # Change __doc__ = "docstring" to "docstring"
        if look_for_docstring and isinstance(
            first_stmt, AssignStatement
        ):
            chain = first_stmt.chain
            if len(chain) == 2 and str(chain[0]) == "__doc__":
                dec.suite[0] = DocString(first_stmt.chain[1].val)
        if include_declarations and (self.globals or self.nonlocals):
            suite = Suite()
            if self.globals:
                stmt = "global " + ", ".join(map(str, self.globals))
                suite.add_statement(SimpleStatement(stmt))
            if self.nonlocals:
                stmt = "nonlocal " + ", ".join(
                    map(str, self.nonlocals)
                )
                suite.add_statement(SimpleStatement(stmt))
            for stmt in dec.suite:
                suite.add_statement(stmt)
            return suite
        else:
            return dec.suite

    def declare_global(self, name):
        """
        Declare name as a global.  Called by STORE_GLOBAL and
        DELETE_GLOBAL
        """
        if name not in self.globals:
            self.globals.append(name)

    def ensure_global(self, name):
        """
        Declare name as global only if it is also a local variable
        name in one of the surrounding code objects.  This is called
        by LOAD_GLOBAL
        """
        parent = self.parent
        while parent:
            if name in parent.varnames:
                return self.declare_global(name)
            parent = parent.parent

    def declare_nonlocal(self, name):
        """
        Declare name as nonlocal.  Called by STORE_DEREF and
        DELETE_DEREF (but only when the name denotes a free variable,
        not a cell one).
        """
        if name not in self.nonlocals:
            self.nonlocals.append(name)


class Address:
    def __init__(self, code, instr_index):
        self.code = code
        self.index = instr_index
        self.addr, (self.opcode, self.arg) = code.instr_seq[
            instr_index
        ]

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.code == other.code
            and self.index == other.index
        )

    def __lt__(self, other):
        return other is None or (
            isinstance(other, type(self))
            and self.code == other.code
            and self.index < other.index
        )

    def __str__(self):
        mark = "* " if self in self.code.else_jumps else "  "
        jump = self.jump()
        jt = '>>' if self.is_jump_target else '  '
        arg = self.arg or "  "
        jdest = (
            '\t(to {})'.format(jump.addr)
            if jump and jump.addr != self.arg
            else ''
        )
        val = ''
        op = opname[self.opcode].ljust(18, ' ')
        try:

            val = (
                self.code.globals[self.arg]
                and self.arg + 1 < len(self.code.globals)
                if 'GLOBAL' in op
                else self.code.names[self.arg]
                if 'ATTR' in op
                else self.code.names[self.arg]
                if 'NAME' in op
                else self.code.names[self.arg]
                if 'LOAD_METHOD' in op
                else self.code.consts[self.arg]
                if 'CONST' in op
                else self.code.varnames[self.arg]
                if 'FAST' in op
                else self.code.derefnames[self.arg]
                if 'DEREF' in op
                else cmp_op[self.arg]
                if 'COMPARE' in op
                else ''
            )
            if val != '':
                val = '\t({})'.format(val)
        except:
            pass

        return "{}{}\t{}\t{}\t{}{}{}".format(
            jt, mark, self.addr, op, arg, jdest, val
        )

    def __add__(self, delta):
        return self.code.address(self.addr + delta)

    def __getitem__(self, index) -> Address:
        return self.code[self.index + index]

    def __iter__(self):
        yield self.opcode
        yield self.arg

    def __hash__(self):
        return hash((self.code, self.index))

    @property
    def is_else_jump(self):
        return self in self.code.else_jumps

    @property
    def is_jump_target(self):
        return self in self.code.jump_targets

    def change_instr(self, opcode, arg=None):
        self.code.instr_seq[self.index] = (self.addr, (opcode, arg))

    def jump(self) -> Address:
        opcode = self.opcode
        if opcode in dis.hasjrel:
            return self[self.arg // (1 + IS_NOT_310) + 1]
        elif opcode in dis.hasjabs:
            return self.code.address(self.arg * (2 - IS_NOT_310))

    def seek(
        self, opcode: Iterable, increment: int, end: Address = None
    ) -> Address:
        if not isinstance(opcode, Iterable):
            opcode = (opcode,)
        a = self[increment]
        while a and a != end:
            if a.opcode in opcode:
                return a
            a = a[increment]

    def seek_back(
        self, opcode: Union[Iterable, int], end: Address = None
    ) -> Address:
        return self.seek(opcode, -1, end)

    def seek_forward(
        self, opcode: Union[Iterable, int], end: Address = None
    ) -> Address:
        return self.seek(opcode, 1, end)

    def seek_back_statement(
        self, opcode: Union[Iterable, int]
    ) -> Address:
        last_statement = self.seek_back(stmt_opcodes)
        return self.seek(opcode, -1, last_statement)

    def seek_forward_statement(
        self, opcode: Union[Iterable, int]
    ) -> Address:
        next_statement = self.seek_forward(stmt_opcodes)
        return self.seek(opcode, 1, next_statement)


class AsyncMixin:
    def __init__(self):
        self.is_async = False

    @property
    def async_prefix(self):
        return 'async ' if self.is_async else ''


class AwaitableMixin:
    def __init__(self):
        self.is_awaited = False

    @property
    def await_prefix(self):
        return 'await ' if self.is_awaited else ''


class PyExpr:
    def wrap(self, condition=True):
        if condition:
            return "({})".format(self)
        else:
            return str(self)

    def store(self, dec, dest):
        chain = dec.assignment_chain
        chain.append(dest)
        if self not in dec.stack:
            chain.append(self)
            dec.suite.add_statement(AssignStatement(chain))
            dec.assignment_chain = []

    def on_pop(self, dec: SuiteDecompiler):
        dec.write(str(self))


class PyConst(PyExpr):
    def __init__(self, val):
        self.val = val
        if isinstance(val, int):
            self.precedence = 14
        else:
            self.precedence = 100

    def __str__(self):
        return repr(self.val)

    def __iter__(self):
        return iter(self.val)

    def __eq__(self, other):
        return isinstance(other, PyConst) and self.val == other.val


class PyFormatValue(PyConst):
    def __init__(self, val):
        super().__init__(val)
        self.formatter = ''

    @staticmethod
    def fmt(string):
        return f'f\'{string}\''

    def base(self):
        return f'{{{self.val}{self.formatter}}}'

    def __str__(self):
        return self.fmt(self.base())


class PyFormatString(PyExpr):
    precedence = 100

    def __init__(self, params):
        super().__init__()
        self.params = params

    def __str__(self):
        return "f'{}'".format(
            ''.join(
                [
                    p.base().replace('\'', '\"')
                    if isinstance(p, PyFormatValue)
                    else p.name
                    if isinstance(p, PyName)
                    else str(p.val.encode('utf-8'))[1:]
                    .replace('\'', '')
                    .replace('{', '{{')
                    .replace('}', '}}')
                    for p in self.params
                ]
            )
        )


class PyTuple(PyExpr):
    precedence = 0

    def __init__(self, values):
        self.values = values

    def __str__(self):
        if not self.values:
            return "()"
        valstr = [
            val.wrap(val.precedence <= self.precedence)
            for val in self.values
        ]
        if len(valstr) == 1:
            return '(' + valstr[0] + "," + ')'
        else:
            return '(' + ", ".join(valstr) + ')'

    def __iter__(self):
        return iter(self.values)

    def wrap(self, condition=True):
        return str(self)


class PyList(PyExpr):
    precedence = 16

    def __init__(self, values):
        self.values = values

    def __str__(self):
        valstr = ", ".join(
            val.wrap(val.precedence <= 0) for val in self.values
        )
        return "[{}]".format(valstr)

    def __iter__(self):
        return iter(self.values)


class PySet(PyExpr):
    precedence = 16

    def __init__(self, values):
        self.values = values

    def __str__(self):
        valstr = ", ".join(
            val.wrap(val.precedence <= 0) for val in self.values
        )
        return "{{{}}}".format(valstr)

    def __iter__(self):
        return iter(self.values)


class PyDict(PyExpr):
    precedence = 16

    def __init__(self):
        self.items = []

    def set_item(self, key, val):
        self.items.append((key, val))

    def __str__(self):
        itemstr = ", ".join(
            f"{kv[0]}: {kv[1]}" if len(kv) == 2 else str(kv[0])
            for kv in self.items
        )
        return f"{{{itemstr}}}"


class PyName(PyExpr, AwaitableMixin):
    precedence = 100

    def __init__(self, name):
        AwaitableMixin.__init__(self)
        self.name = name

    def __str__(self):
        return f'{self.await_prefix}{self.name}'

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and self.name == other.name
        )


class PyUnaryOp(PyExpr):
    def __init__(self, operand):
        self.operand = operand

    def __str__(self):
        opstr = self.operand.wrap(
            self.operand.precedence < self.precedence
        )
        return self.pattern.format(opstr)

    @classmethod
    def instr(cls, stack):
        stack.push(cls(stack.pop()))


class PyBinaryOp(PyExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def wrap_left(self):
        return self.left.wrap(self.left.precedence < self.precedence)

    def wrap_right(self):
        return self.right.wrap(
            self.right.precedence <= self.precedence
        )

    def __str__(self):
        return self.pattern.format(
            self.wrap_left(), self.wrap_right()
        )

    @classmethod
    def instr(cls, stack):
        right = stack.pop()
        left = stack.pop()
        stack.push(cls(left, right))


class PySubscript(PyBinaryOp):
    precedence = 15
    pattern = "{}[{}]"

    def wrap_right(self):
        return str(self.right)


class PyInOp(PyBinaryOp, PyExpr):
    precedence = 15
    pattern = "{item}{_not} in {seq}"

    def __init__(self, left, right, negate):
        super(PyInOp, self).__init__(left, right)
        self.negate = negate

    def wrap_left(self):
        return self.left.wrap(self.left.precedence < self.precedence)

    def wrap_right(self):
        return self.right.wrap(
            self.right.precedence <= self.precedence
        )

    def __str__(self):
        return self.pattern.format(
            item=self.wrap_left(),
            seq=self.wrap_right(),
            _not="not " if self.negate else "",
        )


class PyIsOp(PyBinaryOp, PyExpr):
    precedence = 15
    pattern = "{} is {}{}"

    def __init__(self, left, right, negate):
        super(PyIsOp, self).__init__(left, right)
        self.negate = negate

    def wrap_left(self):
        return self.left.wrap(self.left.precedence < self.precedence)

    def wrap_right(self):
        return self.right.wrap(
            self.right.precedence <= self.precedence
        )

    def __str__(self):
        return self.pattern.format(
            self.wrap_left(),
            "not " if self.negate else "",
            self.wrap_right(),
        )


class PySlice(PyExpr):
    precedence = 1

    def __init__(self, args):
        assert len(args) in (2, 3)
        if len(args) == 2:
            self.start, self.stop = args
            self.step = None
        else:
            self.start, self.stop, self.step = args
        if self.start == PyConst(None):
            self.start = ""
        if self.stop == PyConst(None):
            self.stop = ""

    def __str__(self):
        if self.step is None:
            return "{}:{}".format(self.start, self.stop)
        else:
            return "{}:{}:{}".format(self.start, self.stop, self.step)


class PyCompare(PyExpr):
    precedence = 6

    def __init__(self, complist):
        self.complist = complist

    def __str__(self):
        return " ".join(
            x if i % 2 else x.wrap(x.precedence <= 6)
            for i, x in enumerate(self.complist)
        )

    def extends(self, other):
        if not isinstance(other, PyCompare):
            return False
        else:
            return self.complist[0] == other.complist[-1]

    def chain(self, other):
        return PyCompare(self.complist + other.complist[1:])


class PyBooleanAnd(PyBinaryOp):
    precedence = 4
    pattern = "{} and {}"


class PyBooleanOr(PyBinaryOp):
    precedence = 3
    pattern = "{} or {}"


class PyIfElse(PyExpr):
    precedence = 2

    def __init__(self, cond, true_expr, false_expr):
        self.cond = cond
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __str__(self):
        p = self.precedence
        cond_str = self.cond.wrap(self.cond.precedence <= p)
        true_str = self.true_expr.wrap(self.cond.precedence <= p)
        false_str = self.false_expr.wrap(self.cond.precedence < p)
        return "{} if {} else {}".format(
            true_str, cond_str, false_str
        )


class PyAttribute(PyExpr):
    precedence = 15

    def __init__(self, expr, attrname):
        self.expr = expr
        self.attrname = attrname

    def __str__(self):
        expr_str = self.expr.wrap(
            self.expr.precedence < self.precedence
        )
        attrname = self.attrname

        if isinstance(self.expr, PyName) and self.expr.name == 'self':
            __ = attrname.name.find('__')
            if __ > 0:
                attrname = PyName(self.attrname.name[__:])
        return "{}.{}".format(expr_str, attrname)


class PyCallFunction(PyExpr, AwaitableMixin):
    precedence = 15

    def __init__(
        self,
        func: PyAttribute,
        args: list,
        kwargs: list,
        varargs=None,
        varkw=None,
    ):
        AwaitableMixin.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.varargs = (
            varargs
            if not varargs or isinstance(varargs, Iterable)
            else {varargs}
        )
        self.varkw = (
            varkw
            if not varkw or isinstance(varkw, Iterable)
            else {varkw}
        )

    def __str__(self):
        funcstr = self.func.wrap(
            self.func.precedence < self.precedence
        )
        if (
            hasattr(self.args, '__iter__')
            and len(self.args) == 1
            and not (self.kwargs or self.varargs or self.varkw)
        ):
            arg = self.args[0]
            if isinstance(arg, PyGenExpr):
                # Only one pair of brackets arount a single arg genexpr
                return "{}{}".format(funcstr, arg)
        args = [x.wrap(x.precedence <= 0) for x in self.args]
        if self.varargs is not None:
            for varargs in self.varargs:
                args.append("*{}".format(varargs))
        args.extend(
            "{}={}".format(
                str(k).replace('\'', ''), v.wrap(v.precedence <= 0)
            )
            for k, v in self.kwargs
        )
        if self.varkw is not None:
            for varkw in self.varkw:
                args.append("**{}".format(varkw))
        return "{}{}({})".format(
            self.await_prefix, funcstr, ", ".join(args)
        )


class FunctionDefinition:
    def __init__(
        self,
        code: Code,
        defaults,
        kwdefaults,
        closure,
        paramobjs=None,
        annotations=None,
    ):
        self.code = code
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.closure = closure
        self.paramobjs = paramobjs if paramobjs else {}
        self.annotations = annotations if annotations else []

    def is_coroutine(self):
        return self.code.code_obj.co_flags & 0x100

    def getparams(self):
        code_obj = self.code.code_obj
        l = code_obj.co_argcount
        params = []
        for name in code_obj.co_varnames[:l]:
            if name in self.paramobjs:
                params.append(
                    '{}:{}'.format(name, str(self.paramobjs[name]))
                )
            else:
                params.append(name)
        if self.defaults:
            for i, arg in enumerate(reversed(self.defaults)):
                name = params[-i - 1]
                if name in self.paramobjs:
                    params[-i - 1] = "{}:{}={}".format(
                        name, str(self.paramobjs[name]), arg
                    )
                else:
                    params[-i - 1] = "{}={}".format(name, arg)
        kwcount = code_obj.co_kwonlyargcount
        kwparams = []
        if kwcount:
            for i in range(kwcount):
                name = code_obj.co_varnames[l + i]
                if name in self.kwdefaults and name in self.paramobjs:
                    kwparams.append(
                        "{}:{}={}".format(
                            name,
                            self.paramobjs[name],
                            self.kwdefaults[name],
                        )
                    )
                elif name in self.kwdefaults:
                    kwparams.append(
                        "{}={}".format(name, self.kwdefaults[name])
                    )
                else:
                    kwparams.append(name)
            l += kwcount
        if code_obj.co_flags & VARARGS:
            name = code_obj.co_varnames[l]
            if name in self.paramobjs:
                params.append(f'*{name}:{str(self.paramobjs[name])}')
            else:
                params.append(f'*{name}')
            l += 1
        elif kwparams:
            params.append("*")
        params.extend(kwparams)
        if code_obj.co_flags & VARKEYWORDS:
            name = code_obj.co_varnames[l]
            if name in self.paramobjs:
                params.append(f'**{name}:{str(self.paramobjs[name])}')
            else:
                params.append(f'**{name}')

        return params

    def getreturn(self):
        if self.paramobjs and 'return' in self.paramobjs:
            return self.paramobjs['return']
        return None


class PyLambda(PyExpr, FunctionDefinition):
    precedence = 1

    def __str__(self):
        suite = self.code.get_suite()
        params = ", ".join(self.getparams())
        if len(suite.statements) > 0:

            def strip_return(val):
                return (
                    val[len("return ") :]
                    if val.startswith('return')
                    else val
                )

            def strip_yield_none(val):
                return '(yield)' if val == 'yield None' else val

            if isinstance(suite[0], IfStatement):
                end = suite[1] if len(suite) > 1 else PyConst(None)
                expr = "{} if {} else {}".format(
                    strip_return(str(suite[0].true_suite)),
                    str(suite[0].cond),
                    strip_return(str(end)),
                )
            else:
                expr = strip_return(str(suite[0]))
                expr = strip_yield_none(expr)
        else:
            expr = "None"
        return "lambda {}: {}".format(params, expr)


class PyComp(PyExpr):
    """
    Abstraction for list, set, dict comprehensions and generator expressions
    """

    precedence = 16

    def __init__(
        self,
        code,
        defaults,
        kwdefaults,
        closure,
        paramobjs={},
        annotations=[],
    ):
        assert not defaults and not kwdefaults
        self.code = code
        code[0].change_instr(NOP)
        last_i = len(code.instr_seq) - 1
        code[last_i].change_instr(NOP)
        self.annotations = annotations

    def set_iterable(self, iterable):
        self.code.varnames[0] = iterable

    def __str__(self):
        suite = self.code.get_suite()
        return self.pattern.format(suite.gen_display())


class PyListComp(PyComp):
    pattern = "[{}]"


class PySetComp(PyComp):
    pattern = "{{{}}}"


class PyKeyValue(PyBinaryOp):
    """This is only to create dict comprehensions"""

    precedence = 1
    pattern = "{}: {}"


class PyDictComp(PyComp):
    pattern = "{{{}}}"


class PyGenExpr(PyComp):
    precedence = 16
    pattern = "({})"

    def __init__(
        self,
        code,
        defaults,
        kwdefaults,
        closure,
        paramobjs={},
        annotations=[],
    ):
        self.code = code


class PyYield(PyExpr):
    precedence = 1

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "yield {}".format(self.value)


class PyYieldFrom(PyExpr):
    precedence = 1

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "yield from {}".format(self.value)


class PyStarred(PyExpr):
    """Used in unpacking assigments"""

    precedence = 15

    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        es = self.expr.wrap(self.expr.precedence < self.precedence)
        return "*{}".format(es)


class PyListExtend(PyBinaryOp):
    precedence = 15
    pattern = "({}+{})"

    def wrap_right(self):
        return str(self.right)


class PySetUpdate(PyBinaryOp):
    precedence = 15
    pattern = "{}.update({})"

    def wrap_right(self):
        return str(self.right)


class PyDictMerge(PyBinaryOp):
    precedence = 15
    pattern = "dict(**{},**{})"

    def wrap_right(self):
        return str(self.right)


class PyDictUpdate(PyBinaryOp):
    precedence = 15
    pattern = "{}.update({})"

    def wrap_right(self):
        return str(self.right)


code_map = {
    '<lambda>': PyLambda,
    '<listcomp>': PyListComp,
    '<setcomp>': PySetComp,
    '<dictcomp>': PyDictComp,
    '<genexpr>': PyGenExpr,
}

unary_ops = [
    ('UNARY_POSITIVE', 'Positive', '+{}', 13),
    ('UNARY_NEGATIVE', 'Negative', '-{}', 13),
    ('UNARY_NOT', 'Not', 'not {}', 5),
    ('UNARY_INVERT', 'Invert', '~{}', 13),
]

binary_ops = [
    ('POWER', 'Power', '{}**{}', 14, '{} **= {}'),
    ('MULTIPLY', 'Multiply', '{}*{}', 12, '{} *= {}'),
    ('FLOOR_DIVIDE', 'FloorDivide', '{}//{}', 12, '{} //= {}'),
    ('TRUE_DIVIDE', 'TrueDivide', '{}/{}', 12, '{} /= {}'),
    ('MODULO', 'Modulo', '{} % {}', 12, '{} %= {}'),
    ('ADD', 'Add', '{} + {}', 11, '{} += {}'),
    ('SUBTRACT', 'Subtract', '{} - {}', 11, '{} -= {}'),
    ('SUBSCR', 'Subscript', '{}[{}]', 15, None),
    ('LSHIFT', 'LeftShift', '{} << {}', 10, '{} <<= {}'),
    ('RSHIFT', 'RightShift', '{} >> {}', 10, '{} >>= {}'),
    ('AND', 'And', '{} & {}', 9, '{} &= {}'),
    ('XOR', 'Xor', '{} ^ {}', 8, '{} ^= {}'),
    ('OR', 'Or', '{} | {}', 7, '{} |= {}'),
    ('MATRIX_MULTIPLY', 'MatrixMultiply', '{} @ {}', 12, '{} @= {}'),
]


class PyStatement(object):
    def __str__(self):
        istr = IndentString()
        self.display(istr)
        return str(istr)

    def wrap(self, condition=True):
        if condition:
            assert not condition
            return "({})".format(self)
        else:
            return str(self)

    def on_pop(self, dec):
        # dec.write("#ERROR: Unexpected context 'on_pop': pop on statement:  ")
        pass


class DocString(PyStatement):
    def __init__(self, string):
        self.string = string

    def display(self, indent):
        if '\n' not in self.string:
            indent.write(repr(self.string))
        else:
            if "'''" not in self.string:
                fence = "'''"
            else:
                fence = '"""'
            lines = self.string.split('\n')
            text = '\n'.join(
                l.encode('unicode_escape')
                .decode()
                .replace(fence, '\\' + fence)
                for l in lines
            )
            docstring = "{0}{1}{0}".format(fence, text)
            indent.write(docstring)


class AssignStatement(PyStatement):
    def __init__(self, chain):
        self.chain = chain

    def display(self, indent):
        indent.write(" = ".join(map(str, self.chain)))


class InPlaceOp(PyStatement):
    def __init__(self, left, right):
        self.right = right
        self.left = left

    def store(self, dec, dest):
        # assert dest is self.left
        dec.suite.add_statement(self)

    def display(self, indent):
        indent.write(self.pattern, self.left, self.right)

    @classmethod
    def instr(cls, stack):
        right = stack.pop()
        left = stack.pop()
        stack.push(cls(left, right))


class Unpack:
    precedence = 50

    def __init__(self, val, length, star_index=None):
        self.val = val
        self.length = length
        self.star_index = star_index
        self.dests = []

    def store(self, dec, dest):
        if len(self.dests) == self.star_index:
            dest = PyStarred(dest)
        self.dests.append(dest)
        if len(self.dests) == self.length:
            dec.stack.push(self.val)
            dec.store(PyTuple(self.dests))


class ImportStatement(PyStatement):
    alias = ""
    precedence = 100

    def __init__(self, name, level, fromlist):
        self.name = name
        self.alias = name
        self.level = level
        self.fromlist = fromlist
        self.aslist = []

    def store(self, dec: SuiteDecompiler, dest):
        self.alias = dest
        dec.suite.add_statement(self)

    def on_pop(self, dec):
        dec.suite.add_statement(self)

    def display(self, indent):
        if self.fromlist == PyConst(None):
            name = self.name.name
            alias = self.alias.name
            if name == alias or name.startswith(alias + "."):
                indent.write("import {}", name)
            else:
                indent.write("import {} as {}", name, alias)
        elif self.fromlist == PyConst(('*',)):
            indent.write("from {} import *", self.name.name)
        else:
            names = []
            for name, alias in zip(self.fromlist, self.aslist):
                if name == alias:
                    names.append(name)
                else:
                    names.append("{} as {}".format(name, alias))
            indent.write(
                "from {}{} import {}",
                ''.join(['.' for i in range(self.level.val)]),
                self.name,
                ", ".join(names),
            )


class ImportFrom:
    def __init__(self, name):
        self.name = name

    def store(self, dec, dest):
        imp = dec.stack.peek()
        assert isinstance(imp, ImportStatement)

        if imp.fromlist != PyConst(None):

            imp.aslist.append(dest.name)
        else:
            imp.alias = dest


class SimpleStatement(PyStatement):
    def __init__(self, val):
        assert val is not None
        self.val = val

    def display(self, indent):
        indent.write(self.val)

    def gen_display(self, seq=()):
        return " ".join((self.val,) + seq)


class IfStatement(PyStatement):
    def __init__(self, cond, true_suite, false_suite):
        self.cond = cond
        self.true_suite = true_suite
        self.false_suite = false_suite

    def display(self, indent, is_elif=False):
        ptn = "elif {}:" if is_elif else "if {}:"
        indent.write(ptn, self.cond)
        self.true_suite.display(indent + 1)
        if not self.false_suite:
            return
        if len(self.false_suite) == 1:
            stmt = self.false_suite[0]
            if isinstance(stmt, IfStatement):
                stmt.display(indent, is_elif=True)
                return
        indent.write("else:")
        self.false_suite.display(indent + 1)

    def gen_display(self, seq=()):
        assert not self.false_suite
        s = "if {}".format(self.cond)
        return self.true_suite.gen_display(seq + (s,))


class ForStatement(PyStatement, AsyncMixin):
    def __init__(self, iterable):
        AsyncMixin.__init__(self)
        self.iterable = iterable
        self.else_body: Suite = None

    def store(self, dec, dest):
        self.dest = dest

    def display(self, indent):
        indent.write(
            "{}for {} in {}:",
            self.async_prefix,
            self.dest,
            self.iterable,
        )
        if self.body:
            self.body.display(indent + 1)
        if self.else_body:
            indent.write('else:')
            self.else_body.display(indent + 1)

    def gen_display(self, seq=()):
        s = "{}for {} in {}".format(
            self.async_prefix,
            self.dest,
            self.iterable.wrap()
            if isinstance(self.iterable, PyIfElse)
            else self.iterable,
        )
        return self.body.gen_display(seq + (s,))


class WhileStatement(PyStatement):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    def display(self, indent):
        indent.write("while {}:", self.cond)
        self.body.display(indent + 1)


class DecorableStatement(PyStatement):
    def __init__(self):
        self.decorators = []

    def display(self, indent):
        indent.sep()
        for f in reversed(self.decorators):
            indent.write("@{}", f)
        self.display_undecorated(indent)
        indent.sep()

    def decorate(self, f):
        self.decorators.append(f)


class DefStatement(
    FunctionDefinition, DecorableStatement, AsyncMixin
):
    def __init__(
        self,
        code: Code,
        defaults,
        kwdefaults,
        closure,
        paramobjs=None,
        annotations=None,
    ):
        FunctionDefinition.__init__(
            self,
            code,
            defaults,
            kwdefaults,
            closure,
            paramobjs,
            annotations,
        )
        DecorableStatement.__init__(self)
        AsyncMixin.__init__(self)
        self.is_async = (
            code.flags.coroutine or code.flags.async_generator
        )

    def display_undecorated(self, indent):
        paramlist = ", ".join(self.getparams())
        result = self.getreturn()
        if result:
            indent.write(
                "{}def {}({}) -> {}:",
                self.async_prefix,
                self.code.name,
                paramlist,
                result,
            )
        else:
            indent.write(
                "{}def {}({}):",
                self.async_prefix,
                self.code.name,
                paramlist,
            )
        # Assume that co_consts starts with None unless the function
        # has a docstring, in which case it starts with the docstring
        if self.code.consts[0] != PyConst(None):
            docstring = self.code.consts[0].val
            DocString(docstring).display(indent + 1)
        self.code.get_suite().display(indent + 1)

    def store(self, dec, dest):
        self.name = dest
        dec.suite.add_statement(self)


class TryStatement(PyStatement):
    def __init__(self, try_suite):
        self.try_suite: Suite = try_suite
        self.except_clauses: List[Any, str, Suite] = []
        self.else_suite: Suite = None

    def add_except_clause(self, exception_type, suite):
        self.except_clauses.append([exception_type, None, suite])

    def store(self, dec, dest):
        self.except_clauses[-1][1] = dest

    def display(self, indent):
        indent.write("try:")
        self.try_suite.display(indent + 1)
        for type, name, suite in self.except_clauses:
            if type is None:
                indent.write("except:")
            elif name is None:
                indent.write("except {}:", type)
            else:
                indent.write("except {} as {}:", type, name)
            suite.display(indent + 1)
        if self.else_suite:
            indent.write('else:')
            self.else_suite.display(indent + 1)


class FinallyStatement(PyStatement):
    def __init__(self, try_suite, finally_suite):
        self.try_suite = try_suite
        self.finally_suite = finally_suite

    def display(self, indent):
        # Wrap the try suite in a TryStatement if necessary
        try_stmt = None
        if len(self.try_suite) == 1:
            try_stmt = self.try_suite[0]
            if not isinstance(try_stmt, TryStatement):
                try_stmt = None
        if try_stmt is None:
            try_stmt = TryStatement(self.try_suite)
        try_stmt.display(indent)
        indent.write("finally:")
        self.finally_suite.display(indent + 1)


class WithStatement(PyStatement):
    def __init__(self, with_expr):
        self.with_expr = with_expr
        self.with_name = None
        self.is_async = False

    @property
    def async_prefix(self):
        return 'async ' if self.is_async else ''

    def store(self, dec, dest):
        self.with_name = dest

    def display(self, indent, args=None):
        # args to take care of nested withs:
        # with x as t:
        # with y as u:
        #         <suite>
        # --->
        # with x as t, y as u:
        #     <suite>
        if args is None:
            args = []
        if self.with_name is None:
            args.append(str(self.with_expr))
        else:
            args.append(
                "{} as {}".format(self.with_expr, self.with_name)
            )
        if len(self.suite) == 1 and isinstance(
            self.suite[0], WithStatement
        ):
            self.suite[0].display(indent, args)
        else:
            indent.write(
                self.async_prefix + "with {}:", ", ".join(args)
            )
            self.suite.display(indent + 1)


class ClassStatement(DecorableStatement):
    def __init__(self, func, name, parents, kwargs):
        DecorableStatement.__init__(self)
        self.func = func
        self.parents = parents
        self.kwargs = kwargs

    def store(self, dec, dest):
        self.name = dest
        dec.suite.add_statement(self)

    def display_undecorated(self, indent):
        if self.parents or self.kwargs:
            args = [str(x) for x in self.parents]
            kwargs = [
                "{}={}".format(str(k).replace('\'', ''), v)
                for k, v in self.kwargs
            ]
            all_args = ", ".join(args + kwargs)
            indent.write("class {}({}):", self.name, all_args)
        else:
            indent.write("class {}:", self.name)
        suite = self.func.code.get_suite(look_for_docstring=True)
        if suite:
            # TODO: find out why sometimes the class suite ends with
            # "return __class__"
            last_stmt = suite[-1]
            if isinstance(last_stmt, SimpleStatement):
                if last_stmt.val.startswith("return "):
                    suite.statements.pop()
            clean_vars = ['__module__', '__qualname__']
            for clean_var in clean_vars:
                for i in range(len(suite.statements)):
                    stmt = suite.statements[i]
                    if isinstance(stmt, AssignStatement) and str(
                        stmt
                    ).startswith(clean_var):
                        suite.statements.pop(i)
                        break

        suite.display(indent + 1)


class Suite:
    def __init__(self):
        self.statements = []

    def __bool__(self) -> bool:
        return bool(self.statements)

    def __len__(self) -> int:
        return len(self.statements)

    def __getitem__(self, i) -> PyStatement:
        return self.statements[i]

    def __setitem__(self, i, val: PyStatement):
        self.statements[i] = val

    def __str__(self):
        istr = IndentString()
        self.display(istr)
        return str(istr)

    def display(self, indent):
        if self.statements:
            for stmt in self.statements:
                stmt.display(indent)
        else:
            indent.write("pass")

    def gen_display(self, seq=()):
        if len(self) != 1:
            raise Exception(
                'There should only be one statement in a generator.'
            )
        return self[0].gen_display(seq)

    def add_statement(self, stmt):
        self.statements.append(stmt)


class SuiteDecompiler:
    # An instruction handler can return this to indicate to the run()
    # function that it should return immediately
    END_NOW = object()

    # This is put on the stack by LOAD_BUILD_CLASS
    BUILD_CLASS = object()

    def __init__(self, start_addr, end_addr=None, stack=None):
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.code: Code = start_addr.code
        self.stack = Stack() if stack is None else stack
        self.suite: Suite = Suite()
        self.assignment_chain = []
        self.popjump_stack = []
        self.is_loop = False
        self.convert_return_break = False
        other_self = sys._getframe(1).f_locals.get("self")
        if other_self and hasattr(other_self, "is_loop"):
            self.is_loop = other_self.is_loop

    def IS_OP(self, addr, oparg):
        # case TARGET(IS_OP):
        right = self.stack.pop()
        left = self.stack.peek()  # was TOP()
        # res: int = (1 if left == right else 0) ^ oparg;
        # b : bool = True if res else False;
        self.stack.push(PyIsOp(left, right, oparg))
        # self.stack.push(b);

    def CONTAINS_OP(self, addr, oparg):
        right = self.stack.pop()
        left = self.stack.pop()
        pyseq = right
        item = left
        # res: int = 1 if (left in pyseq) else 0;
        # if res < 0: raise AssertionError("res < 0: %d" % res)
        # b: bool = true if (res ^ oparg) else false
        # self.stack.push(b)
        self.stack.push(PyInOp(left, right, oparg))

    def push_popjump(self, jtruthiness, jaddr, jcond, original_jaddr):
        stack = self.popjump_stack
        if jaddr and jaddr[-1].is_else_jump:
            # Increase jaddr to the 'else' address if it jumps to the 'then'
            jaddr = jaddr[-1].jump()
        while stack:
            truthiness, addr, cond, original_addr = stack[-1]
            # if jaddr == None:
            #     raise Exception("#ERROR: jaddr is None")
            # jaddr == None \
            if jaddr and jaddr < addr or jaddr == addr:
                break
            stack.pop()
            obj_maker = PyBooleanOr if truthiness else PyBooleanAnd
            if truthiness and jtruthiness:
                if original_jaddr.arg == original_addr.arg:
                    obj_maker = PyBooleanAnd
                    cond = PyNot(cond)
                    jcond = PyNot(jcond)
                elif original_jaddr.arg > original_addr.arg:
                    obj_maker = PyBooleanOr
                    jcond = PyNot(jcond)
            if not truthiness and not jtruthiness:
                if original_jaddr.arg < original_addr.arg:
                    obj_maker = PyBooleanOr
                    cond = PyNot(cond)
                elif original_jaddr.arg > original_addr.arg:
                    obj_maker = PyBooleanOr
                    cond = PyNot(cond)
            if truthiness and not jtruthiness:
                if original_jaddr.arg == original_addr.arg:
                    obj_maker = PyBooleanAnd
                    cond = PyNot(cond)
            if isinstance(jcond, obj_maker):
                # Use associativity of 'and' and 'or' to minimise the
                # number of parentheses
                jcond = obj_maker(
                    obj_maker(cond, jcond.left), jcond.right
                )
            else:
                jcond = obj_maker(cond, jcond)
        stack.append((jtruthiness, jaddr, jcond, original_jaddr))

    def pop_popjump(self):
        if not self.popjump_stack:
            raise Exception(
                'Attempted to pop an empty popjump stack.'
            )
        (
            truthiness,
            addr,
            cond,
            original_addr,
        ) = self.popjump_stack.pop()
        return cond

    def run(self):
        addr, end_addr = self.start_addr, self.end_addr
        while addr and addr < end_addr:
            opcode, arg = addr
            args = (addr,) if opcode < HAVE_ARGUMENT else (addr, arg)
            try:
                method = getattr(self, opname[opcode])
                new_addr = method(*args)
                if new_addr is self.END_NOW:
                    break
                elif new_addr is None:
                    new_addr = addr[1]
                addr = new_addr
            except BaseException as e:
                traceback.print_exception(BaseException, e, e.__traceback__)
                addr = None
        return addr

    def write(self, template, *args):
        def fmt(x):
            if isinstance(x, int):
                return self.stack.getval(x)
            else:
                return x

        if args:
            line = template.format(*map(fmt, args))
        else:
            line = template
        self.suite.add_statement(SimpleStatement(line))

    def store(self, dest):
        val = self.stack.pop()
        val.store(self, dest)

    def is_for_loop(self, addr, end_addr):
        i = 0
        while 1:
            cur_addr = addr[i]
            if cur_addr == end_addr:
                break
            elif cur_addr.opcode in else_jump_opcodes:
                cur_addr = cur_addr.jump()
                if cur_addr and cur_addr.opcode in for_jump_opcodes:
                    return True
                break
            elif cur_addr.opcode in for_jump_opcodes:
                return True
            i = i + 1
        return False

    def scan_to_first_jump_if(
        self, addr: Address, end_addr: Address
    ) -> Union[Address, None]:
        i = 0
        while 1:
            cur_addr = addr[i]
            if cur_addr == end_addr:
                break
            elif cur_addr.opcode in pop_jump_if_opcodes:
                return cur_addr
            elif cur_addr.opcode in else_jump_opcodes:
                break
            elif cur_addr.opcode in for_jump_opcodes:
                break
            i = i + 1
        return None

    def scan_for_final_jump(self, start_addr, end_addr):
        i = 0
        end = None
        while 1:
            cur_addr = end_addr[i]
            if cur_addr == start_addr:
                break
            elif cur_addr.opcode is JUMP_ABSOLUTE:
                end = cur_addr
                return end
            elif cur_addr.opcode in else_jump_opcodes:
                break
            elif cur_addr.opcode in pop_jump_if_opcodes:
                break
            i = i - 1
        return end

    #
    # All opcode methods in CAPS below.
    #

    def SETUP_LOOP(self, addr: Address, delta):
        jump_addr = addr.jump()
        end_addr = jump_addr[-1]

        end_cond = self.scan_to_first_jump_if(addr[1], end_addr)
        if (
            end_addr.opcode in (POP_BLOCK, POP_TOP)
            or not end_cond
            or end_addr.seek_back(
                else_jump_opcodes, end_addr.seek_back(stmt_opcodes)
            )
        ):  # assume conditional
            # scan to first jump
            end_jump = None if not end_cond else end_cond.jump()
            if end_jump and end_jump.opcode is POP_BLOCK:
                end_jump = end_jump[1]

            if end_cond and end_cond[1].opcode is BREAK_LOOP:
                end_cond = None
            if end_cond and end_jump == jump_addr:
                # scan for conditional
                d_cond = SuiteDecompiler(addr[1], end_cond)
                #
                d_cond.run()
                cond = d_cond.stack.pop()
                if end_cond.opcode is POP_JUMP_IF_TRUE:
                    cond = PyNot(cond)
                d_body = SuiteDecompiler(end_cond[1], end_addr)
                while_stmt = WhileStatement(cond, d_body.suite)
                d_body.stack.push(while_stmt)
                d_body.run()
                while_stmt.body = d_body.suite
                self.suite.add_statement(while_stmt)
                return jump_addr
            elif (
                not end_cond or not end_cond.jump()[1] == addr.jump()
            ) and not self.is_for_loop(addr[1], end_addr):
                d_body = SuiteDecompiler(addr[1], end_addr)
                while_stmt = WhileStatement(
                    PyConst(True), d_body.suite
                )
                d_body.stack.push(while_stmt)
                d_body.run()
                while_stmt.body = d_body.suite
                self.suite.add_statement(while_stmt)
                return jump_addr
        return None

    def BREAK_LOOP(self, addr):
        self.write("break")

    def CONTINUE_LOOP(self, addr, *argv):
        self.write("continue")

    def SETUP_FINALLY(self, addr, delta):
        start_finally: Address = addr.jump()
        d_try = SuiteDecompiler(addr[1], start_finally)
        d_try.run()
        d_finally = SuiteDecompiler(start_finally)
        end_finally = d_finally.run()
        self.suite.add_statement(
            FinallyStatement(d_try.suite, d_finally.suite)
        )
        if end_finally:
            return end_finally[1]
        else:
            return self.END_NOW

    def END_FINALLY(self, addr):
        return self.END_NOW

    def SETUP_EXCEPT(self, addr, delta):
        end_addr = addr
        start_except = addr.jump()
        start_try = addr[1]
        end_try = start_except
        if sys.version_info < (3, 7):
            if end_try.opcode is JUMP_FORWARD:
                end_try = end_try[1] + end_try.arg
            elif end_try.opcode is JUMP_ABSOLUTE:
                end_try = end_try[-1]
            else:
                end_try = end_try[1]
        d_try = SuiteDecompiler(start_try, end_try)
        d_try.run()

        stmt = TryStatement(d_try.suite)
        j_except: Address = None
        while start_except.opcode is not END_FINALLY:
            if start_except.opcode is DUP_TOP:
                # There's a new except clause
                d_except = SuiteDecompiler(start_except[1])
                d_except.stack.push(stmt)
                d_except.run()
                start_except = stmt.next_start_except
                j_except = start_except[-1]
                end_addr = start_except[1]
            elif start_except.opcode is POP_TOP:
                # It's a bare except clause - it starts:
                # POP_TOP
                # POP_TOP
                # POP_TOP
                # <except stuff>
                # POP_EXCEPT
                start_except = start_except[3]
                end_except = start_except

                nested_try: int = 0
                while (
                    end_except
                    and end_except[-1].opcode is not RETURN_VALUE
                ):
                    if end_except.opcode is SETUP_EXCEPT:
                        nested_try += 1
                    if end_except.opcode is POP_EXCEPT:
                        if nested_try == 0:
                            break
                        nested_try -= 1
                    end_except = end_except[1]
                # Handle edge case where there is a return in the except
                if end_except[-1].opcode is RETURN_VALUE:
                    d_except = SuiteDecompiler(
                        start_except, end_except
                    )
                    end_except = d_except.run()
                    stmt.add_except_clause(None, d_except.suite)
                    self.suite.add_statement(stmt)
                    return end_except

                d_except = SuiteDecompiler(start_except, end_except)
                end_except = d_except.run()
                stmt.add_except_clause(None, d_except.suite)
                start_except = end_except[2]
                assert start_except.opcode is END_FINALLY

                end_addr = start_except[1]
                j_except: Address = end_except[1]
        self.suite.add_statement(stmt)
        if j_except and j_except.opcode in (
            JUMP_FORWARD,
            JUMP_ABSOLUTE,
            RETURN_VALUE,
        ):
            j_next = j_except.jump()
            start_else = end_addr
            if j_next:
                if j_next < start_else:
                    if j_next < start_else and "SETUP_LOOP" in globals():
                        j_next = j_next.seek_back(SETUP_LOOP)
                        if j_next:
                            j_next = j_next.jump()
            else:
                return_count = 0
                next_return = start_else
                while next_return:
                    if next_return.opcode in pop_jump_if_opcodes:
                        j_next_return = next_return.jump()
                        if j_next_return > next_return:
                            next_return = j_next_return
                    if next_return.opcode is RETURN_VALUE:
                        return_count += 1
                    next_return = next_return[1]
                if return_count == 1:
                    return end_addr

            end_else = j_next
            d_else = SuiteDecompiler(start_else, end_else)
            end_addr = d_else.run()
            if not end_addr:
                end_addr = self.END_NOW
            stmt.else_suite = d_else.suite
        return end_addr

    def SETUP_WITH(self, addr, delta):
        end_with = addr.jump()
        with_stmt = WithStatement(self.stack.pop())
        d_with = SuiteDecompiler(addr[1], end_with)
        d_with.stack.push(with_stmt)
        d_with.run()
        with_stmt.suite = d_with.suite
        self.suite.add_statement(with_stmt)
        if sys.version_info <= (3, 4):
            assert end_with.opcode is WITH_CLEANUP
            assert end_with[1].opcode is END_FINALLY
            return end_with[2]
        elif end_with.opcode is WITH_CLEANUP_START:
            assert end_with.opcode is WITH_CLEANUP_START
            assert end_with[1].opcode is WITH_CLEANUP_FINISH
            return end_with[3]
        elif end_with.opcode is WITH_EXCEPT_START:
            """
            TARGET(WITH_EXCEPT_START) {
                /* At the top of the stack are 4 values:
                   - TOP = exc_info()
                   - SECOND = previous exception
                   - THIRD: lasti of exception in exc_info()
                   - FOURTH: the context.__exit__ bound method
                   We call FOURTH(type(TOP), TOP, GetTraceback(TOP)).
                   Then we push the __exit__ return value.
                */
                PyObject *exit_func;
                PyObject *exc, *val, *tb, *res;
                val = TOP();
                assert(val && PyExceptionInstance_Check(val));
                exc = PyExceptionInstance_Class(val);
                tb = PyException_GetTraceback(val);
                Py_XDECREF(tb);
                assert(PyLong_Check(PEEK(3)));
                exit_func = PEEK(4);
                PyObject *stack[4] = {NULL, exc, val, tb};
                res = PyObject_Vectorcall(exit_func, stack + 1,
                        3 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
            }
            """
            start_index = [
                idx
                for idx, (a, (k, v)) in enumerate(
                    end_with.code.instr_seq
                )
                if a == end_with[0].addr
            ][0]
            end_index = [
                idx
                for idx, (a, (k, v)) in enumerate(
                    end_with.code.instr_seq
                )
                if idx > start_index and k == POP_EXCEPT
            ][0]
            """
>>      182     WITH_EXCEPT_START
        184     POP_JUMP_IF_TRUE        188
        186     RERAISE
>>      188     POP_TOP
        190     POP_TOP
        192     POP_TOP
        194     POP_EXCEPT
            """
            assert end_with.opcode is WITH_EXCEPT_START
            assert (
                end_with[end_index - start_index].opcode is POP_EXCEPT
            )
            return end_with[end_index - start_index]
        else:
            raise AssertionError(
                "Unexpectee opcode at start of BEGIN_WITH:"
                " end_with.opcode = {}".format(end_with.opcode)
            )

    def POP_BLOCK(self, addr):
        pass

    def POP_EXCEPT(self, addr):
        return self.END_NOW

    def NOP(self, addr):
        return

    def SETUP_ANNOTATIONS(self, addr):
        return

    def COMPARE_OP(self, addr, compare_opname):
        left, right = self.stack.pop(2)
        if compare_opname != 10:  # 10 is exception match
            self.stack.push(
                PyCompare([left, cmp_op[compare_opname], right])
            )
        else:
            # It's an exception match
            # left is a TryStatement
            # right is the exception type to be matched
            # It goes:
            # COMPARE_OP 10
            # POP_JUMP_IF_FALSE <next except>
            # POP_TOP
            # POP_TOP or STORE_FAST (if the match is named)
            # POP_TOP
            # SETUP_FINALLY if the match was named
            assert addr[1].opcode is POP_JUMP_IF_FALSE
            left.next_start_except = addr[1].jump()
            assert addr[2].opcode is POP_TOP
            assert addr[4].opcode is POP_TOP
            if addr[5].opcode is SETUP_FINALLY:
                except_start = addr[6]
                except_end = addr[5].jump()
            else:
                except_start = addr[5]
                except_end = left.next_start_except
            d_body = SuiteDecompiler(except_start, except_end)
            d_body.run()
            left.add_except_clause(right, d_body.suite)
            if addr[3].opcode is not POP_TOP:
                # The exception is named
                d_exc_name = SuiteDecompiler(addr[3], addr[4])
                d_exc_name.stack.push(left)
                # This will store the name in left:
                d_exc_name.run()
            # We're done with this except clause
            return self.END_NOW

    #
    # Stack manipulation
    #

    def POP_TOP(self, addr):
        val = self.stack.pop()
        if (sys.version_info > (3, 7)
                and isinstance(val, (ForStatement, WhileStatement))
                and addr[1].opcode is JUMP_ABSOLUTE):
                self.write("break")
                return addr[2]
        val.on_pop(self)

    calls = []

    def PRINT_EXPR(self, addr):

        expr = self.stack.pop()
        self.calls += [(self, addr, expr)]

        self.write("{}", expr)
        # expr.on_pop(self)

    def ROT_TWO(self, addr):
        # special case: x, y = z, t
        if (
            addr[2]
            and addr[1].opcode is STORE_NAME
            and addr[2].opcode is STORE_NAME
        ):
            val = PyTuple(self.stack.pop(2))
            unpack = Unpack(val, 2)
            self.stack.push(unpack)
            self.stack.push(unpack)
        else:
            tos1, tos = self.stack.pop(2)
            self.stack.push(tos, tos1)

    def ROT_THREE(self, addr):
        # special case: x, y, z = a, b, c
        if (
            addr[4]
            and addr[1].opcode is ROT_TWO
            and addr[2].opcode is STORE_NAME
            and addr[3].opcode is STORE_NAME
            and addr[4].opcode is STORE_NAME
        ):
            val = PyTuple(self.stack.pop(3))
            unpack = Unpack(val, 3)
            self.stack.push(unpack)
            self.stack.push(unpack)
            self.stack.push(unpack)
            return addr[2]
        else:
            tos2, tos1, tos = self.stack.pop(3)
            self.stack.push(tos, tos2, tos1)

    def DUP_TOP(self, addr):
        self.stack.push(self.stack.peek())

    def DUP_TOP_TWO(self, addr):
        self.stack.push(*self.stack.peek(2))

    #
    # LOAD / STORE / DELETE
    #

    # FAST

    def LOAD_FAST(self, addr, var_num):
        name = self.code.varnames[var_num]
        self.stack.push(name)

    def STORE_FAST(self, addr, var_num):
        name = self.code.varnames[var_num]
        self.store(name)

    def DELETE_FAST(self, addr, var_num):
        name = self.code.varnames[var_num]
        self.write("del {}", name)

    # DEREF

    def LOAD_DEREF(self, addr, i):
        name = self.code.derefnames[i]
        self.stack.push(name)

    def LOAD_CLASSDEREF(self, addr, i):
        name = self.code.derefnames[i]
        self.stack.push(name)

    def STORE_DEREF(self, addr, i):
        name = self.code.derefnames[i]
        if not self.code.iscellvar(i):
            self.code.declare_nonlocal(name)
        self.store(name)

    def DELETE_DEREF(self, addr, i):
        name = self.code.derefnames[i]
        if not self.code.iscellvar(i):
            self.code.declare_nonlocal(name)
        self.write("del {}", name)

    # GLOBAL

    def LOAD_GLOBAL(self, addr, namei):
        name = self.code.names[namei]
        self.code.ensure_global(name)
        self.stack.push(name)

    def STORE_GLOBAL(self, addr, namei):
        name = self.code.names[namei]
        self.code.declare_global(name)
        self.store(name)

    def DELETE_GLOBAL(self, addr, namei):
        name = self.code.names[namei]
        self.declare_global(name)
        self.write("del {}", name)

    # NAME

    def LOAD_NAME(self, addr, namei):
        name = self.code.names[namei]
        self.stack.push(name)

    def STORE_NAME(self, addr, namei):
        name = self.code.names[namei]
        self.store(name)

    def DELETE_NAME(self, addr, namei):
        name = self.code.names[namei]
        self.write("del {}", name)

    # METHOD
    def LOAD_METHOD(self, addr, namei):
        expr = self.stack.pop()
        attrname = self.code.names[namei]
        self.stack.push(PyAttribute(expr, attrname))

    def CALL_METHOD(self, addr, argc, have_var=False, have_kw=False):
        kw_argc = argc >> 8
        pos_argc = argc
        varkw = self.stack.pop() if have_kw else None
        varargs = self.stack.pop() if have_var else None
        kwargs_iter = iter(self.stack.pop(2 * kw_argc))
        kwargs = list(zip(kwargs_iter, kwargs_iter))
        posargs = self.stack.pop(pos_argc)
        func = self.stack.pop()
        if func is self.BUILD_CLASS:
            # It's a class construction
            # TODO: check the assert statement below is correct
            assert not (have_var or have_kw)
            func, name, *parents = posargs
            self.stack.push(
                ClassStatement(func, name, parents, kwargs)
            )
        elif isinstance(func, PyComp):
            # It's a list/set/dict comprehension or generator expression
            assert not (have_var or have_kw)
            assert len(posargs) == 1 and not kwargs
            func.set_iterable(posargs[0])
            self.stack.push(func)
        elif posargs and isinstance(posargs[0], DecorableStatement):
            # It's a decorator for a def/class statement
            assert len(posargs) == 1 and not kwargs
            defn = posargs[0]
            defn.decorate(func)
            self.stack.push(defn)
        else:
            # It's none of the above, so it must be a normal function call
            func_call = PyCallFunction(
                func, posargs, kwargs, varargs, varkw
            )
            self.stack.push(func_call)

    # ATTR

    def LOAD_ATTR(self, addr, namei):
        expr = self.stack.pop()
        attrname = self.code.names[namei]
        self.stack.push(PyAttribute(expr, attrname))

    def STORE_ATTR(self, addr, namei):
        expr = self.stack.pop1()
        attrname = self.code.names[namei]
        self.store(PyAttribute(expr, attrname))

    def DELETE_ATTR(self, addr, namei):
        expr = self.stack.pop()
        attrname = self.code.names[namei]
        self.write("del {}.{}", expr, attrname)

    # SUBSCR

    def STORE_SUBSCR(self, addr):
        expr, sub = self.stack.pop(2)
        self.store(PySubscript(expr, sub))

    def DELETE_SUBSCR(self, addr):
        expr, sub = self.stack.pop(2)
        self.write("del {}[{}]", expr, sub)

    # CONST
    CONST_LITERALS = {Ellipsis: PyName('...')}

    def LOAD_CONST(self, addr, consti):
        const = self.code.consts[consti]
        if const.val in self.CONST_LITERALS:
            const = self.CONST_LITERALS[const.val]
        self.stack.push(const)

    #
    # Import statements
    #

    def IMPORT_NAME(self, addr, namei):
        name = self.code.names[namei]

        if len(self.stack._stack) > 1:
            level, fromlist = self.stack.pop(2)
        else:
            return addr[0]
        self.stack.push(ImportStatement(name, level, fromlist))
        # special case check for import x.y.z as w
        # syntax which uses attributes and assignments
        # and is difficult to workaround
        i = 1
        while addr[i].opcode is LOAD_ATTR:
            i = i + 1
        if i > 1 and addr[i].opcode in (STORE_FAST, STORE_NAME, STORE_ATTR):
            return addr[i]
        return None

    def IMPORT_FROM(self, addr: Address, namei):
        name = self.code.names[namei]
        self.stack.push(ImportFrom(name))
        if addr[1].opcode is ROT_TWO:
            return addr.seek_forward(STORE_NAME)

    def IMPORT_STAR(self, addr):
        self.POP_TOP(addr)

    #
    # Function call
    #

    def STORE_LOCALS(self, addr):
        self.stack.pop()
        return addr[3]

    def LOAD_BUILD_CLASS(self, addr):
        self.stack.push(self.BUILD_CLASS)

    def RETURN_VALUE(self, addr):
        value = self.stack.pop()
        if isinstance(value, PyConst) and value.val is None:
            if addr[1] is not None:
                if (
                    self.code.flags.generator
                    and addr[3]
                    and not self.code[0].seek_forward(
                        {YIELD_FROM, YIELD_VALUE}
                    )
                ):
                    self.write('yield')
                else:
                    self.write("return")
            return
        if self.code.flags.iterable_coroutine:
            self.write("yield {}", value)
        else:
            self.write("return {}", value)
            if self.code.flags.generator:
                self.write('yield')

    def GET_YIELD_FROM_ITER(self, addr):
        pass

    def YIELD_VALUE(self, addr):
        if self.code.name == '<genexpr>':
            return
        value = self.stack.pop()
        self.stack.push(PyYield(value))

    def YIELD_FROM(self, addr):
        value = self.stack.pop()  # TODO:  from statement ?
        value = self.stack.pop()
        self.stack.push(PyYieldFrom(value))

    def CALL_FUNCTION_CORE(
        self, func, posargs, kwargs, varargs, varkw
    ):
        if func is self.BUILD_CLASS:
            # It's a class construction
            # TODO: check the assert statement below is correct
            # assert not (have_var or have_kw)
            func, name, *parents = posargs
            self.stack.push(
                ClassStatement(func, name, parents, kwargs)
            )
        elif isinstance(func, PyComp):
            # It's a list/set/dict comprehension or generator expression
            # assert not (have_var or have_kw)
            assert len(posargs) == 1 and not kwargs
            func.set_iterable(posargs[0])
            self.stack.push(func)
        elif (
            posargs
            and isinstance(posargs, list)
            and isinstance(posargs[0], DecorableStatement)
        ):
            # It's a decorator for a def/class statement
            assert len(posargs) == 1 and not kwargs
            defn = posargs[0]
            defn.decorate(func)
            self.stack.push(defn)
        else:
            # It's none of the above, so it must be a normal function call
            func_call = PyCallFunction(
                func, posargs, kwargs, varargs, varkw
            )
            self.stack.push(func_call)

    def CALL_FUNCTION(
        self, addr, argc, have_var=False, have_kw=False
    ):
        if sys.version_info >= (3, 6):
            pos_argc = argc
            posargs = self.stack.pop(pos_argc)
            func = self.stack.pop()
            self.CALL_FUNCTION_CORE(func, posargs, [], None, None)
        else:
            kw_argc = argc >> 8
            pos_argc = argc & 0xFF
            varkw = self.stack.pop() if have_kw else None
            varargs = self.stack.pop() if have_var else None
            kwargs_iter = iter(self.stack.pop(2 * kw_argc))
            kwargs = list(zip(kwargs_iter, kwargs_iter))
            posargs = self.stack.pop(pos_argc)
            func = self.stack.pop()
            self.CALL_FUNCTION_CORE(
                func, posargs, kwargs, varargs, varkw
            )

    def CALL_FUNCTION_VAR(self, addr, argc):
        self.CALL_FUNCTION(addr, argc, have_var=True)

    def CALL_FUNCTION_KW(self, addr, argc):
        if sys.version_info >= (3, 6):
            keys = self.stack.pop()
            kwargc = len(keys.val)
            kwarg_values = self.stack.pop(kwargc)
            posargs = self.stack.pop(argc - kwargc)
            func = self.stack.pop()
            kwarg_dict = list(
                zip([PyName(k) for k in keys], kwarg_values)
            )
            self.CALL_FUNCTION_CORE(
                func, posargs, kwarg_dict, None, None
            )
        else:
            self.CALL_FUNCTION(addr, argc, have_kw=True)

    def CALL_FUNCTION_EX(self, addr, flags):
        kwarg_unpacks = []
        if flags & 1:
            kwarg_unpacks = self.stack.pop()

        kwarg_dict = PyDict()
        if isinstance(kwarg_unpacks, PyDict):
            kwarg_dict = kwarg_unpacks
            kwarg_unpacks = []
        elif isinstance(kwarg_unpacks, list):
            if len(kwarg_unpacks):
                if isinstance(kwarg_unpacks[0], PyDict):
                    kwarg_dict = kwarg_unpacks[0]
                    kwarg_unpacks = kwarg_unpacks[1:]
        else:
            kwarg_unpacks = [kwarg_unpacks]

        if any(
            filter(lambda kv: '.' in str(kv[0]), kwarg_dict.items)
        ):
            kwarg_unpacks.append(kwarg_dict)
            kwarg_dict = PyDict()

        posargs_unpacks = self.stack.pop()
        posargs = PyTuple([])
        if isinstance(posargs_unpacks, PyTuple):
            posargs = posargs_unpacks
            posargs_unpacks = []
        elif isinstance(posargs_unpacks, list):
            if len(posargs_unpacks) > 0:
                posargs = posargs_unpacks[0]
                if isinstance(posargs, PyConst):
                    posargs = PyTuple(
                        [PyConst(a) for a in posargs.val]
                    )
                elif isinstance(posargs, PyAttribute):
                    posargs = PyTuple([posargs])
                posargs_unpacks = posargs_unpacks[1:]
        else:
            posargs_unpacks = [posargs_unpacks]

        func = self.stack.pop()
        self.CALL_FUNCTION_CORE(
            func,
            list(posargs.values),
            list(kwarg_dict.items),
            posargs_unpacks,
            kwarg_unpacks,
        )

    def CALL_FUNCTION_VAR_KW(self, addr, argc):
        self.CALL_FUNCTION(addr, argc, have_var=True, have_kw=True)

    # a, b, ... = ...

    def UNPACK_SEQUENCE(self, addr, count):
        unpack = Unpack(self.stack.pop(), count)
        for i in range(count):
            self.stack.push(unpack)

    def UNPACK_EX(self, addr, counts):
        rcount = counts >> 8
        lcount = counts & 0xFF
        count = lcount + rcount + 1
        unpack = Unpack(self.stack.pop(), count, lcount)
        for i in range(count):
            self.stack.push(unpack)

    # Build operations

    def BUILD_SLICE(self, addr, argc):
        assert argc in (2, 3)
        self.stack.push(PySlice(self.stack.pop(argc)))

    def BUILD_TUPLE(self, addr, count):
        values = [self.stack.pop() for i in range(count)]
        values.reverse()
        self.stack.push(PyTuple(values))

    def DICT_MERGE(self, addr, count):
        values = []
        for o in self.stack.pop(count):
            if isinstance(o, PyTuple):
                values.extend(o.values)
            else:
                values.append(PyStarred(o))
        self.stack.push(PyList(values))

    def BUILD_TUPLE_UNPACK(self, addr, count):
        values = []
        for o in self.stack.pop(count):
            if isinstance(o, PyTuple):
                values.extend(o.values)
            else:
                values.append(PyStarred(o))

        self.stack.push(PyTuple(values))

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, addr, count):
        self.stack.push(self.stack.pop(count))

    def BUILD_LIST(self, addr, count):
        values = [self.stack.pop() for i in range(count)]
        values.reverse()
        self.stack.push(PyList(values))

    def BUILD_LIST_UNPACK(self, addr, count):
        values = []
        for o in self.stack.pop(count):
            if isinstance(o, PyTuple):
                values.extend(o.values)
            else:
                values.append(PyStarred(o))

        self.stack.push(PyList(values))

    def BUILD_SET(self, addr, count):
        values = [self.stack.pop() for i in range(count)]
        values.reverse()
        self.stack.push(PySet(values))

    def BUILD_SET_UNPACK(self, addr, count):
        values = []
        for o in self.stack.pop(count):
            if isinstance(o, PySet):
                values.extend(o.values)
            else:
                values.append(PyStarred(o))

        self.stack.push(PySet(values))

    def BUILD_MAP(self, addr, count):
        d = PyDict()
        if sys.version_info >= (3, 5):
            for i in range(count):
                d.items.append(tuple(self.stack.pop(2)))
            d.items = list(reversed(d.items))
        self.stack.push(d)

    def BUILD_MAP_UNPACK(self, addr, count):
        d = PyDict()
        for i in range(count):
            o = self.stack.pop()
            if isinstance(o, PyDict):
                for item in reversed(o.items):
                    k, v = item
                    d.set_item(
                        PyConst(
                            k.val
                            if isinstance(k, PyConst)
                            else k.name
                        ),
                        v,
                    )
            else:
                d.items.append((PyStarred(PyStarred(o)),))
        d.items = list(reversed(d.items))
        self.stack.push(d)

    def BUILD_MAP_UNPACK_WITH_CALL(self, addr, count):
        self.stack.push(self.stack.pop(count))

    def BUILD_CONST_KEY_MAP(self, addr, count):
        keys = self.stack.pop()
        vals = self.stack.pop(count)
        dict = PyDict()
        for i in range(count):
            dict.set_item(PyConst(keys.val[i]), vals[i])
        self.stack.push(dict)

    def STORE_MAP(self, addr):
        v, k = self.stack.pop(2)
        d = self.stack.peek()
        d.set_item(k, v)

    # Comprehension operations - just create an expression statement

    def LIST_APPEND(self, addr, i):
        self.POP_TOP(addr)

    def SET_ADD(self, addr, i):
        self.POP_TOP(addr)

    def MAP_ADD(self, addr, i):
        value, key = self.stack.pop(2)
        self.stack.push(PyKeyValue(key, value))
        self.POP_TOP(addr)

    """
    def LIST_TO_TUPLE(self, addr):
      list_obj = self.stack.pop()
      tuple_obj = PyTuple(list_obj)
      #self.POP_TOP(addr)
      self.stack.push(tuple_obj)
      
    def LIST_EXTEND(self, addr, oparg):
      iterable = self.stack.pop()
      list_obj = self.stack.peek(oparg)
      none_val = None
      try:
       list_obj += [iterable]
       #self.stack.push(PyList(list_obj))
      except TypeError as _te:
        if not hasattr(iterable, "__iter__") and \
           not hasattr(iterable, "__getitem__"):
          raise TypeError(
              "Value after * must be an iterable, not {!s}".format(
                  type(iterable).__name__))
        list_obj[0].values + [PyStarred(iterable)]
      self.stack.push(PyList(list_obj[0].values))
      #self.POP_TOP(addr)
      
    def SET_UPDATE(self, addr, i, oparg):
      iterable = self.stack.pop()
      set_obj = self.stack.peek(oparg)[0]
      for item in iterable:
        set_obj.values.add(item)
      self.stack.push(set_obj)
      #self.POP_TOP(addr)
    
    def DICT_UPDATE(self, addr, i, oparg):
      update = self.stack.pop()
      dict_obj = self.stack.peek(oparg)[0]
      if (PyDict_Update(dict_obj, update) < 0):
        if (_PyErr_ExceptionMatches(AttributeError)):
          raise TypeError("'{!s}' object_obj is not a mapping".format(
                  type(update).__name__))
      #self.POP_TOP(addr)
      self.stack.push(dict_obj)
      
    def DICT_MERGE(self, addr, oparg):
      other = self.stack.pop()
      dict_obj = self.stack.peek()
      if isinstance(other, PyDict):
        self.print("{}.update({})".format(dict_obj, other))
        self.stack.push(dict_obj)
      else:
        for key, value in dict_obj.items:
          self.stack.push(PyKeyValue(key, value))
      self.stack.push(dict_obj)
      #self.POP_TOP(addr)

    # and operator
"""

    def LIST_TO_TUPLE(self, addr):
        list_value = self.stack.pop()
        values = list_value.values
        self.stack.push(PyTuple(values))

    def LIST_EXTEND(self, addr, i):
        items = self.stack.pop(1)
        item2 = self.stack.pop(1)
        if hasattr(item2[0], "expr"):
            exprs = [item2[0].expr]
        else:
            exprs = item2[0].values
        new_list = PyList([*exprs, PyStarred(items[0])])
        self.stack.push(new_list)

    def SET_UPDATE(self, addr, i):
        self.POP_TOP(addr)
        pass  # self.POP_TOP(addr)

    def DICT_UPDATE(self, addr, i):
        items = self.stack.pop(1)
        self.stack.push(items[0])
        self.POP_TOP(addr)

    def DICT_MERGE(self, addr, i):
        items = self.stack.pop(1)
        item2 = self.stack.pop(1)
        new_list = [*item2[0].items] + [items[0]]
        item = None
        if len(new_list) == 1:
            item = new_list[0]
        else:
            item = PyList(new_list)
        self.stack.push(item)

    def JUMP_IF_FALSE_OR_POP(self, addr: Address, target):
        end_addr = addr.jump()
        truthiness = not addr.seek_back_statement(POP_JUMP_IF_TRUE)
        self.push_popjump(
            truthiness, end_addr, self.stack.pop(), addr
        )
        left = self.pop_popjump()
        if end_addr.opcode is ROT_TWO:
            opc, arg = end_addr[-1]
            if opc == JUMP_FORWARD and arg == 2:
                end_addr = end_addr[2]
            elif opc == RETURN_VALUE or opc == JUMP_FORWARD:
                end_addr = end_addr[-1]
                d = SuiteDecompiler(addr[1], end_addr, self.stack)
                d.run()
                right = self.stack.pop()
                if isinstance(right, PyCompare) and right.extends(
                    left
                ):
                    py_and = left.chain(right)
                else:
                    py_and = PyBooleanAnd(left, right)
                self.stack.push(py_and)
                return end_addr[3]

        d = SuiteDecompiler(addr[1], end_addr, self.stack)
        if end_addr[-1].opcode is RETURN_VALUE:
            d = SuiteDecompiler(addr[1], end_addr[-1], self.stack)
        else:
            d = SuiteDecompiler(addr[1], end_addr, self.stack)

        d.run()
        # if end_addr.opcode is RETURN_VALUE:
        #     return end_addr[2]
        right = self.stack.pop()
        if isinstance(right, PyCompare) and right.extends(left):
            py_and = left.chain(right)
        else:
            py_and = PyBooleanAnd(left, right)
        self.stack.push(py_and)
        return end_addr

    # This appears when there are chained comparisons, e.g. 1 <= x < 10

    def JUMP_FORWARD(self, addr, delta):
        ## if delta == 2 and addr[1].opcode is ROT_TWO and addr[2].opcode is POP_TOP:
        ##     # We're in the special case of chained comparisons
        ##     return addr[3]
        ## else:
        ##     # I'm hoping its an unused JUMP in an if-else statement
        ##     return addr[1]
        return addr.jump()

    # or operator

    def JUMP_IF_TRUE_OR_POP(self, addr, target):
        end_addr = addr.jump()
        self.push_popjump(True, end_addr, self.stack.pop(), addr)
        left = self.pop_popjump()
        d = SuiteDecompiler(addr[1], end_addr, self.stack)
        d.run()
        right = self.stack.pop()
        self.stack.push(PyBooleanOr(left, right))
        return end_addr

    #
    # If-else statements/expressions and related structures
    #

    def POP_JUMP_IF(
        self, addr: Address, target: int, truthiness: bool
    ) -> Union[Address, None]:
        jump_addr = addr.jump()

        last_loop = None 
        if "SETUP_LOOP" in globals():
            last_loop = addr.seek_back(SETUP_LOOP)
        
        last_loop = last_loop or addr
        in_loop = last_loop and last_loop.jump() > addr
        end_of_loop = (
            jump_addr.opcode is FOR_ITER
            or (
                "SETUP_LOOP" in globals()
                and jump_addr[-1].opcode is SETUP_LOOP
            )
        )
        if jump_addr.opcode is FOR_ITER:
            # We are in a for-loop with nothing after the if-suite
            # But take care: for-loops in generator expression do
            # not end in POP_BLOCK, hence the test below.
            jump_addr = jump_addr.jump()
        else:
            if end_of_loop:
                # We are in a while-loop with nothing after the if-suite
                jump_addr = jump_addr[-1].jump()[-1]
            #else:
                #jump_addr = addr[1]
                # raise Exception("unhandled")
        if self.stack._stack:
            cond = self.stack.pop()
        else:
            cond = not truthiness
        # chained compare
        # ex:
        # if x <= y <= z:
        if (
            addr[-3]
            and addr[-1].opcode is COMPARE_OP
            and addr[-2].opcode is ROT_THREE
            and addr[-3].opcode is DUP_TOP
        ):
            if self.popjump_stack:
                c = self.pop_popjump()
                c = c.chain(cond)
                self.push_popjump(not truthiness, jump_addr, c, addr)
            else:
                self.push_popjump(
                    not truthiness, jump_addr, cond, addr
                )
            return

        is_chained = isinstance(cond, PyCompare) and addr.seek_back(
            ROT_THREE, addr.seek_back(stmt_opcodes)
        )
        if is_chained and self.popjump_stack:
            pj = self.pop_popjump()
            if isinstance(pj, PyCompare):
                cond = pj.chain(cond)

        if not addr.is_else_jump:
            # Handle generator expressions with or clause
            for_iter = addr.seek_back(FOR_ITER)
            if for_iter:
                end_of_for = for_iter.jump()
                if end_of_for.addr > addr.addr:
                    gen = jump_addr.seek_forward(
                        (YIELD_VALUE, LIST_APPEND), end_of_for
                    )
                    if gen:
                        if not truthiness:
                            truthiness = not truthiness
                            if truthiness:
                                cond = PyNot(cond)
                        self.push_popjump(
                            truthiness, jump_addr, cond, addr
                        )
                        return None

            self.push_popjump(truthiness, jump_addr, cond, addr)
            # Dictionary comprehension
            if jump_addr.seek_forward(MAP_ADD):
                return None

            if addr.code.name == '<lambda>':
                return None
            # Generator
            if jump_addr.seek_forward(YIELD_VALUE):
                return None

            if jump_addr.seek_back(
                JUMP_IF_TRUE_OR_POP, jump_addr[-2]
            ):
                return None
            # Generator
            if (
                jump_addr.opcode is not END_FINALLY
                and jump_addr[1]
                and jump_addr[1].opcode is JUMP_ABSOLUTE
            ):
                return None

            next_addr = addr[1]
            while next_addr and next_addr < jump_addr:
                if next_addr.opcode in stmt_opcodes:
                    break
                if next_addr.opcode in pop_jump_if_opcodes:
                    next_jump_addr = next_addr.jump()
                    if next_jump_addr > jump_addr or (
                        next_jump_addr == jump_addr
                        and jump_addr[-1].opcode in else_jump_opcodes
                    ):
                        return None

                if next_addr.opcode in (
                    JUMP_IF_FALSE_OR_POP,
                    JUMP_IF_TRUE_OR_POP,
                ):
                    next_jump_addr = next_addr.jump()
                    if next_jump_addr > jump_addr or (
                        next_jump_addr == jump_addr
                        and jump_addr[-1].opcode in else_jump_opcodes
                    ):
                        return None
                next_addr = next_addr[1]
            # if there are no nested conditionals and no else clause, write the true portion and jump ahead to the end of the conditional
            cond = self.pop_popjump()
            end_true = jump_addr
            if truthiness:
                cond = PyNot(cond)
            d_true = SuiteDecompiler(addr[1], end_true, self.stack[:])
            d_true.run()
            stmt = IfStatement(cond, d_true.suite, None)
            self.suite.add_statement(stmt)
            return end_true
        # Increase jump_addr to pop all previous jumps
        self.push_popjump(truthiness, jump_addr[1], cond, addr)
        cond = self.pop_popjump()
        end_true = jump_addr[-1]
        if truthiness:
            last_pj = addr.seek_back(pop_jump_if_opcodes)
            if (
                last_pj
                and last_pj.arg == addr.arg
                and isinstance(cond, PyBooleanAnd)
                or isinstance(cond, PyBooleanOr)
            ):
                if last_pj.opcode is not addr.opcode:
                    cond.right = PyNot(cond.right)
            else:
                cond = PyNot(cond)

        if end_true.opcode is RETURN_VALUE:
            end_false = jump_addr.seek_forward(RETURN_VALUE)
            if (
                end_false
                and end_false[2]
                and end_false[2].opcode is RETURN_VALUE
            ):
                d_true = SuiteDecompiler(addr[1], jump_addr, self.stack[:])
                d_true.run()
                d_false = SuiteDecompiler(jump_addr, end_false[1], self.stack[:])
                d_false.run()
                self.suite.add_statement(
                    IfStatement(cond, d_true.suite, d_false.suite)
                )

                return end_false[1]

        if (
            end_true.opcode is RAISE_VARARGS
            and addr[1].opcode is LOAD_GLOBAL
        ):
            assert_addr = addr[1]
            if (
                assert_addr.code.names[assert_addr.arg].name
                == 'AssertionError'
            ):
                cond = (
                    cond.operand
                    if isinstance(cond, PyNot)
                    else PyNot(cond)
                )
                d_true = SuiteDecompiler(addr[1], end_true, self.stack[:])
                d_true.run()
                assert_pop = d_true.stack.pop()
                assert_args = (
                    assert_pop.args
                    if isinstance(assert_pop, PyCallFunction)
                    else []
                )
                assert_arg_str = ', '.join(
                    map(str, [cond, *assert_args])
                )
                self.suite.add_statement(
                    SimpleStatement(f'assert {assert_arg_str}')
                )
                return jump_addr
        if self.is_loop:
            # - If the true clause ends in return and we're not in
            # another loop, break instead
            # - If the true clause jumps forward and it's past loop
            # boundaries, break instead
            if self.convert_return_break and end_true.opcode is RETURN_VALUE:
                d_true = SuiteDecompiler(addr[1], end_true, self.stack[:])
                d_true.run()
                d_true.suite.add_statement(SimpleStatement("break"))
                self.suite.add_statement(
                    IfStatement(cond, d_true.suite, Suite())
                )
                return jump_addr
            elif end_true.opcode is JUMP_FORWARD:
                end_false = end_true.jump()
                instr = self.wrap_addr(end_false)[-1]
                if instr.opcode is JUMP_ABSOLUTE:
                    d_true = SuiteDecompiler(addr[1], end_true, self.stack[:])
                    d_true.run()
                    d_true.suite.add_statement(SimpleStatement("break"))
                    self.suite.add_statement(
                        IfStatement(cond, d_true.suite, Suite())
                    )
                    return jump_addr
        # - If the true clause ends in return, make sure it's included
        # - If the true clause ends in RAISE_VARARGS, then it's an
        # assert statement. For now I just write it as a raise within
        # an if (see below)
        if end_true.opcode in (RETURN_VALUE, RAISE_VARARGS, POP_TOP):
            d_true = SuiteDecompiler(addr[1], jump_addr, self.stack[:])
            d_true.run()
            self.suite.add_statement(
                IfStatement(cond, d_true.suite, Suite())
            )
            return jump_addr
        if is_chained and addr[1].opcode is JUMP_ABSOLUTE:
            end_true = end_true[-2]
        d_true = SuiteDecompiler(addr[1], end_true, self.stack[:])
        d_true.run()
        l = None
        if addr[1].opcode is JUMP_ABSOLUTE:
            j = addr[1].jump()
            if last_loop != None and len(last_loop) > 1:
                l = last_loop[1]
            while l != None and l.opcode not in stmt_opcodes:
                if l == j:
                    d_true.suite.add_statement(
                        SimpleStatement('continue')
                    )

                    self.suite.add_statement(
                        IfStatement(cond, d_true.suite, None)
                    )
                    return addr[2]
                l = l[1]

        if jump_addr.opcode is POP_BLOCK and not end_of_loop:
            # It's a while loop
            stmt = WhileStatement(cond, d_true.suite)
            self.suite.add_statement(stmt)
            return jump_addr[1]
        # It's an if-else (expression or statement)
        if end_true.opcode is JUMP_FORWARD:
            end_false = end_true.jump()
        elif end_true.opcode is JUMP_ABSOLUTE:
            end_false = end_true.jump()
            if end_false.opcode is FOR_ITER:
                # We are in a for-loop with nothing after the else-suite
                end_false = end_false.jump()[-1]
            elif "SETUP_LOOP" in globals() and self.wrap_addr(end_false)[-1].opcode is SETUP_LOOP:
                # We are in a while-loop with nothing after the else-suite
                end_false = self.wrap_addr(end_false)[-1].jump()[-1]
            elif self.is_loop:
                end_false = jump_addr
            if end_false.opcode is RETURN_VALUE:
                end_false = self.wrap_addr(end_false)[1]
        elif end_true.opcode is RETURN_VALUE:
            # find the next RETURN_VALUE
            end_false = jump_addr
            while end_false.opcode is not RETURN_VALUE:
                end_false = end_false[1]
            end_false = end_false[1]
        elif end_true.opcode is BREAK_LOOP:
            # likely in a loop in a try/except
            end_false = jump_addr
        else:
            end_false = jump_addr
            # # normal statement
            # raise Exception("#ERROR: Unexpected statement: {} | {}\n".format(end_true, jump_addr, jump_addr[-1]))
            # # raise Unknown
            # jump_addr = end_true[-2]
            # stmt = IfStatement(cond, d_true.suite, None)
            # self.suite.add_statement(stmt)
            # return jump_addr or self.END_NOW
        d_false = SuiteDecompiler(jump_addr, end_false, self.stack[:])
        d_false.run()
        stack_level = len(self.stack)
        if len(d_true.stack) == len(d_false.stack) > stack_level:
            assert not (d_true.suite or d_false.suite)
            # this happens in specific if else conditions with assigments
            true_expr = d_true.stack.pop()
            false_expr = d_false.stack.pop()
            self.stack.push(PyIfElse(cond, true_expr, false_expr))
        else:
            stmt = IfStatement(cond, d_true.suite, d_false.suite)
            self.suite.add_statement(stmt)
        return end_false or self.END_NOW

    def POP_JUMP_IF_FALSE(self, addr, target):
        return self.POP_JUMP_IF(addr, target, truthiness=False)

    def POP_JUMP_IF_TRUE(self, addr, target):
        return self.POP_JUMP_IF(addr, target, truthiness=True)

    def addr2list(self, addr):
        import attrdict, inspect

        if not (addr[-1] is None):
            return addr

        a = dict(
            (idx, addr.code[addr.index + idx])
            for idx in range(-5, len(list(addr.code)) - addr.index)
        )
        [a.__setitem__(*pair) for pair in inspect.getmembers(addr)]
        wa = attrdict()
        wa.update(a)
        if a[-1] is None:
            wa.update({-1: Address(addr.code, addr.index - 1)})
        return wa

    def wrap_addr(self, addr):
        if not isinstance(addr, Address):
            return addr
        if addr[-1] is None:
            return self.addr2list(addr)
        return addr

    def JUMP_ABSOLUTE(self, addr, target):
        import operator, sys

        if 'debug' in globals():
            print("*** JUMP ABSOLUTE ***", addr)
            globals()['debug'] += [
                (
                    self.JUMP_ABSOLUTE,
                    (addr, target),
                    sys._current_frames(),
                )
            ]

        # return addr.jump()

        # TODO: print out continue if not final jump
        jump_addr = addr.jump()

        if "SETUP_LOOP" in globals() and self.wrap_addr(jump_addr)[-1].opcode is SETUP_LOOP:
            end_addr = jump_addr + self.wrap_addr(jump_addr)[-1].arg

            last_jump = self.scan_for_final_jump(
                jump_addr, self.wrap_addr(end_addr)[-1]
            )
            if last_jump != addr:
                self.suite.add_statement(SimpleStatement('continue'))
        pass

    #
    # For loops
    #

    def GET_ITER(self, addr):
        pass

    def FOR_ITER(self, addr: Address, delta):
        if addr[-1] and addr[-1].opcode is RETURN_VALUE:
            # Dead code
            return self.END_NOW
        iterable = self.stack.pop()
        jump_addr = addr.jump()
        end_body = jump_addr
        if end_body.opcode is not POP_BLOCK:
            end_body = end_body[-1]
        d_body = SuiteDecompiler(addr[1], end_body)
        d_body.is_loop = sys.version_info > (3, 7)
        d_body.convert_return_break = not self.is_loop
        for_stmt = ForStatement(iterable)
        d_body.stack.push(*self.stack._stack, for_stmt, for_stmt)
        d_body.run()
        for_stmt.body = d_body.suite
        loop = None
        outer_loop = None
        if "SETUP_LOOP" in globals():
            loop = addr.seek_back(SETUP_LOOP)
        while loop:
            if "SETUP_LOOP" in globals():
                outer_loop = loop.seek_back(SETUP_LOOP)
            if outer_loop:
                if outer_loop.jump().addr < loop.addr:
                    break
                else:
                    loop = outer_loop
            else:
                break
        end_addr = jump_addr
        if loop:
            end_of_loop = loop.jump()[-1]
            if end_of_loop.opcode is not POP_BLOCK:
                else_start = end_of_loop.seek_back(POP_BLOCK)
                d_else = SuiteDecompiler(else_start, loop.jump())
                d_else.run()
                for_stmt.else_body = d_else.suite
                end_addr = loop.jump()
        self.suite.add_statement(for_stmt)
        return end_addr

    # Function creation

    def MAKE_FUNCTION_OLD(self, addr, argc, is_closure=False):
        testType = self.stack.pop().val
        if isinstance(testType, str):
            code = Code(self.stack.pop().val, self.code)
        else:
            code = Code(testType, self.code)
        closure = self.stack.pop() if is_closure else None
        # parameter annotation objects
        paramobjs = {}
        paramcount = (argc >> 16) & 0x7FFF
        if paramcount:
            paramobjs = dict(
                zip(
                    self.stack.pop().val,
                    self.stack.pop(paramcount - 1),
                )
            )
        # default argument objects in positional order
        defaults = self.stack.pop(argc & 0xFF)
        # pairs of name and default argument, with the name just below the object on the stack, for keyword-only parameters
        kwdefaults = {}
        for i in range((argc >> 8) & 0xFF):
            k, v = self.stack.pop(2)
            if hasattr(k, 'name'):
                kwdefaults[k.name] = v
            elif hasattr(k, 'val'):
                kwdefaults[k.val] = v
            else:
                kwdefaults[str(k)] = v
        func_maker = code_map.get(code.name, DefStatement)
        self.stack.push(
            func_maker(code, defaults, kwdefaults, closure, paramobjs)
        )

    def MAKE_FUNCTION_NEW(self, addr, argc, is_closure=False):
        testType = self.stack.pop().val
        if isinstance(testType, str):
            code = Code(self.stack.pop().val, self.code)
        else:
            code = Code(testType, self.code)
        closure = self.stack.pop() if is_closure else None
        annotations = {}
        kwdefaults = {}
        defaults = {}
        if argc & 8:
            annotations = list(self.stack.pop())
        if argc & 4:
            annotations = self.stack.pop()
            if isinstance(annotations, PyDict):
                annotations = {
                    str(k[0].val).replace('\'', ''): str(k[1])
                    for k in annotations.items
                }
        if argc & 2:
            kwdefaults = self.stack.pop()
            if isinstance(kwdefaults, PyDict):
                kwdefaults = {
                    str(k[0].val): str(
                        k[1]
                        if isinstance(k[1], PyExpr)
                        else PyConst(k[1])
                    )
                    for k in kwdefaults.items
                }
            if not kwdefaults:
                kwdefaults = {}
        if argc & 1:
            defaults = list(
                map(
                    lambda x: str(
                        x if isinstance(x, PyExpr) else PyConst(x)
                    ),
                    self.stack.pop(),
                )
            )
        func_maker = code_map.get(code.name, DefStatement)
        self.stack.push(
            func_maker(
                code,
                defaults,
                kwdefaults,
                closure,
                annotations,
                annotations,
            )
        )

    def MAKE_FUNCTION(self, addr, argc, is_closure=False):
        if sys.version_info < (3, 6):
            self.MAKE_FUNCTION_OLD(addr, argc, is_closure)
        else:
            self.MAKE_FUNCTION_NEW(addr, argc, is_closure)

    def LOAD_CLOSURE(self, addr, i):
        # Push the varname.  It doesn't matter as it is not used for now.
        self.stack.push(self.code.derefnames[i])

    def MAKE_CLOSURE(self, addr, argc):
        self.MAKE_FUNCTION(addr, argc, is_closure=True)

    #
    # Raising exceptions
    #

    def RAISE_VARARGS(self, addr, argc):
        # TODO: find out when argc is 2 or 3
        # Answer: In Python 3, only 0, 1, or 2 argument (see PEP 3109)
        if argc == 0:
            self.write("raise")
        elif argc == 1:
            exception = self.stack.pop()
            self.write("raise {}", exception)
        elif argc == 2:
            from_exc, exc = self.stack.pop(), self.stack.pop()
            self.write("raise {} from {}".format(exc, from_exc))
        else:
            raise Unknown

    def EXTENDED_ARG(self, addr, ext):
        # self.write("# ERROR: {} : {}".format(addr, ext) )
        pass

    def WITH_CLEANUP(self, addr, *args, **kwargs):
        # self.write("# ERROR: {} : {}".format(addr, args))
        pass

    def WITH_CLEANUP_START(self, addr, *args, **kwargs):
        pass

    def WITH_CLEANUP_FINISH(self, addr, *args, **kwargs):
        jaddr = addr.jump()
        return jaddr

    # Formatted string literals
    def FORMAT_VALUE(self, addr, flags):
        formatter = ''
        if flags == 1:
            formatter = '!s'
        elif flags == 2:
            formatter = '!r'
        elif flags == 3:
            formatter = '!a'
        elif flags == 4:
            formatter = f':{self.stack.pop().val}'
        val = self.stack.pop()
        f = PyFormatValue(val)
        f.formatter = formatter
        self.stack.push(f)

    def BUILD_STRING(self, addr, c):
        params = self.stack.pop(c)
        self.stack.push(PyFormatString(params))

    # Coroutines
    def GET_AWAITABLE(self, addr: Address):
        func: AwaitableMixin = self.stack.pop()
        func.is_awaited = True
        self.stack.push(func)
        yield_op = addr.seek_forward(YIELD_FROM)
        return yield_op[1]

    def BEFORE_ASYNC_WITH(self, addr: Address):
        with_addr = addr.seek_forward(SETUP_ASYNC_WITH)
        end_with = with_addr.jump()
        with_stmt = WithStatement(self.stack.pop())
        with_stmt.is_async = True
        d_with = SuiteDecompiler(addr[1], end_with)
        d_with.stack.push(with_stmt)
        d_with.run()
        with_stmt.suite = d_with.suite
        self.suite.add_statement(with_stmt)
        if sys.version_info <= (3, 4):
            assert end_with.opcode is WITH_CLEANUP
            assert end_with[1].opcode is END_FINALLY
            return end_with[2]
        else:
            assert end_with.opcode is WITH_CLEANUP_START
            assert end_with[1].opcode is GET_AWAITABLE
            assert end_with[4].opcode is WITH_CLEANUP_FINISH
            return end_with[5]

    def SETUP_ASYNC_WITH(self, addr: Address, arg):
        pass

    def GET_AITER(self, addr: Address):
        return addr[2]

    def GET_ANEXT(self, addr: Address):
        iterable = self.stack.pop()
        for_stmt = ForStatement(iterable)
        for_stmt.is_async = True
        jump_addr = addr[-1].jump()
        d_body = SuiteDecompiler(addr[3], jump_addr[-1])
        d_body.stack.push(for_stmt)
        d_body.run()
        jump_addr = jump_addr[-1].jump()
        new_start = jump_addr
        new_end = jump_addr[-2].jump()[-1]
        d_body.start_addr = new_start

        d_body.end_addr = new_end

        d_body.run()

        for_stmt.body = d_body.suite
        self.suite.add_statement(for_stmt)
        new_end = new_end.seek_forward(POP_BLOCK)
        return new_end


SuiteDecompiler.RESUME = SuiteDecompiler.__dict__["CONTINUE_LOOP"]
SuiteDecompiler.CALL_NO_KW = SuiteDecompiler.__dict__[
    "CALL_FUNCTION_VAR"
]


def make_dynamic_instr(cls):
    def method(self, addr):
        cls.instr(self.stack)

    return method


# Create unary operators types and opcode handlers
for op, name, ptn, prec in unary_ops:
    name = 'Py' + name
    tp = type(name, (PyUnaryOp,), dict(pattern=ptn, precedence=prec))
    globals()[name] = tp
    setattr(SuiteDecompiler, op, make_dynamic_instr(tp))

# Create binary operators types and opcode handlers
for op, name, ptn, prec, inplace_ptn in binary_ops:
    # Create the binary operator
    tp_name = 'Py' + name
    tp = globals().get(tp_name, None)
    if tp is None:
        tp = type(
            tp_name, (PyBinaryOp,), dict(pattern=ptn, precedence=prec)
        )
        globals()[tp_name] = tp

    setattr(SuiteDecompiler, 'BINARY_' + op, make_dynamic_instr(tp))
    # Create the in-place operation
    if inplace_ptn is not None:
        inplace_op = "INPLACE_" + op
        tp_name = 'InPlace' + name
        tp = type(tp_name, (InPlaceOp,), dict(pattern=inplace_ptn))
        globals()[tp_name] = tp
        setattr(SuiteDecompiler, inplace_op, make_dynamic_instr(tp))

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            'USAGE: {} <filename.pyc> [START [END=-1]]'.format(
                sys.argv[0]
            )
        )
        sys.exit(1)

    start = 0
    end = -1
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    if len(sys.argv) > 3:
        end = int(sys.argv[3])
    with open(sys.argv[1], "rb") as stream:
        code_obj = read_code(stream)
        code = Code(code_obj)
        dc = SuiteDecompiler(
            code.address(code.instr_list[start][0]),
            code.address(code.instr_list[end][0]),
        )
        try:
            dc.run()
        except Exception as e:
            print("Exception during dc.run():", e, file=sys.stderr)

        s = IndentString()
        dc.suite.display(s)

        print("\x0a".join(s.lines))
