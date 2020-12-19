
class TestClass:

    def test_func1(x):
        x = x*(f(x) + 1 - x[1])
        x = (y, [z, t]), {1, 2, 3}
        t = {1:[x, y], 3:'x'}
        a.x.y, b[2] = b, a
        t = 1 <= x < 2 < y
        g(x, y + 1, x=12)
        x = a and (b or c)
        y = 1 if x else 2
        z = 1 if not x else 2
        a = b = 3
        x[y.z] = a, b = u
        if x:
            f(x)
            del x
        else:
            g(x)
            h[y] = 3
        if y:
            foo()
        if x:
            a()
            if z:
                a1()
            else:
                a2()
            b()
        elif y:
            b()
        else:
            c()
        x = a and b or c
        return "hello"
    def test_func2():
        if a and ((b and c and d) or e or f) and g: g()
        if a or (b and (c1 or c2) and d) or e: g()
        if a and b or c: g()
        if a or b and c: g()
        if a and (b or c): g()
    def test_func3():
        x = a and b or c
        x = a and (b1 or b2) and c or c
        x = (a and b) + (c or (not d and e))
    def test_func4():
        def f(x, y=2):
            return x + y if x else x - y
        g = lambda x: x + 1
    def test_func5(x):
        x += 2
        x[3] *= 10
    def test_func6(x):
        while f(x):
            if x and y:
                g(x)
            else:
                x + 2
            x += f(x, y=2)
        while a and b:
            while c and d:
                print(a, c)
    def test_func7(x):
        for i in x:
            print(i)
        for a, b in x:
            for c, (d, e) in a:
                print(a + c)
    def test_func8(x):
        for i in x:
            if i == 2:
                f()
            else:
                g()
        for i in x:
            if i:
                break
        while x:
            if x:
                f()
    def test_func9():
        try:
            x = 1
        except A:
            x = 2
        except B as b:
            x = 3
        try:
            x = 2
            y = 3
        except A:
            x = 5
        finally:
            z = 2
        try:
            frobz()
        except:
            bar()
        finally:
            frobn()
    def test_func10(fname):
        with open(fname) as f:
            for line in f:
                print(line)
        with x as y, s as t:
            bar()
    def test_func11():
        l = [x for x in y for z in x]
        l1 = [x for x in y if f(x)]
        s = {x + 1 for x, y in T}
        d = {x: y for x, y in f(a)}
    def test_func12():
        class A:
            def f(self): return 1
        class B(A, metaclass=MyType):
            bar = 12
            def __init__(self, x):
                self.x = x
    def test_func13():
        g = (x for x in y)
        f(y - 2 for x in S for y in f(x))
    def test_func14():
        def g(x):
            for i in x:
                yield f(i) + 2
            a = yield 5
            b = 1 + (yield 12)
    def test_func15(x, y):
        def f(z):
            return z + x
        def g(z):
            global x
            return z + x
        def h(z):
            nonlocal x
            x = 12
    def test_func16():
        if a:
            return
        if b:
            foo()
            if c:
                return
    #foo = SuiteDecompiler.POP_JUMP_IF
    def test_func17():
        if a:
            if b:
                f()
            elif c:
                g()
    def test_func18():
        if a:
            if b:
                f()
        elif c:
            g()
    def test_func19():
        assert a, b
    def test_func20():
        assert a
    def test_func21():
        raise
    def test_func22():
        @decorate
        def f(): pass
        @foo
        @bar.baz(3)
        class A: pass
    def test_func23():
        class B(A):
            def foo(): pass
            def bar(): pass
            
    def test_func24():
        c = 2
        while 1:
            if a:
                break
            if b:
                continue
            c = 1
            
    def test_func25():
        c = 2
        while not 1:
            if a:
                break
            if b:
                continue
            c = 1

    def test_func26():
        c = 2
        while c:
            if a:
                break
            if b:
                continue
            c = 1
            
    def test_func27():
        c = 2
        while c:
            if a:
                break
            if b:
                continue
            c = 1
            
    def test_func28():
        c = 2
        while 1:
            if a:
                break
            if b:
                continue
            if c == '\b':
                pw = pw[:-1]
            else:
                pw = pw + c                
            #c = 1
            
    def test_func29():
        c = 2
        while 1:
            if a:
                break
            if b:
                continue
            if c == '\b':
                pw = pw[:-1]
            else:
                pw = pw + c                
            c = 1
            
    def test_func30():
        result_set.update(((self._res_id_group_map.get(r, 0), r) for r in res_dict))

if __name__ == "__main__":
    import unpyc3
    import sys
    
    if len(sys.argv) == 1:
        import types
        import difflib
        # run through and compile all functions
        for k,func in TestClass.__dict__.items():
            if isinstance(func,types.FunctionType):
                code = unpyc3.Code(func.__code__)
                source = str(code.get_suite(include_declarations=False, look_for_docstring=True))

        import unpyc3_tests
        code = unpyc3.decompile(unpyc3_tests)
        compiled = compile(str(code), '<string>', 'exec')
        code2 = unpyc3.decompile(compiled)
        diff = difflib.unified_diff(str(code), str(code2), fromfile='original', tofile='converted')
        sys.stdout.writelines(diff)
