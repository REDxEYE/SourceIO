class Value:
    def __hash__(self):
        return hash(self.value)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, self.__class__):
            o: Value
            return self.value == o.value
        else:
            return False

    def __init__(self, value, ):
        self.value = value

    def __repr__(self):
        return f'{self.value:.4}'


class Neg(Value):
    def __repr__(self):
        return f'-{self.value}'


class FetchController(Value):
    def __repr__(self):
        return f'{self.value.replace(" ", "_")}'


class FetchFlex(Value):
    def __repr__(self):
        return f'{self.value.replace(" ", "_")}'


class Expr:

    def __hash__(self) -> int:
        return hash(self.left) + hash(self.right)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, self.__class__):
            o: Expr
            return self.left == o.left and self.right == o.right
        else:
            return False

    def __init__(self, lhs, rhs, ):
        self.left = lhs
        self.right = rhs


class Add(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, (Value, Function)) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, (Value, Function)) else f'({self.right})'
        return f'{arg1}+{arg2}'


class Sub(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, (Value, Function)) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, (Value, Function)) else f'({self.right})'
        return f'{arg2}-{arg1}'


class Mul(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, (Value, Function)) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, (Value, Function)) else f'({self.right})'
        return f'{arg1}*{arg2}'


class Div(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, (Value,Function)) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, (Value,Function)) else f'({self.right})'
        return f'{arg1}/{arg2}'


class Function:
    def __hash__(self) -> int:
        return hash(sum(map(hash, self.values)))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, self.__class__):
            o: Function
            return len(self.values) == len(o.values) and all(a == b for (a, b) in zip(self.values, o.values))
        else:
            return False

    def __init__(self, *values):
        self.values: list = list(values)


class CustomFunction(Function):
    def __init__(self, function_name: str, *values):
        super().__init__(*values)
        self.function_name = function_name

    def __repr__(self) -> str:
        return f'{self.function_name}({", ".join(map(str, self.values))})'


class Max(Function):
    def __repr__(self):
        arg1 = self.values[0] if isinstance(self.values[0], Value) else f'({self.values[0]})'
        arg2 = self.values[1] if isinstance(self.values[1], Value) else f'({self.values[1]})'

        return f'max({arg1},{arg2})'


class Min(Function):
    def __repr__(self):
        arg1 = self.values[0] if isinstance(self.values[0], Value) else f'({self.values[0]})'
        arg2 = self.values[1] if isinstance(self.values[1], Value) else f'({self.values[1]})'

        return f'min({arg1},{arg2})'


class Combo(Function):
    def __repr__(self):
        return f'combo({", ".join(map(str, self.values))})'


class Dominator(Combo):
    def __repr__(self):
        return f'dom({self.values[0]}, {", ".join(map(str, self.values[1:]))})'
