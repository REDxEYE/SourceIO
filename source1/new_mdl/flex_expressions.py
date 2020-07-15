class Value:
    def __init__(self, value, precedence):
        self.value = value
        self.precedence = precedence

    def __repr__(self):
        return f'{self.value:.4}'


class Neg(Value):
    def __repr__(self):
        return f'(-{self.value})'


class FetchController(Value):
    def __repr__(self):
        return f'{self.value}'


class FetchFlex(Value):
    def __repr__(self):
        return f'{self.value}'


class Expr:
    def __init__(self, lhs, rhs, precedence):
        self.left = lhs
        self.right = rhs
        self.precedence = precedence


class Add(Expr):
    def __repr__(self):
        arg1 = self.left if issubclass(self.left.__class__, Value) else f'({self.left})'
        arg2 = self.right if issubclass(self.right.__class__, Value) else f'({self.right})'
        return f'{arg1} + {arg2}'


class Sub(Expr):
    def __repr__(self):
        arg1 = self.left if issubclass(self.left.__class__, Value) else f'({self.left})'
        arg2 = self.right if issubclass(self.right.__class__, Value) else f'({self.right})'
        return f'{arg1} - {arg2}'


class Mul(Expr):
    def __repr__(self):
        arg1 = self.left if issubclass(self.left.__class__, Value) else f'({self.left})'
        arg2 = self.right if issubclass(self.right.__class__, Value) else f'({self.right})'
        return f'{arg1} * {arg2}'


class Div(Expr):
    def __repr__(self):
        arg1 = self.left if issubclass(self.left.__class__, Value) else f'({self.left})'
        arg2 = self.right if issubclass(self.right.__class__, Value) else f'({self.right})'
        return f'{arg1} / {arg2}'


class Function:
    def __init__(self, values, precedence):
        self.values = values
        self.precedence = precedence


class Max(Function):
    def __repr__(self):
        arg1 = self.values[0] if issubclass(self.values[0].__class__, Value) else f'({self.values[0]})'
        arg2 = self.values[1] if issubclass(self.values[1].__class__, Value) else f'({self.values[1]})'

        return f'max({arg1}, {arg2})'


class Min(Function):
    def __repr__(self):
        arg1 = self.values[0] if issubclass(self.values[0].__class__, Value) else f'({self.values[0]})'
        arg2 = self.values[1] if issubclass(self.values[1].__class__, Value) else f'({self.values[1]})'

        return f'min({arg1}, {arg2})'


class Combo(Function):
    def __repr__(self):
        res = '*'.join([f"{v}" if issubclass(v.__class__, Value) else f"({v})" for v in self.values])

        return res


class Dominator(Combo):
    def __repr__(self):
        dom = self.values[0]
        slv = '*'.join([str(v) for v in self.values[1:]])

        return f"({dom} * (1 - {slv}))"
