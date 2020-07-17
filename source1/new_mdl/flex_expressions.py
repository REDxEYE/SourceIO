from .structs.flex import FlexController


class Value:
    def __init__(self, value, ):
        self.value = value

    def __repr__(self):
        return f'{self.value:.4}'


class Neg(Value):
    def __repr__(self):
        return f'(-{self.value})'


class FetchController(Value):
    def __repr__(self):
        if type(self.value) is FlexController:
            return f'{self.value.name}'
        return f'{self.value}'


class FetchFlex(Value):
    def __repr__(self):
        return f'{self.value}'


class Expr:
    def __init__(self, lhs, rhs, ):
        self.left = lhs
        self.right = rhs


class Add(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, Value) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, Value) else f'({self.right})'
        return f'{arg1} + {arg2}'


class Sub(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, Value) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, Value) else f'({self.right})'
        return f'{arg1} - {arg2}'


class Mul(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, Value) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, Value) else f'({self.right})'
        return f'{arg1} * {arg2}'


class Div(Expr):
    def __repr__(self):
        arg1 = self.left if isinstance(self.left, Value) else f'({self.left})'
        arg2 = self.right if isinstance(self.right, Value) else f'({self.right})'
        return f'{arg1} / {arg2}'


class Function:
    def __init__(self, *values):
        self.values = values


class Max(Function):
    def __repr__(self):
        arg1 = self.values[0] if isinstance(self.values[0], Value) else f'({self.values[0]})'
        arg2 = self.values[1] if isinstance(self.values[1], Value) else f'({self.values[1]})'

        return f'max({arg1}, {arg2})'


class Min(Function):
    def __repr__(self):
        arg1 = self.values[0] if isinstance(self.values[0], Value) else f'({self.values[0]})'
        arg2 = self.values[1] if isinstance(self.values[1], Value) else f'({self.values[1]})'

        return f'min({arg1}, {arg2})'


class Combo(Function):
    def __repr__(self):
        return '*'.join(
            [
                f"{v}" if isinstance(v, Value) else f"({v})"
                for v in self.values
            ]
        )


class Dominator(Combo):
    def __repr__(self):
        dom = self.values[0]
        slv = '*'.join([str(v) for v in self.values[1:]])

        return f"({dom} * (1 - {slv}))"
