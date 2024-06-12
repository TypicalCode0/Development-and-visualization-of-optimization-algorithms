import sympy


class FunctionObj:
    def __init__(self, expression):  # expression в виде выражения
        self.exp = sympy.sympify(expression, evaluate=False)
        self.variables = self.get_unique_variables()
        self.constraints = []
        self.border = None

    def add_border(self, x, y):
        if x > y:
            print("Incorrect range")
            raise ValueError
        if self.border is not None:
            print("Variable already has border")
            raise ValueError
        self.border = (x, y)

    def change_border(self, x, y):
        if x > y:
            print("Incorrect range")
            raise ValueError
        if self.border is None:
            print("Variable hasn't border")
            raise ValueError
        self.border = (x, y)

    def get_border(self):
        return self.border

    def check_input_values(self, values):
        if set(self.variables) != set(values) or not self.check_for_constraints(values) or self.border is None:
            return False
        return True

    def get_unique_variables(self) -> list:
        return [str(i) for i in self.exp.free_symbols]

    def get_unique_symbols(self):
        return self.exp.free_symbols

    def add_constraint(self, constraint_expression):
        constraint_exp = sympy.sympify(constraint_expression, evaluate=False)
        constraint_variables = self.get_unique_variables()
        self.constraints.append((constraint_exp, constraint_variables))

    def check_for_constraints(self, values) -> bool:  # values = {'x' :4, 'y' : 0, .....}
        for constraint, variables in self.constraints:
            subs = {var: values[var] for var in values if var in self.variables}
            result = constraint.subs(subs)
            if not result:
                return False
        return True

    def compute_despite_constraints(self, values) -> float:  # values = {'x' : 4, 'y' : 0, .....}
        subs = {var: values[var] for var in values if var in self.variables}
        result = self.exp.subs(subs)
        return result

    def solve(self, values):
        if not self.check_input_values(values):
            return None
        res = self.compute_despite_constraints(values)
        if isinstance(res, sympy.core.add.Add):
            return None
        return res

    def delete_constraint(self, index):
        del self.constraints[index]

    def clear_all_constraint(self):
        self.constraints.clear()

    def update_expression(self, expression):
        self.exp = sympy.sympify(expression, evaluate=False)
        self.variables = self.get_unique_variables()

    def get_expression(self):
        return self.exp

    def get_variables(self) -> list:
        return self.variables

    def __str__(self):
        text = '\n'.join(list(map(lambda x: str(x[0]), self.constraints)))
        return f"Expression: {self.exp}\nConstraints:\n{text}"
