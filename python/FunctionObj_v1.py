import sympy


class FunctionObj:
    def __init__(self, expression):  # expression в виде выражения
        self.exp = sympy.sympify(expression)
        self.variables = FunctionObj.get_unique_variables(expression)
        self.constraints = []
        self.border = None

    def add_border(self, x, y):
        if x > y:
            raise "Incorrect range"
        if self.border is not None:
            raise "Variable already has border"
        self.border = (x, y)

    def change_border(self, x, y):
        if x > y:
            raise "Incorrect range"
        if self.border is None:
            raise "Variable hasnt border"
        self.border = (x, y)

    def get_border(self):
        return self.border

    def check_input_values(self, values):
        if set(self.variables) != set(values) or not self.check_for_constraints(values) or self.border is None:
            return False
        return True

    @staticmethod
    def get_unique_variables(expression) -> list:
        set_garbage = set(" +-()*/^1234567890<>=.,")
        set_variables = set(expression) - set_garbage
        return list(set_variables)

    def add_constraint(self, constraint_expression):
        constraint_exp = sympy.sympify(constraint_expression)
        constraint_variables = FunctionObj.get_unique_variables(constraint_expression)
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
        return self.compute_despite_constraints(values)

    def delete_constaint(self, index):
        del self.constraints[index]

    def clear_all_constaint(self):
        self.constraints.clear()

    def update_expression(self, expression):
        self.exp = sympy.sympify(expression)
        self.variables = FunctionObj.get_variables(expression)

    def get_expression(self):
        return self.exp

    def get_variables(self) -> list:
        return self.variables

    def __str__(self):
        text = '\n'.join(list(map(lambda x: str(x[0]), self.constraints)))
        return f"Expression: {self.exp}\nConstaints:\n{text}"
