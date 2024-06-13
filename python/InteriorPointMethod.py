import aesara
import aesara.tensor as tensor
import sympy
from numpy import *
import numpy as np


class InteriorPointMethod:
    def __init__(self, func):
        self.num_var = len(func.get_unique_symbols())
        self.num_constraints = len(func.constraints)  # g(*args)<=0
        self.num_iteration = 20
        self.history = []

        self.mu = 0.1  # коэф. барьера
        self.nu = 10  # коэф. барьера
        self.Kkt_toler = 0.0001  # коэф. для KKT
        self.rho = 0.1  # для уменьшение nu
        self.apprx_fraction_boundary = 0.995
        self.eta = 0.0001  # коэф. размер шага
        self.diagonal_shift_coef = 0

        self.eps = 0.0000001
        self.coef_regul = np.sqrt(self.eps)

        self.x0 = np.random.randn(self.num_var)
        self.coords_history = [self.x0]

        self.func_vector_var = tensor.vector()  # создание вектора символьных переменных для функции
        self.constraints_vector = tensor.vector()  # создание вектора символьных переменных для ограничений
        self.lagrange_vector = tensor.vector()  # создание вектора символьных переменных для множителей Лагранжа
        self.func = func.get_expression()

        # заменяю символьные переменные изначального выражения для работы с func_vector_var
        self.symbols_function = func.get_unique_symbols()
        for ind, var in enumerate(self.symbols_function):
            self.func = self.func.subs(var, sympy.Symbol(f"self.func_vector_var[{ind}]"))

        self.func = eval(str(self.func))  # поскольку func изначально является Sympy объектом, то func безопасна

        self.constraints = tensor.zeros((self.num_constraints,))
        for i in range(self.num_constraints):
            constr = func.constraints[i][0]
            if constr.rel_op != '<=' or constr.rhs != 0:  # алгоритм работает только для ограничений типа g(*args)<=0
                raise ValueError
            constr = constr.lhs
            for ind, var in enumerate(self.symbols_function):
                constr = constr.subs(var, sympy.Symbol(f"self.func_vector_var[{ind}]"))
            self.constraints = tensor.set_subtensor(self.constraints[i], -eval(str(constr)))

        self.constraints_vector = tensor.vector()

    def minimize(self):
        # создание вектора символьных переменных для изменения множителей Лагранжа
        self.change_lagrange_vector = tensor.vector()
        self.free_memb_eq = tensor.matrix()  # матрица свободных членов системы уравнения
        self.coef_system_eq = tensor.matrix()  # матрица коэф. системы уравнения
        self.constraints_vector = tensor.vector()

        self.compile_functions()
        x = self.x0
        if self.num_constraints:
            s = self.init_slack(x)
        else:
            s = np.array([])
            self.mu = self.Kkt_toler

        if self.num_constraints:
            lda = self.init_lagrange(x)
            lda[lda < 0] = self.Kkt_toler
        else:
            lda = np.array([])

        kkt = self.KKT(x, s, lda)

        self.not_ok = False
        self.history = [x]
        for i in range(self.num_iteration):
            if all([np.linalg.norm(kkt[0]) <= self.Kkt_toler, np.linalg.norm(kkt[1]) <= self.Kkt_toler,
                    np.linalg.norm(kkt[2]) <= self.Kkt_toler]):
                break

            for j in range(self.num_iteration):
                mu_tol = np.max([self.Kkt_toler, self.mu])
                if all([np.linalg.norm(kkt[0]) <= mu_tol, np.linalg.norm(kkt[1]) <= mu_tol,
                        np.linalg.norm(kkt[2]) <= mu_tol]):
                    break
                g = -self.grad(x, s, lda)
                reg_hess_mx = self.regul_hessian_mx(self.hessian_func(x, s, lda))
                dz = self.sym_solve_cmp(reg_hess_mx, g.reshape(
                    (g.size, 1))).reshape((g.size,))

                if self.num_constraints:
                    dz[self.num_var + self.num_constraints:] = -dz[self.num_var + self.num_constraints:]
                    nu_thres = np.dot(self.barrier_cost_grad(x, s), dz[:self.num_var + self.num_constraints]) / (
                            1 - self.rho) / np.sum(np.abs(self.con(x, s)))
                    if self.nu < nu_thres:
                        self.nu = nu_thres

                alpha_smax = 1
                alpha_lmax = 1
                if self.num_constraints:
                    alpha_smax = self.step(s, dz[self.num_var:(self.num_var + self.num_constraints)])
                    alpha_lmax = self.step(lda, dz[(self.num_var + self.num_constraints):])
                x, s, lda = self.search(x, s, lda, dz, alpha_smax, alpha_lmax)

                kkt = self.KKT(x, s, lda)

                self.history.append(x)

                if self.not_ok:
                    with open("../tmp.txt", "w+") as file:
                        file.write('\n'.join([' '.join(list(map(str, h))) for h in list(self.history)]))
                    return

            if self.num_constraints:
                # обновление mu
                x_i = self.num_constraints * np.min(s * lda) / (np.dot(s, lda) + self.eps)
                new_mu = (0.1 * np.min([0.05 * (1.0 - x_i) / (x_i + self.eps), 2.0]) ** 3 *
                          np.dot(s, lda) / self.num_constraints)
                if new_mu < 0:
                    new_mu = 0
                self.mu = new_mu
        with open("../tmp.txt", "w+") as file:
            file.write('\n'.join([' '.join(list(map(str, h))) for h in list(self.history)]))
        return

    def regul_hessian_mx(self, hess_mx):  # регуляризация матрицы Гесс
        eigenvalue = self.eigh(hess_mx)  # собственные значения
        rcond = np.min(np.abs(eigenvalue)) / np.max(np.abs(eigenvalue))  # число обусловленности

        if rcond <= self.eps or self.num_constraints != np.sum(eigenvalue < -self.eps):
            if self.diagonal_shift_coef == 0.0:
                self.diagonal_shift_coef = np.sqrt(self.eps)
            else:
                self.diagonal_shift_coef = np.max([self.diagonal_shift_coef / 2, np.sqrt(self.eps)])
            hess_mx[:self.num_var, :self.num_var] += self.diagonal_shift_coef * np.eye(self.num_var)
            eigenvalue = self.eigh(hess_mx)
            while self.num_constraints != np.sum(eigenvalue < -self.eps):
                hess_mx[:self.num_var, :self.num_var] -= self.diagonal_shift_coef * np.eye(self.num_var)
                self.diagonal_shift_coef *= 10.0
                hess_mx[:self.num_var, :self.num_var] += self.diagonal_shift_coef * np.eye(self.num_var)
                eigenvalue = self.eigh(hess_mx)
        return hess_mx

    def compile_functions(self):
        function_gradient = tensor.grad(self.func, self.func_vector_var)
        function_hessian = aesara.gradient.hessian(cost=self.func, wrt=self.func_vector_var)

        if self.num_constraints:
            Sigma = tensor.basic.diag(self.lagrange_vector / (self.constraints_vector + self.eps))
            dci = aesara.gradient.jacobian(
                self.constraints, wrt=self.func_vector_var).reshape((self.num_constraints, self.num_var)).T

            constraint_hessian = aesara.gradient.hessian(cost=tensor.sum(self.constraints * self.lagrange_vector),
                                                         wrt=self.func_vector_var)

            con = tensor.zeros((self.num_constraints,))
            con = tensor.set_subtensor(con[0:], self.constraints - self.constraints_vector)

            jaco = tensor.zeros((self.num_var + self.num_constraints, self.num_constraints))
            jaco = tensor.set_subtensor(jaco[:self.num_var, 0:], dci)
            jaco = tensor.set_subtensor(jaco[self.num_var:, 0:], -tensor.eye(self.num_constraints))

        grad = tensor.zeros((self.num_var + 2 * self.num_constraints,))
        grad = tensor.set_subtensor(grad[:self.num_var], function_gradient)
        if self.num_constraints:
            grad = tensor.inc_subtensor(
                grad[:self.num_var], -tensor.dot(dci, self.lagrange_vector))
            grad = tensor.set_subtensor(grad[self.num_var:self.num_var + self.num_constraints], self.lagrange_vector -
                                        self.mu / (self.constraints_vector + self.eps))
            grad = tensor.set_subtensor(
                grad[self.num_var + self.num_constraints:], self.constraints - self.constraints_vector)

        phi = self.func
        if self.num_constraints:
            phi += self.nu * \
                   tensor.sum(tensor.abs(self.constraints - self.constraints_vector)) - \
                   self.mu * tensor.sum(tensor.log(self.constraints_vector))

        change_lagrange = tensor.dot(function_gradient, self.change_lagrange_vector[:self.num_var])
        if self.num_constraints:
            change_lagrange -= (self.nu * tensor.sum(tensor.abs(self.constraints - self.constraints_vector)) +
                                tensor.dot(self.mu / (self.constraints_vector + self.eps),
                                           self.change_lagrange_vector[self.num_var:]))

            lagrange_vector0 = tensor.dot(tensor.nlinalg.pinv(jaco[:self.num_var, :]),  # начальные значения
                                          function_gradient.reshape((self.num_var, 1)))  # множителей лагранжа
            lagrange_vector0 = lagrange_vector0.reshape((self.num_constraints,))

            init_slack = tensor.max(tensor.concatenate([
                self.constraints.reshape((self.num_constraints, 1)),
                self.Kkt_toler * tensor.ones((self.num_constraints, 1))
            ], axis=1), axis=1)

        barrier_cost_grad = tensor.zeros((self.num_var + self.num_constraints,))
        barrier_cost_grad = tensor.set_subtensor(
            barrier_cost_grad[:self.num_var], function_gradient)
        if self.num_constraints:
            barrier_cost_grad = tensor.set_subtensor(barrier_cost_grad[self.num_var:],
                                                     -self.mu / (self.constraints_vector + self.eps))

        if self.num_constraints:
            function_hessian -= constraint_hessian

        hessian = tensor.zeros((self.num_var + 2 * self.num_constraints, self.num_var + 2 * self.num_constraints))
        hessian = tensor.set_subtensor(hessian[:self.num_var, :self.num_var], tensor.triu(function_hessian))
        if self.num_constraints:
            hessian = tensor.set_subtensor(hessian[:self.num_var, (self.num_var + self.num_constraints):], dci)
            hessian = tensor.set_subtensor(hessian[self.num_var:(self.num_var + self.num_constraints),
                                           self.num_var:(self.num_var + self.num_constraints)], Sigma)
            hessian = tensor.set_subtensor(
                hessian[self.num_var:(self.num_var + self.num_constraints),
                (self.num_var + self.num_constraints):],
                -tensor.eye(self.num_constraints)
            )
        hessian = tensor.triu(hessian) + tensor.triu(hessian).T
        hessian = hessian - tensor.diag(tensor.diagonal(hessian) / 2.0)

        lin_soln = tensor.slinalg.solve(self.coef_system_eq, self.free_memb_eq)

        self.cost = aesara.function(inputs=[self.func_vector_var], outputs=self.func)

        self.barrier_cost_grad = aesara.function(inputs=[self.func_vector_var, self.constraints_vector],
                                                 outputs=barrier_cost_grad, on_unused_input='ignore')

        self.grad = aesara.function(inputs=[self.func_vector_var, self.constraints_vector, self.lagrange_vector],
                                    outputs=grad, on_unused_input='ignore')

        self.hessian_func = aesara.function(
            inputs=[self.func_vector_var, self.constraints_vector, self.lagrange_vector],
            outputs=hessian, on_unused_input='ignore'
        )

        self.phi_func = aesara.function(
            inputs=[self.func_vector_var, self.constraints_vector],
            outputs=phi, on_unused_input='ignore'
        )

        self.change_lagrange = aesara.function(
            inputs=[self.func_vector_var, self.constraints_vector, self.change_lagrange_vector],
            outputs=change_lagrange, on_unused_input='ignore'
        )

        self.eigh = aesara.function(
            inputs=[self.coef_system_eq],
            outputs=tensor.slinalg.eigvalsh(self.coef_system_eq, tensor.eye(self.coef_system_eq.shape[0])),
        )

        self.sym_solve_cmp = aesara.function(
            inputs=[self.coef_system_eq, self.free_memb_eq],
            outputs=lin_soln,
        )

        if self.num_constraints:
            self.con = aesara.function(inputs=[self.func_vector_var, self.constraints_vector],
                                       outputs=con, on_unused_input='ignore')
            self.jaco = aesara.function(inputs=[self.func_vector_var], outputs=jaco, on_unused_input='ignore')
            self.init_lagrange = aesara.function(inputs=[self.func_vector_var], outputs=lagrange_vector0)
            self.init_slack = aesara.function(inputs=[self.func_vector_var], outputs=init_slack)

    def step(self, x, dx):
        start = 0.0
        end = 1.0
        if np.all(x + end * dx >= (1.0 - self.apprx_fraction_boundary) * x):
            return end
        while end - start > self.eps:
            mid = (start + end) / 2
            if np.any(x + mid * dx < (1.0 - self.apprx_fraction_boundary) * x):
                end = mid
            else:
                start = mid
        return start

    def search(self, x0, s0, lda0, dz, alpha_smax, alpha_lmax):
        dx = dz[:self.num_var]
        if self.num_constraints:
            ds = dz[self.num_var:(self.num_var + self.num_constraints)]

        if self.num_constraints:
            dl = dz[(self.num_var + self.num_constraints):]
        else:
            dl = 0
            alpha_lmax = 0

        x = np.copy(x0)
        s = np.copy(s0)
        phi0 = self.phi_func(x0, s0)
        change_lagrange0 = self.change_lagrange(x0, s0, dz[:self.num_var + self.num_constraints])
        is_updated = False
        if self.num_constraints:
            if self.phi_func(x0 + alpha_smax * dx,
                             s0 + alpha_smax * ds) > phi0 + alpha_smax * self.eta * change_lagrange0:
                c_old = self.con(x0, s0)
                c_new = self.con(x0 + alpha_smax * dx, s0 + alpha_smax * ds)
                if np.sum(np.abs(c_new)) > np.sum(np.abs(c_old)):
                    A = self.jaco(x0).T
                    try:
                        dz_p = -self.sym_solve_cmp(A,
                            c_new.reshape((self.num_var + self.num_constraints, 1))
                        ).reshape((self.num_var + self.num_constraints,))
                    except:
                        dz_p = -np.linalg.lstsq(A, c_new, rcond=None)[0]
                    if (self.phi_func(x0 + alpha_smax * dx + dz_p[:self.num_var],
                                      s0 + alpha_smax * ds + dz_p[self.num_var:]) <=
                            phi0 + alpha_smax * self.eta * change_lagrange0):
                        alpha_corr = self.step(
                            s0, alpha_smax * ds + dz_p[self.num_var:])
                        if (self.phi_func(x0 + alpha_corr * (alpha_smax * dx + dz_p[:self.num_var]),
                                          s0 + alpha_corr * (alpha_smax * ds + dz_p[self.num_var:])) <=
                                phi0 + alpha_smax * self.eta * change_lagrange0):
                            is_updated = True
                if not is_updated:
                    alpha_smax *= self.apprx_fraction_boundary
                    alpha_lmax *= self.apprx_fraction_boundary
                    while self.phi_func(x0 + alpha_smax * dx, s0 + alpha_smax * ds) > phi0 + \
                            alpha_smax * self.eta * change_lagrange0:
                        if (np.sqrt(np.linalg.norm(alpha_smax * dx) ** 2 + np.linalg.norm(alpha_lmax * ds) ** 2) <
                                self.eps):
                            self.not_ok = True
                            return x0, s0, lda0
                        alpha_smax *= self.apprx_fraction_boundary
                        alpha_lmax *= self.apprx_fraction_boundary
            if is_updated:
                s = s0 + alpha_corr * (alpha_smax * ds + dz_p[self.num_var:])
            else:
                s = s0 + alpha_smax * ds
        else:
            if self.phi_func(x0 + alpha_smax * dx, s0) > phi0 + alpha_smax * self.eta * change_lagrange0:
                if not is_updated:
                    alpha_smax *= self.apprx_fraction_boundary
                    alpha_lmax *= self.apprx_fraction_boundary
                    while self.phi_func(x0 + alpha_smax * dx, s0) > phi0 + alpha_smax * self.eta * change_lagrange0:
                        if np.linalg.norm(alpha_smax * dx) < self.eps:
                            self.not_ok = True
                            return x0, s0, lda0
                        alpha_smax *= self.apprx_fraction_boundary
                        alpha_lmax *= self.apprx_fraction_boundary
        if is_updated:  # обновление
            x = x0 + alpha_corr * (alpha_smax * dx + dz_p[:self.num_var])
        else:
            x = x0 + alpha_smax * dx

        if self.num_constraints:
            lda = lda0 + alpha_lmax * dl
        else:
            lda = np.copy(lda0)
        return x, s, lda

    def KKT(self, x, s, lda):
        kkts = self.grad(x, s, lda)
        if self.num_constraints:
            k1 = kkts[:self.num_var]
            k2 = kkts[self.num_var:(self.num_var + self.num_constraints)] * s
            k3 = kkts[(self.num_var + self.num_constraints):]
            return k1, k2, k3
        return kkts[:self.num_var], 0, 0
