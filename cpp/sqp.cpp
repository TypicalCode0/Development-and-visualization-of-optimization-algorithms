#include <string>
#include "function.cpp"
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>


enum Mode {
    ND = -2, NEED_GRAD = -1, START = 0, NEED_EVAL_F = 1, TOO_MANY_ITERATIONS = 3, CONSTRAINTS_INCOMPATIBLE = 4, 
    SINGULAR_MATRIX_E = 5, SINGULAR_MATRIX_C = 6, POSITIVE_DIRECTIONAL_DERIVATIVE = 8, MORE_THAN_MAX_ITERATIONS = 9, 
    NO_PLACE = 10};

struct InequalityConstraint {
    unsigned dim;             
    Function f;
    double tol = 1e-6;    
};

struct Stop {
    unsigned dim;
    double result = 1e-12, tol = 1e-8, relative_tol = 0.01;
    int count_evals = 0, maxeval = 10000;
    std::string stop_msg = "\nThe algorithm has been stopped\n";
};

void gradient(const Function &f, const Point& point, const size_t n, std::vector<double> &res, const double H = 1e-9) {
    Point delta(point);
    double y = f(point);
    for (size_t i = 0; i < n; ++i) {
        delta.x[i] += H;
        res[i] = (f(delta) - y) / H;
        delta.x[i] = point.x[i];
    }
}

class SQP {
public:
    int ndim, count_constraints;
    Function f;
    std::vector<InequalityConstraint> inequality_constraints; 
    double max_value_univers, min_value_univers, result = INFINITY, xnorm;
    Stop stop;
    std::vector<std::vector<double>> gradient_constraints;
    std::vector<double> constraints;
    std::vector<double> gradient_f;
    std::vector<double> m, semi_matrix, x, r, s, upper_bounds, lower_bounds, work;
    std::vector<int> additional;
    Point curr_values_vars;
    Point prev_values_vars;
    double f_curr_val, f_prev_val, h1 = 0, h2 = 0, h3 = 0, h4 = 0, tmp_f = 0, t0 = 0, alpha = 0, t = 0;
    Mode mode = START, prev_mode = START;
    int feasible_cur = 0, feasible = 0;
    double infeasibility_cur = INFINITY, infeasibility = INFINITY;
    int is_inexact = 0, count_reset_BFGS = 0, size_matrix = 0, n3 = 0, is_inconsistencies = 0, line = 0;

    SQP(int ndim, std::string math_expression, 
        int count_constraints, 
        std::vector<InequalityConstraint>&& inequality_constraints,
        double max_value_univers, double min_value_univers) {
        this->ndim = ndim;
        f = Function(std::move(math_expression));
        this->count_constraints = count_constraints;
        this->inequality_constraints = inequality_constraints;
        this->max_value_univers = max_value_univers;
        this->min_value_univers = min_value_univers;
        this->count_constraints = count_constraints;

        gradient_constraints = std::vector<std::vector<double>>(ndim + 1, std::vector<double>(count_constraints, 0));
        constraints = std::vector<double>(count_constraints, 0);
        gradient_f = std::vector<double>(ndim + 1, 0);
        curr_values_vars = Point(std::move(std::vector<double>(ndim, min_value_univers)));
        prev_values_vars = Point(curr_values_vars);
        int len_work_arr = 3 * ndim * count_constraints + 9 * count_constraints + 
                                 25/2 * ndim * ndim + 81/2 * ndim + 16;
        additional = std::vector<int>(count_constraints+ndim*2, 0);
        m = std::vector<double>(std::max(count_constraints, 1), 0);
        semi_matrix = std::vector<double>((ndim + 1) * ndim / 2 + 1, 0); 
        x = std::vector<double>(ndim, 0);
        r = std::vector<double>(2 * ndim + std::max(count_constraints, 1), 0);
        s = std::vector<double>(ndim + 1, 0);
        upper_bounds = std::vector<double>(ndim + 1, 0);
        lower_bounds = std::vector<double>(ndim + 1, 0);
        work = std::vector<double>(len_work_arr - std::max(count_constraints, 1) - (ndim + 1) * ndim / 2 - 1 - ndim - 2 * ndim - std::max(count_constraints, 1) - ndim - 1, 0);

        f_curr_val = f(curr_values_vars); f_prev_val = INFINITY;
        feasible = 0; feasible_cur = 1; infeasibility_cur = 0;
        gradient(f, curr_values_vars, ndim, gradient_f);
        ++stop.count_evals;
        if (abs(f_curr_val) <= DBL_MAX) {
            for (int i = 0; i < count_constraints; ++i) {
                constraints[i] = -inequality_constraints[i].f(curr_values_vars);
                std::vector<double> tmp(ndim);
                gradient(inequality_constraints[i].f, curr_values_vars, inequality_constraints[i].f.get_ndim(), tmp);
                for (int k = 0; k < ndim; ++k) {
                    gradient_constraints[k][i] = -tmp[k];
                }
                for (int j = 0; j < inequality_constraints[i].f.get_ndim(); ++j) {
                    infeasibility_cur = infeasibility_cur >= constraints[i] ? infeasibility_cur : constraints[i];
			        feasible_cur = feasible_cur && constraints[i] <= inequality_constraints[i].tol;
                }
            }
            if (feasible_cur || !feasible_cur || !feasible && infeasibility_cur < infeasibility) {
                result = f_curr_val;
                feasible = feasible_cur;
	            infeasibility = infeasibility_cur;
            }
        }
        prev_mode = mode;
        f_prev_val = f_curr_val;
        if (feasible && result <= stop.result) {
            return;
        }
        start_solution();
    }

private:
    void start_solution() {
        bool is_gradient_need = 0;
        do {
            sqp();
            switch (mode) {
            case NEED_GRAD:
                if (prev_mode == -2 && !is_gradient_need) {
                    break;
                }
            case ND:
                is_gradient_need = 1;
            case NEED_EVAL_F:
                feasible_cur = 1; infeasibility_cur = 0;
                f_curr_val = f(curr_values_vars);
                if (is_gradient_need) {
                    gradient(f, curr_values_vars, ndim, gradient_f);
                }
                ++stop.count_evals;
                if (f_curr_val != INFINITY) {
                    for (int i = 0; i < count_constraints; ++i) {
                        constraints[i] = -inequality_constraints[i].f(curr_values_vars);
                        if (is_gradient_need) {
                            std::vector<double> tmp(ndim);
                            gradient(inequality_constraints[i].f, curr_values_vars, inequality_constraints[i].f.get_ndim(), tmp);
                            for (int k = 0; k < ndim; ++k) {
                                gradient_constraints[k][i] = -tmp[k];
                            }
                        }
                        for (int j = 0; j < inequality_constraints[i].f.get_ndim(); ++j) {
                            infeasibility_cur = infeasibility_cur >= constraints[i] ? infeasibility_cur : constraints[i];
                            feasible_cur = feasible_cur && constraints[i] <= inequality_constraints[i].tol;
                        }
                    }
                    if (feasible_cur || !feasible_cur || !feasible && infeasibility_cur < infeasibility) {
                        result = f_curr_val;
                        feasible = feasible_cur;
                        infeasibility = infeasibility_cur;
                    }
                }
                break;
            case START:
                end_solution();
                return;
            default:
                end_solution();
                std::cout<<"Error "<<mode<<"\n";
                return;
            }
            prev_mode = mode;
            if (abs(f_curr_val) <= DBL_MAX && ((f_curr_val < result && (feasible_cur || !feasible)) || (!feasible && infeasibility_cur < infeasibility))) {
                result = f_curr_val;
                feasible = feasible_cur;
	            infeasibility = infeasibility_cur;
            }
            if (mode == NEED_GRAD) {
                if (f_prev_val < INFINITY * 0.99 && feasible) {
                    if (stop.tol > abs(f_curr_val - f_prev_val) || convergence_vars()) {
                        end_solution();
                        return;
                    }
                    f_prev_val = f_curr_val;
                    prev_values_vars = curr_values_vars;
                }
            }
            if (stop.count_evals >= stop.maxeval || feasible && result < stop.result) {
                end_solution();
                return;
            }
        } while (true);
    }

    bool convergence_vars() {
        double diff1 = 0, diff2 = 0;
        for (int i = 0; i < curr_values_vars.x.size(); ++i) {
            diff1 += abs(curr_values_vars.x[i] - prev_values_vars.x[i]);
            diff2 += abs(curr_values_vars.x[i]);
        }
        if (diff1 < stop.relative_tol * diff2) {
            return true;
        }
        for (int i = 0; i < curr_values_vars.x.size(); ++i) {
            if (abs(curr_values_vars.x[i] - prev_values_vars.x[i]) > stop.tol) {
                return false;
            }
        }
        return true;
    }

    void end_solution() {
        if (result >= INFINITY * 0.99) {
            if (f_curr_val == INFINITY * 0.99) {
                result = f_prev_val;
                curr_values_vars = prev_values_vars;
            } else {
                result = f_curr_val;
            }
        }
    }

    void sqp() {
        if (mode == NEED_GRAD) {
            for (int i = 0; i < ndim; ++i) {
                upper_bounds[i] = gradient_f[i] - get_scalar(count_constraints, &gradient_constraints[i][0], 1,  &r[0], 1) - lower_bounds[i];
            }
            double k = 0;
            for (int i = 0; i < ndim; ++i) {
                h1 = 0;
                ++k;
                for (int j = i + 1; j < ndim; ++j) {
                    ++k;
                    h1 += semi_matrix[k] * s[j];
                }
                lower_bounds[i] = s[i] + h1;
            }
            k = 0;
            for (int i = 0; i < ndim; ++i) {
                lower_bounds[i] *= semi_matrix[k];
                k = k + ndim - i;
            }
            for (int i = ndim - 1; i >= 0; --i) {
                h1 = 0.0;
                k = i;
                for (int j = 0; j < i - 1; ++j) {
                    h1 += semi_matrix[k] * lower_bounds[j];
                    k = k + ndim - j;
                }
                lower_bounds[i] += h1;
            }
            h1 = get_scalar(ndim, &s[0], 1, &upper_bounds[0], 1);
            h2 = get_scalar(ndim, &s[0], 1, &lower_bounds[0], 1);
            h3 = h2 * 0.2;
            if (h1 < h3) {
                h4 = (h2 - h3) / (h2 - h1);
                h1 = h3;
                vector_by_constant(ndim, &upper_bounds[0], h4, 1);
                axpy(ndim, &lower_bounds[0], 1, &upper_bounds[0], 1, 1 - h4);
            }
            update_ldl_matrix(semi_matrix, upper_bounds, 1 / h1, lower_bounds);
            update_ldl_matrix(semi_matrix, lower_bounds, -1 / h2, upper_bounds);
            search_direction();
            while (h3 >= 0.0) {
                if (reset_BFGS_matrix()) {
                    return;
                }
                search_direction();
            }
            line = 0;
            alpha = 1;
            linesearch();
        } else if (mode == START) {
            is_inexact = 0;
            count_reset_BFGS = 0;
            size_matrix = (ndim + 1) * ndim / 2;
            n3 = size_matrix + 1;
            s[0] = 0.0;
            copy_with_step(ndim, &s[0], 0, &s[0], 1);
            m[0] = 0.0;
            copy_with_step(count_constraints, &m[0], 0, &m[0], 1);
            do {
                if (reset_BFGS_matrix()) {
                    return;
                }
                search_direction();
            } while (h3 >= 0.0);
            line = 0;
            alpha = 1;
            linesearch();
        } else {
            t = f_curr_val;
            for (int i = 0; i < count_constraints; ++i) {
                t += m[i] * std::max(-constraints[i], 0.0);
            }
            h1 = t - t0;
            switch (is_inexact) {
            case 0:
                if (abs(h1) <= DBL_MAX) {
                    if (h1 <= h3 / 10 || line > 10) {
                        h3 = h1 = 0.0;
                        for (int i = 0; i < count_constraints; ++i) {
                            h3 += std::max(-constraints[i], 0.0);
                        }
                        if ((abs(f_curr_val - tmp_f) < 0 || compute_L2_norm(ndim, &s[0], 1) < 0) && h3 < 0) {
                            mode = START;
                        } else {
                            mode = NEED_GRAD;
                        }
                        return;
                    }
                    alpha = std::max(h3 / (2 * (h3 - h1)), 0.1);
                } else {
                    alpha = std::max(alpha * 0.5, 0.1);
                }
                linesearch();
                break;
            case 1:
                mode = MORE_THAN_MAX_ITERATIONS;
                break;
            }
        }
    }

    bool reset_BFGS_matrix() {
        if ((++count_reset_BFGS) > 5) {
            if ((abs(f_curr_val - tmp_f) < 0 || compute_L2_norm(ndim, &s[0], 1) < 0) && h3 < 0) {
                mode = START;
            } else {
                mode = POSITIVE_DIRECTIONAL_DERIVATIVE;
            }
            return 1;
        } else {
            semi_matrix[0] = 0.0;
            copy_with_step(size_matrix, &semi_matrix[0], 0, &semi_matrix[0], 1);
            int diagonal_i = 0;
            for (int i = 0; i < ndim; ++i) {
                semi_matrix[diagonal_i] = 1;
                diagonal_i = diagonal_i + ndim - i; 
            }
        }
        return 0;
    }

    void search_direction() {
        std::fill(upper_bounds.begin(), upper_bounds.end(), min_value_univers);
        std::fill(lower_bounds.begin(), lower_bounds.end(), max_value_univers);
        axpy(ndim, &curr_values_vars.x[0], 1, &upper_bounds[0], 1, -1);
        axpy(ndim, &curr_values_vars.x[0], 1, &lower_bounds[0], 1, -1);
        h4 = 1;
        std::cout<<"serch_direction\n";
        quadratic_programming();
        std::cout<<"serch_direction 2\n";
        if (mode == CONSTRAINTS_INCOMPATIBLE) {
            for (int i = 0; i < count_constraints; ++i) {
                gradient_f[i] = std::max(0.0, -constraints[i]);
            }
            s[0] = 0.0;
            copy_with_step(ndim, &s[0], 0, &s[0], 1);
            h3 = 0.0;
            gradient_f[ndim] = 0.0;
            semi_matrix[n3 - 1] = 100;
            s[ndim] = 1;
            upper_bounds[ndim] = 0.0;
            lower_bounds[ndim] = 1;
            is_inconsistencies = 0; 

            do {
                std::cout<<"QP while\n";
                quadratic_programming();
                std::cout<<"QP while 2\n";
                h4 = 1 - s[ndim + 1];
                if (mode == CONSTRAINTS_INCOMPATIBLE) {
                    semi_matrix[n3 - 1] *= 10;
                    ++is_inconsistencies;
                    if (is_inconsistencies > 5) {
                        return;
                    }
                }
            } while (mode == CONSTRAINTS_INCOMPATIBLE);

            if (mode != NEED_EVAL_F) {
                return;
            }
        } else if (mode != NEED_EVAL_F) {
            return;
        }
        for (int i = 0; i < ndim; ++i) {
            lower_bounds[i] = gradient_f[i] - get_scalar(count_constraints, &gradient_constraints[i][0], 1, &r[0], 1);
        }
        tmp_f = f_curr_val;
        copy_with_step(ndim, &curr_values_vars.x[0], 1, &x[0], 1);
        h1 = abs(get_scalar(ndim, &gradient_f[0], 1, &s[0], 1));
        h2 = 0.0;
        for (int i = 0; i < count_constraints; ++i) {
            h2 += std::max(-constraints[i], 0.0);
            h3 = abs(r[i]);
            m[i] = std::max(h3, (m[i] + h3) / 2);
            h1 += h3 * abs(constraints[i]);
        }
        mode = START;
        if (h1 < 0 && h2 < 0) {
            return;
        }
        h1 = h3 = 0.0;
        for (int i = 0; i < count_constraints; ++i) {
            h1 += m[i] * std::max(-constraints[i], 0.0);
        }
        t0 = f_curr_val + h1;
        h3 = abs(get_scalar(ndim, &gradient_f[0], 1, &s[0], 1)) - h1 * h4;
        mode = POSITIVE_DIRECTIONAL_DERIVATIVE;
    }

    void update_ldl_matrix(std::vector<double> &x, std::vector<double> &z, double sigma, std::vector<double> &w) {
        if (sigma == 0.0) {
            return;
        }
        double tmp = 1 / sigma, index = 0, tp;
        if (sigma < 0.0) {
            for (int i = 0; i < ndim; ++i) {
                w[i] = z[i];
            }
            for (int i = 0; i < ndim; ++i) {
                t += w[i] * w[i] / x[index];
                for (int j = i + 1; j < ndim; ++j) {
                    ++index;
                    w[j] -= w[i] * x[index];
                }
                ++index;
            }
            if (tmp >= 0.0) {
                t = 2.22E-16 / sigma;
            }
            for (int i = 0; i < ndim; ++i) {
                int j = ndim - i;
                index -= i;
                double _ = w[j];
                w[j] = t;
                t -= _ * _ / x[index];
            }
        }
        for (int i = 0; i < ndim; ++i) {
            double delta = z[i] / x[index];
            if (sigma < 0.0) {
                tp = w[i];
            } else {
                tp = t + delta + z[i];
            }
            alpha = tp / t;
            x[index] *= alpha;
            if (i == ndim - 1) {
                return;
            }
            double beta = delta / tp;
            if (alpha > 4) {
                double gamma = t / tp;
                for (int j = i + 1; j < ndim; ++j) {
                    ++index;
                    x[index] = gamma * x[index] + beta * z[j];
                    z[j] -= x[index] * z[i];
                }
            } else {
                for (int j = i + 1; j < ndim; ++j) {
                    ++index;
                    z[j] -= z[i] * x[index];
                    x[index] += beta * z[j];
                }
            }
            ++index;
            t = tp;
        }

    }

    void linesearch() {
        ++line;
        h3 = alpha * h3;
        vector_by_constant(ndim, &s[0], alpha, 1);
        copy_with_step(ndim, &x[0], 1, &curr_values_vars.x[0], 1);
        axpy(ndim, &s[0], 1, &curr_values_vars.x[0], 1, 1);
        for (int i = 0; i < ndim; ++i) {
            if (curr_values_vars.x[i] < min_value_univers) {
                curr_values_vars.x[i] = min_value_univers;
            } else if (curr_values_vars.x[i] > max_value_univers) {
                curr_values_vars.x[i] = max_value_univers;
            }
        }
        mode = line == 1 ? ND : NEED_EVAL_F;
    }

    void axpy(int n, double *x, int step_x, double *y, int step_y, double a ) {
        for (int i = 0; i < n; ++i) {
            y[i] += a * x[i];
        }
    }

    double get_scalar(int n, double *x, int step_x, double *y, int step_y) {
        double res = 0;
        if (n <= 0) return res;
        for (int i = 0; i < n; ++i) {
            res += x[i * step_x] * y[i * step_y];
        }
        return res;
    }

    void vector_by_constant(int n, double *x, double alpha, int step) {
        for (int i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    double compute_L2_norm(int n, double *x, int step) {
        double max = 0, sum = 0;
        for (int i = 0; i < n; ++i) {
            if (max < abs(x[i*step])) {
                max = abs(x[i*step]);
            }
        }
        if (max == 0) {
            return 0;
        }
        double scale = 1.0 / max;
        for (int i = 0; i < n; ++i) {
            sum += scale*x[i*step] * scale*x[i*step];
        }
        return max * sqrt(sum);
    }

    void quadratic_programming() {
        int is_inconsistent;
        if ((ndim + 1) * ndim / 2 + 1 == n3) {
            is_inconsistent = 0;
        } else {
            is_inconsistent = 1;
        }
        int i_from = 0, i_work = 0, i_semi_matrix = 0;
        for (int i = 0; i < ndim - is_inconsistent; ++i) {
            work[i_work] = 0.0;
            copy_with_step(ndim - i, &work[i_work], 0, &work[i_work], 1);
            copy_with_step(ndim - i - is_inconsistent, &semi_matrix[i_semi_matrix], 1, &work[i_work], ndim);
            vector_by_constant(ndim - i - is_inconsistent, &work[i_work], sqrt(semi_matrix[i_semi_matrix]), ndim);
            work[i_work] = sqrt(semi_matrix[i_semi_matrix]);
            work[ndim * ndim + i] = (gradient_f[i] - get_scalar(i, &work[i_from], 1, &work[ndim * ndim], 1)) / sqrt(semi_matrix[i_semi_matrix]);
            i_semi_matrix = i_semi_matrix + ndim - i;
            i_work += ndim + 1;
            i_from += ndim;
        }
        if (is_inconsistent == 1) {
            work[i_work] = semi_matrix[n3];
            work[i_from] = 0.0;
            copy_with_step(ndim - is_inconsistent, &work[i_from], 0, &work[i_from], 1);
            work[ndim * ndim + ndim - 1] = 0.0;
        }
        vector_by_constant(ndim, &work[ndim * ndim], -1, 1);
        for (int i = 0; i < count_constraints; ++i) {
            copy_with_step(ndim, &gradient_constraints[0][i], count_constraints, &work[ndim * ndim  + ndim + i], count_constraints + 2 * ndim);
        }
        for (int i = 0; i < ndim; ++i) {
            int j = ndim * ndim + ndim + count_constraints + i;
            work[j] = 0.0;
            copy_with_step(ndim, &work[j], 0, &work[j], count_constraints + 2 * ndim);
        }
        for (int i = 0; i < ndim; ++i) {
            if (upper_bounds[i] < INFINITY * 0.99) {
                int j = ndim * ndim - ndim + i * (count_constraints + 2 * ndim + 1) - 1;
                work[j] = 1.0;
            }
        }
        for (int i = 0; i < ndim; ++i) {
            int j = ndim * ndim + 2 * ndim + count_constraints + i;
            work[j] = 0.0;
            copy_with_step(ndim, &work[j], 0, &work[j], count_constraints + 2 * ndim);
        }
        for (int i = 0; i < ndim; ++i) {
            if (lower_bounds[i] < INFINITY * 0.99) {
                int j = ndim * ndim + i * (count_constraints + 2 * ndim + 1) - 1;
                work[j] = -1.0;
            }
        }
        int j = ndim * ndim + 1 + ndim + (count_constraints + 2 * ndim) * ndim;
        if (count_constraints > 0) {
            copy_with_step(count_constraints, &constraints[0], 1, &work[j], 1);
            vector_by_constant(count_constraints, &work[j], -1.0, 1);
        }
        for (int i = 0; i < ndim; ++i) {
            work[j + count_constraints - 1 + i] = upper_bounds[i] >= INFINITY * 0.99 ? 0 : upper_bounds[i];
            work[j + count_constraints + ndim - 1 + i] = lower_bounds[i] >= INFINITY * 0.99 ? 0 : -lower_bounds[i];
        }
        std::cout<<"QP\n";
        least_squares(&work[ndim + ndim*ndim], &work[ndim + ndim*ndim], &work[ndim*ndim], &work[ndim*ndim + ndim], &work[j - 1], &work[j + count_constraints + ndim + ndim - 1], count_constraints + ndim * 2);
        std::cout<<"QP 2\n";
        if (mode == NEED_EVAL_F) {
            j += ndim + ndim + count_constraints;
            copy_with_step(count_constraints, &work[j - 1], 1, &r[0], 1);
            copy_with_step(ndim - is_inconsistent, &work[j + count_constraints - 1], 1, &r[count_constraints], 1);
            copy_with_step(ndim - is_inconsistent, &work[j + count_constraints + ndim - 1], 1, &r[count_constraints + ndim - is_inconsistent], 1);
            for (int i = 0; i < ndim; ++i) {
                if (s[i] < upper_bounds[i]) {
                    s[i] = upper_bounds[i];
                } else if (s[i] > lower_bounds[i]) {
                    s[i] = lower_bounds[i];
                }
            }
        }
    }

    void least_squares(double* c, double *d, double *f, double *g, double *h, double *w, int m) {
        mode = NEED_EVAL_F;
        w[0] = 0.0;
        copy_with_step(m, &w[0], 0, &w[0], 1);
        for (int i = 0; i < ndim; ++i) {
            int j = (ndim + 1) * (m + 2) + (m<<1) + 1 + ndim * ndim;
            w[j - 1 + i] = f[i];
        }
        for (int i = 0; i < ndim; ++i) {
            int j = (ndim + 1) * (m + 2) + (m<<1) + 1;
            copy_with_step(ndim, &work[i], ndim, &w[j - 1 + i], ndim);
        }
        for (int i = 0; i < m; ++i) {
            int j = (ndim + 1) * (m + 2) + (m<<1) + 1 + ndim * ndim + ndim;
            copy_with_step(ndim, &g[i], m, &w[j - 1 + i], m);
        }
        std::cout<<"least_squares\n";
        least_squares_with_inequalities(&w[(ndim + 1) * (m + 2) + (m<<1)], &w[(ndim + 1) * (m + 2) + (m<<1) + ndim * ndim], &w[(ndim + 1) * (m + 2) + (m<<1) + ndim * ndim + ndim], h, &w[0]);
        std::cout<<"least_squares 2\n";
    }

    void least_squares_with_inequalities(double *e, double *f, double *g, double *h, double *w) {
        double tmp;
        for (int i = 0; i < ndim; ++i) {
            int j = std::min(i + 1, ndim - 1);
            std::cout<<e[0]<<"\t"<<e[3]<<"\n";
            transformation_matrix_E(1, i, i + 1, ndim, &e[i * ndim], 1, tmp, &e[j * ndim], 1, ndim, ndim - i);
            std::cout<<"after 1 "<<e[0]<<"\t"<<e[3]<<"\n";
            transformation_matrix_E(2, i, i + 1, ndim, &e[i * ndim], 1, tmp, &f[0], 1, 1, 1);
            std::cout<<"after 2 "<<e[0]<<"\t"<<e[3]<<"\n";
        }
        mode = SINGULAR_MATRIX_E;
        for (int i = 0; i < count_constraints + 2 * ndim; ++i) {
            int dim = count_constraints + 2 * ndim;
            for (int j = 0; j < ndim; ++j) {
                std::cout<<"e = "<<e[j + j * ndim]<<"\n";
                if (abs(e[j + j * ndim]) < 2.22E-16) {
                    return;
                }
                g[i + j * dim] = (g[i + j * dim] - get_scalar(j, &g[i + dim], dim, &e[j * ndim], 1) / e[j + j * ndim]);
            }
            h[i] -= get_scalar(ndim, &g[i + dim], dim, &f[0], 1);
        }
        std::cout<<"least_squares_with_inequalities\n";
        minimization_quadratic_form(&g[0], count_constraints + ndim * 2, ndim, &h[0], &w[0]);
        std::cout<<"least_squares_with_inequalities 2\n";
        if (mode != NEED_EVAL_F) {
            return;
        }
        axpy(ndim, &f[0], 1, &s[0], 1, 1);
        for (int i = ndim - 1; i >= 0; --i) {
            int min = std::min(i + 1, ndim - 1);
            s[i] = (s[i] - get_scalar(ndim - i, &e[i + min * ndim], ndim, &s[min - 1], 1)) / e[i + i * ndim];
        }
        xnorm = sqrt(xnorm * xnorm);
    }

    void transformation_matrix_E(const int mode, int pivot, int l, int size, double *vec, const int dim_vec, double &tmp, double *matrix, const int step_col, const int step_row, const int cnt_rows) {
        if (pivot < 0 || pivot >= l || l > size) {
            return;
        }
        double cl = abs(vec[pivot * dim_vec]);
        for (int i = l - 1; i < size; ++i) {
            cl = std::max(abs(vec[i * dim_vec]), cl);
        }
        if (cl <= 0.0) {
            return;
        }
        double clinv = 1 / cl;
        double sm = vec[pivot * dim_vec] * clinv * vec[pivot * dim_vec] * clinv;
        for (int i = l - 1; i < size; ++i) {
            sm += vec[i * dim_vec] * clinv * vec[i * dim_vec] * clinv;
        }
        cl *= sqrt(sm);
        if (vec[pivot * dim_vec] > 0.0) {
            cl -= cl;
        }
        tmp = vec[pivot * dim_vec] - cl;
        vec[pivot * dim_vec] = cl;
        if (cnt_rows <= 0) {
            return;
        }
        double t = tmp * vec[pivot * dim_vec];
        if (t >= 0.0) {
            return;
        }
        t = 1 / t;
        int i2 = 1 - step_row + step_col * (pivot - 1), inc = step_col * (l - pivot), i3;
        for (int i = 0; i < cnt_rows; ++i) {
            i2 += step_row;
            i3 = i2 + inc;
            sm = matrix[i2 - 1] * tmp;
            for (int j = l - 1; j < size; ++j) {
                sm += matrix[i3 - 1] * vec[j * dim_vec];
                i3 += step_col;
            }
            if (sm == 0.0) {
                continue;
            }
            sm *= t;
            matrix[i2 - 1] += sm * tmp;
            for (int j = l; j < size; ++j) {
                matrix[i3 - 1] += sm * vec[i * dim_vec];
                i3 += step_col;
            }
        }
    }

    void minimization_quadratic_form(double *g, int m, int n, double* h, double *w) {
        if (n <= 0) {
            return;
        }
        mode = NEED_EVAL_F;
        s[0] = 0.0;
        copy_with_step(ndim, &s[0], 0, &s[0], 1);
        xnorm = 0;
        if (m == 0) {
            return;
        }
        int index = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < ndim; ++j) {
                ++index;
                w[index] = g[i + j * m];
            }
            ++index;
            w[index] = h[i];
        }
        int index_f = index + 1;
        for (int i = 0; i < n; ++i) {
            ++index;
            w[index] = 0.0;
        }
        w[index + 1] = 1;
        double norm;
        std::cout<<"minimization_quadratic_form\n";
        nnls_(&w[0], n + 1, m, &w[index_f], &w[index + 2 + n + 1], norm, &w[index + 2 + n + 1 + m], &w[index + 2]);
        std::cout<<"minimization_quadratic_form 2\n";
        if (mode != NEED_EVAL_F) {
            return;
        }
        mode = CONSTRAINTS_INCOMPATIBLE;
        if (norm <= 0.0) {
            return;
        }
        if (2 - get_scalar(m, &h[0], 1, &w[index + 2 + ndim + 1], 1) <= 0.0) {
            return;
        }
        mode = NEED_EVAL_F;
        double tmp = 1 / (2 - get_scalar(m, &h[0], 1, &work[index + 2 + ndim + 1], 1));
        for (int i = 0; i < ndim; ++i) {
            s[i] = tmp * get_scalar(m, &g[i * m], 1, &w[index + 2 + ndim + 1], 1);
        }
        xnorm = compute_L2_norm(ndim, &s[0], 1);
        w[0] = 0.0;
        copy_with_step(m, &w[0], 0, &w[0], 1);
        axpy(m, &w[index + 2 + ndim + 1], 1, &w[0], 1, tmp);
    }

    void nnls_(double* a, int m, int n, double *b, double *x, double &norm, double *w, double *z) {
        if (m <= 0 || n <= 0) {
            return;
        }
        mode = NEED_EVAL_F;
        int iter = 0, iter_max = n * 3;
        for (int i = 0; i < n; ++i) {
            additional[i] = i;
        }
        x[0] = 0.0;
        copy_with_step(n, &x[0], 0, &x[0], 1);
        int z1 = 0, z2 = n - 1, nsetp = 0, npp1 = 0, z_max, j;
        double tmp;
        do {
            if (z1 > z2 || nsetp >= m) {
                solution(m, n, b, w, norm, npp1, nsetp);
                return;
            }
            for (int i = 0; i <= z2; ++i) {
                w[additional[i]] = get_scalar(m - nsetp, &a[npp1 + additional[i] * m], 1, &b[npp1], 1);
            }
            do {
                double w_max = 0.0;
                for (int i = z1; i <= z2; ++i) {
                    if (w[additional[i]] <= w_max) {
                        continue;
                    }
                    w_max = w[additional[i]];
                    z_max = i;
                }
                if (w_max <= 0.0) {
                    solution(m, n, b, w, norm, npp1, nsetp);
                    return;
                }
                double save = a[npp1 + additional[z_max] * m];
                transformation_matrix_E(1, npp1, npp1 + 1, m, &a[additional[z_max] * m], 1, tmp, &z[0], 1, 1, 0);
                if (0.01 * abs(a[npp1 + additional[z_max] * m]) > 0.0) {
                    copy_with_step(m, &b[0], 1, &z[0], 1);
                    transformation_matrix_E(2, npp1, npp1 + 1, m, &a[additional[z_max] * m], 1, tmp, &z[0], 1, 1, 1);
                    if (x[npp1] / a[npp1 + additional[z_max] * m] > 0.0) {
                        break;
                    }
                }
                a[npp1 + additional[z_max] * m] = save;
                w[additional[z_max]] = 0.0;
            } while (true);
            copy_with_step(m, &z[0], 1, &b[0], 1);
            std::swap(additional[z_max], additional[z1]);
            ++z1;
            nsetp = npp1;
            ++npp1;
            for (int i = z1; i <= z2; ++i) {
                j = additional[i];
                transformation_matrix_E(2, nsetp, npp1, m, &a[additional[z_max] * m + 1], 1, tmp, &a[j * m + 1], 1, m, 1);
            }
            w[additional[z_max]] = 0.0;
            copy_with_step(m - nsetp - 1, &w[additional[z_max]], 0, &a[std::min(npp1, m) + additional[z_max] * m], 1);
            do {
                for (int i = nsetp; i >= 0; --i) {
                    if (i != nsetp) {
                        axpy(i, &a[j * m + 1], 1, &z[0], 1, -z[i + 1]);
                    }
                    j = additional[i];
                    z[i] /= a[i + j * m];
                }
                ++iter;
                if (iter > iter_max) {
                    mode = TOO_MANY_ITERATIONS;
                    solution(m, n, b, w, norm, npp1, nsetp);
                    return;
                }
                double alpha = 1, t;
                j = 0;
                for (int i = 0; i < nsetp; ++i) {
                    if (z[i] <= 0.0) {
                        t = -x[additional[i]] / (z[i] - z[additional[i]]);
                        if (alpha >= t) {
                            alpha = t;
                            j = i;
                        }
                    }
                }
                for (int i = 0; i < nsetp; ++i) {
                    x[additional[i]] = (1 - alpha) * x[additional[i]] + alpha * z[i];
                }
                if (j == 0) {
                    break;
                }
                do {                
                    x[additional[j]] = 0.0;
                    ++j;
                    double c, s;
                    for (int i = j; i < nsetp; ++i) {
                        additional[i - 1] = additional[i];
                        construct_givens_rotations(&a[i - 1 + additional[i] * m], &a[i + additional[i] * m], c, s);
                        t = a[i - 1 + additional[i] * m];
                        apply_givens_rotation(n, &a[i - 1], m, &a[i], m, c, s);
                        a[i - 1 + additional[i] * m] = t;
                        a[i + additional[i] * m] = 0.0;
                        apply_givens_rotation(1, &b[i - 1], 1, &b[i], 1, c, s);
                    }
                    npp1 = nsetp;
                    --nsetp;
                    --z1;
                    additional[z1] = additional[j];
                    if (nsetp <= 0) {
                        mode = TOO_MANY_ITERATIONS;
                        solution(m, n, b, w, norm, npp1, nsetp);
                        return;
                    }
                    bool is_pass = 1;
                    for (j = 0; j < nsetp; ++j) {
                        if (x[additional[j]] <= 0.0) {
                            is_pass = 0;
                            break;
                        }
                    }
                    if (is_pass) {
                        break;
                    }
                } while (true);
                copy_with_step(m, &b[0], 1, &z[0], 1);
            } while (true);
        } while (true);
        
    }

    void solution(int m, int n, double *b, double *w, double &norm, int npp1, int nsetp) {
        norm = compute_L2_norm(m - nsetp, &b[std::min(npp1, m)], 1);
        if (npp1 > m) {
            w[0] = 0.0;
            copy_with_step(n, &w[0], 0, &w[0], 1);
        }
    }

    void apply_givens_rotation(int n, double *dx, int step_x, double *dy, int step_y,  double &c, double &s) {
        for (int i = 0; i < n; ++i) {
            double x = dx[step_x * i], y = dy[step_y * i];
            dx[step_x * i] = c * x + s * y;
            dy[step_y * i] = c * y - s * x;
        }
    }

    void construct_givens_rotations(double *da, double* db, double &c, double &s) {
        double absa, absb, roe, scale;
        absa = abs(*da);
        absb = abs(*db);
        if (absa > absb) {
            roe = *da;
            scale = absa;
        } else {
            roe = *db;
            scale = absb;
        }
        if (scale != 0) {
            double r, iscale = 1 / scale;
            double tmpa = (*da) * iscale, tmpb = (*db) * iscale;
            r = (roe < 0 ? -scale : scale) * sqrt((tmpa * tmpa) + (tmpb * tmpb));
            c = *da / r;
            s = *db / r;
            *da = r;
            if (c != 0 && abs(c) <= s) {
                *db = 1 / c;
            } else {
                *db = s;
            }
        } else {
            c = 1;
            s = *da = *db = 0;
        }
    }

    void copy_with_step(int n, double *x, int mul_x, double *y, int mul_y) {
        for (int i = 0; i < n; ++i) {
            y[i * mul_y] = x[i * mul_x];
        }
    }
};


int main(int argc, char* argv[]) {
    InequalityConstraint ine;
    ine.dim = 1;
    ine.f = Function("1 - x - y");
    std::string s = "(x - 2)**2 + (y - 3)**2";
    std::vector<InequalityConstraint> ines;
    ines.push_back(ine);
    SQP a(2, s, 1, std::move(ines), 20, 0);
    std::cout<<"x = "<<a.curr_values_vars.x[0]<<" y = "<<a.curr_values_vars.x[1]<<"\n";
    std::cout<<"result = "<<a.result<<"\n";
    std::cout<<"mode = "<<a.mode;
}