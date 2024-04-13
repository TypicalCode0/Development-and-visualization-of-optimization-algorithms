#include "function.h"

const std::vector<Operator> OPERATORS = {
    Operator("(",      -2),
    Operator(")",      -1),
    Operator("+",       0, [](double a)             { return a; },          [](double a, double b){ return a + b; }),
    Operator("-",       0, [](double a)             { return -a; },         [](double a, double b){ return a - b; }),
    Operator("^",       2, [](double a, double b)   { return pow(a,b); }),
    Operator("**",      2, [](double a, double b)   { return pow(a,b); }),
    Operator("*",       1, [](double a, double b)   { return a * b; }),
    Operator("/",       1, [](double a, double b)   { return a / b; }),
    Operator("%",       1, [](double a, double b)   { return fmod(a,b); }),
    Operator("abs",     0, [](double a)             { return fabs(a); }),
    Operator("sqrt",    0, [](double a)             { return sqrt(a); }),
    Operator("ln",      0, [](double a)             { return log(a); }),
    Operator("log10",   0, [](double a)             { return log10(a); }),
    Operator("exp",     0, [](double a)             { return exp(a); }),
    Operator("sin",     0, [](double a)             { return sin(a); }),
    Operator("cos",     0, [](double a)             { return cos(a); }),
    Operator("tan",     0, [](double a)             { return tan(a); }),
    Operator("arcsin",  0, [](double a)             { return asin(a); }),
    Operator("arccos",  0, [](double a)             { return acos(a); }),
    Operator("arctan",  0, [](double a)             { return atan(a); }),
    Operator("sinh",    0, [](double a)             { return sinh(a); }),
    Operator("cosh",    0, [](double a)             { return cosh(a); }),
    Operator("tanh",    0, [](double a)             { return tanh(a); })
};

std::ostream& operator<<(std::ostream& out, const Token& t) {
    switch (t.type) {
    case NUMBER:
        out << "  NUMBER " << t.num;
        break;
    case VARIABLE:
        out << "VARIABLE " << t.var;
        break;
    case OPERATOR:
        out << "OPERATOR " << t.op.str;
        break;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const Point& p) {
    for (double x : p.x) out << x << ' ';
    return out;
}

Point::Point(size_t ndim) {
    x.resize(ndim);
}

Point::Point(std::vector<double>&& x) {
    this->x = std::move(x);
}

Point Point::operator-(const Point other) const {
    Point res(*this);
    for (size_t i = 0; i < x.size(); ++i)
        res.x[i] -= other.x[i];
    return res;
}

Point Point::operator*(double d) const {
    Point res(*this);
    for (size_t i = 0; i < x.size(); ++i)
        res.x[i] *= d;
    return res;
}

Function::Function(std::string&& function) {
    parse(tokenize(function));
}

std::vector<Token> Function::tokenize(std::string& function) {
    std::vector<Token> tokens;
    size_t i = 0;
    bool operand = false;
    auto try_tokenize_number = [&](){
        std::string number = "";
        while (i < function.size() && (std::isdigit(function[i]) || function[i] == '.'))
            number += function[i++];
        if (number.empty()) return false;
        tokens.emplace_back(Token(std::stod(number)));
        operand = true;
        return true;
    };
    auto try_tokenize_operator = [&](){
        for (auto& op : OPERATORS) {
            if (i + op.str.size() <= function.size() && function.substr(i, op.str.size()) == op.str) {
                tokens.emplace_back(op);
                if (op.is_unary && op.is_binary) {
                    if (operand) tokens.back().op.is_unary = false;
                    else tokens.back().op.is_binary = false;
                }
                i += op.str.size();
                operand = op.str == ")";
                return true;
            }
        }
        return false;
    };
    auto try_tokenize_variable = [&](){
        std::string var = "";
        while (i < function.size() && (std::isalnum(function[i]) || function[i] == '_'))
            var += function[i++];
        if (var.empty()) return false;
        size_t v = 0;
        while (v < variables.size() && variables[v] != var) ++v;
        tokens.emplace_back(Token(v));
        if (v == variables.size())
            variables.emplace_back(var);
        operand = true;
        return true;
    };
    while (i < function.size()) {
        if (std::isblank(function[i])) { ++i; continue; }
        if (try_tokenize_number()) continue;
        if (try_tokenize_operator()) continue;
        if (!try_tokenize_variable()) {
            throw std::invalid_argument("Failed to tokenize function");
        }
    }
    ndim = variables.size();
    return tokens;
}

void Function::parse(std::vector<Token>&& tokens) {
    std::stack<Operator> operators;
    auto push_op = [&](){
        if (operators.top().str == "(") {
            throw std::invalid_argument("Failed to parse function");
        }
        expression.emplace_back(operators.top());
        operators.pop();
    };
    for (auto& token : tokens) {
        if (token.type == OPERATOR) {
            if (token.op.str != "(")
                while (!operators.empty() && token.op.priority <= operators.top().priority)
                    push_op();
            if (token.op.str == ")") {
                if (operators.empty()) {
                    throw std::invalid_argument("Failed to parse function");
                }
                operators.pop();
            }
            else operators.push(token.op);
        } else expression.emplace_back(token);
    }
    while (!operators.empty())
        push_op();
}

double Function::operator()(const Point& p) const {
    if (p.x.size() != variables.size()) {
        throw std::invalid_argument("Number of dimensions of given point and function do not match");
    }
    std::stack<double> calc;
    for (auto& token : expression) {
        switch (token.type) {
        case NUMBER:
            calc.push(token.num);
            break;
        case VARIABLE:
            calc.push(p.x[token.var]);
            break;
        case OPERATOR:
            if (calc.empty()) {
                throw std::runtime_error("Failed to calculate function");
            }
            double a = calc.top(); calc.pop();
            if (token.op.is_binary) {
                if (calc.empty()) {
                    throw std::runtime_error("Failed to calculate function");
                }
                double b = a;
                a = calc.top(); calc.pop();
                calc.push(token.op.calc2(a,b));
            } else calc.push(token.op.calc1(a));
            break;
        }
    }
    if (calc.size() != 1) {
        throw std::runtime_error("Failed to calculate function");
    }
    return calc.top();
}

size_t Function::get_ndim() const {
    return ndim;
}

void Function::print_vars() const {
    for (auto& v : variables)
        std::cout << v << ' ';
    std::cout << std::endl;
}

void Function::print_expr() const {
    for (auto& t : expression)
        std::cout << t << std::endl;
}
