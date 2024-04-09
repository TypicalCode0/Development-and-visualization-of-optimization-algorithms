#ifndef FUNCTION_H
#define FUNCTION_H
#include <iostream>
#include <vector>
#include <stack>
#include <cmath>
#include <cctype>
#include <stdexcept>

struct Operator {
    std::string str;
    int8_t priority;
    bool is_unary, is_binary;
    double (*calc1)(double);
    double (*calc2)(double,double);

    Operator() {}
    Operator(std::string str, int8_t priority) :
        str(str), is_unary(false), is_binary(false), priority(priority) {}
    Operator(std::string str, int8_t priority, double (*calc)(double)) :
        str(str), priority(priority), is_unary(true), is_binary(false), calc1(calc) {}
    Operator(std::string str, int8_t priority, double (*calc)(double,double)) :
        str(str), priority(priority), is_unary(false), is_binary(true), calc2(calc) {}
    Operator(std::string str, int8_t priority, double (*calc1)(double), double (*calc2)(double,double)) :
        str(str), priority(priority), is_unary(true), is_binary(true), calc1(calc1), calc2(calc2) {}
};

enum TokenType {
    NUMBER, OPERATOR, VARIABLE
};

struct Token {
    TokenType type;
    Operator op;
    double num;
    size_t var;

    Token(double num) : type(NUMBER), num(num) {}
    Token(Operator op) : type(OPERATOR), op(op) {}
    Token(size_t var) : type(VARIABLE), var(var) {}
};

struct Point {
    std::vector<double> x;
    Point(size_t ndim);
    Point(std::vector<double>&& x);
    Point(const Point& other);
    Point(const Point&& other);
    Point operator-(Point other) const;
    Point operator*(double d) const;
};

std::ostream& operator<<(std::ostream& out, const Token& t);
std::ostream& operator<<(std::ostream& out, const Point& p);

class Function {
public:
    Function() {}
    Function(std::string&& function);
    size_t get_ndim() const;
    void print_vars() const;
    void print_expr() const;
    double operator()(const Point& p) const;

private:
    size_t ndim;
    std::vector<std::string> variables;
    std::vector<Token> expression;
    
    std::vector<Token> tokenize(std::string& function);
    void parse(std::vector<Token>&& tokens);
};

#endif // FUNCTION_H
