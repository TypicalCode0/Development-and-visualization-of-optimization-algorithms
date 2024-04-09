#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include "function.h"

#define sqr(x) ((x)*(x))

const double D = 1e-6; // for stopping gradient descent
const double H = 1e-9; // for calculating gradient
std::mt19937 engine;
Function f;
size_t n;
bool LOG;

int64_t now() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

double dist(const Point& p1, const Point& p2) {
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d += sqr(p1.x[i] - p2.x[i]);
    return sqrt(d);
}

Point gradient(const Point& point) {
    Point res(n);
    Point delta(point);
    double y = f(point);
    for (size_t i = 0; i < n; ++i) {
        delta.x[i] += H;
        res.x[i] = (f(delta) - y) / H;
        delta.x[i] = point.x[i];
    }
    return res;
}

std::vector<Point> gradientDescent(double left_border, double right_border, double step_len, size_t max_steps_count) {
    std::vector<Point> points(1, Point(n));
    auto range = std::uniform_real_distribution<double>(left_border, right_border);
    for (size_t i = 0; i < n; ++i)
        points[0].x[i] = range(engine);
    if (max_steps_count == 0) return points;
    size_t step = 0;
    do {
        points.emplace_back(points.back() - gradient(points.back()) * step_len);
        for (double x : points.back().x)
            if (left_border > x || x > right_border) {
                points.pop_back();
                break;
            }
        ++step;
    } while (dist(points[points.size()-2], points.back()) > D && step < max_steps_count);
    if (LOG) {
        std::cout << "Start point: " << points[0] << std::endl;
        std::cout << "End point: " << points.back() << std::endl;
        std::cout << "Steps: " << step << std::endl;
    }
    return points;
}

int main(int argc, char* argv[]) { // Arguments: function as a string, left border, right_border, step length, maximum number of steps[, LOG]
    if (argc <= 5) {
        throw std::invalid_argument("Not enough arguments");
    }
    engine.seed((unsigned int)now());
    f = Function(argv[1]);
    n = f.get_ndim();
    double left_border = std::stod(argv[2]);
    double right_border = std::stod(argv[3]);
    double step_len = std::stod(argv[4]);
    size_t max_steps_count = std::stoi(argv[5]);
    LOG = argc > 6 && std::string(argv[6]) == "LOG";
    if (LOG) {
        std::cout << "Expression:" << std::endl;
        f.print_expr();
        std::cout << "Variables: ";
        f.print_vars();
    }
    auto v = gradientDescent(left_border, right_border, step_len, max_steps_count);
    std::ofstream fout("tmp.txt");
    fout << std::fixed << std::setprecision(10);
    for (auto& p : v) fout << p << '\n';
    fout.close();
    return 0;
}
