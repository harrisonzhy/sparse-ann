#include "include/timer.hpp"
#include <vector>
#include <unordered_set>
#include <iostream>

int main() {
    int n = 1000000;
    std::vector<int> generated(n);
    for (auto i = 0; i < n; ++i) {
        generated[i] = int{(rand() / (double)RAND_MAX) * n};
    }
    Timer t;
    t.Start();
    for (auto i = 0; i < 1; ++i) {
        std::unordered_set<int> set;
        set.insert(std::begin(generated), std::end(generated));
    }
    t.Stop();
    std::cout << t.Seconds() / 1. << std::endl;
}
