#include <iostream>
#include <vector>

#include "planning/env.h"

int main() {
  LinearEnv<1, double, false, 3, 3> env{
      Fastor::Tensor<double, 1, 2, 3>(
          std::vector<double>{3, 0.1, 0.3, 2, 0.1, 0.3}),
      Fastor::Tensor<double, 1, 2>(std::vector<double>{0, 0})};
  env.train(0.9, 0.01, 0.5, 1, 20000000);
  std::cout << env.q();
}
