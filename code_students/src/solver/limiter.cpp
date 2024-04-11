#include "solver/limiter.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double first, double second, double third) {
	if ((std::signbit(first) == std::signbit(second)) && (std::signbit(second) == std::signbit(third))) {
		if (std::signbit(first)) {
			return std::max(first, std::max(second, third));
		} else {
			return std::min(first, std::min(second, third));
		}
	} else {
		return 0;
	}
}