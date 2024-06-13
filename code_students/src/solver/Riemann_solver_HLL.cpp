#include "setup/fluid.hpp"
#include "setup/physics.hpp"
#include "solver/Riemann_solvers.hpp"
#include "util/utility_functions.hpp"

HLL_solver::HLL_solver(std::size_t num_fields_) : Riemann_solver(num_fields_) { epsilon = 1.e-50; }

void HLL_solver::get_num_flux(fluid_cell &fluid_left_cell, fluid_cell &fluid_right_cell, const std::vector<double> &phys_flux_left_cell,
                              const std::vector<double> &phys_flux_right_cell, std::vector<double> &num_flux, double v_char_slowest, double v_char_fastest) {

	// Apply HLL fluxes
	if (v_char_slowest > 0.0) {
		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			num_flux[i_field] = phys_flux_left_cell[i_field];
		}
	} else if (v_char_fastest < 0.0) {
		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			num_flux[i_field] = phys_flux_right_cell[i_field];
		}
	} else { // F_hll
		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			double S_r = v_char_fastest;
			double S_l = v_char_slowest;
			double F_r = phys_flux_right_cell[i_field];
			double F_l = phys_flux_left_cell[i_field];
			double q_r = fluid_right_cell.fluid_data[i_field];
			double q_l = fluid_left_cell.fluid_data[i_field];
			num_flux[i_field] = (S_r * F_l - S_l * F_r + S_l * S_r * (q_r - q_l)) / (S_r - S_l + epsilon);
		}
	}
}
