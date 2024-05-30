#include "mpi.h"
#include <bits/chrono.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <tuple>
#include <vector>

// Include that allows to print result as an image
// Also, ignore some warnings that pop up when compiling this as C++ mode
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#define ASSIGNMENT_TAG 1
#define DATA_TAG 2

constexpr int default_size_x = 1344;
constexpr int default_size_y = 768;

// RGB image will hold 3 color channels
constexpr int num_channels = 3;
// max iterations cutoff
constexpr int max_iterations = 10000;

size_t index(int y, int x, int size_x, int channel) { return y * size_x * num_channels + x * num_channels + channel; }

size_t lindex(int x, int channel) { return x * num_channels + channel; }

using Image = std::vector<uint8_t>;

auto HSVToRGB(double H, const double S, double V) {
	if (H >= 1.0) {
		V = 0.0;
		H = 0.0;
	}

	const double step = 1.0 / 6.0;
	const double vh = H / step;

	const int i = (int)floor(vh);

	const double f = vh - i;
	const double p = V * (1.0 - S);
	const double q = V * (1.0 - (S * f));
	const double t = V * (1.0 - (S * (1.0 - f)));
	double R = 0.0;
	double G = 0.0;
	double B = 0.0;

	// clang-format off
	switch (i) {
	case 0: { R = V; G = t; B = p; break; }
	case 1: { R = q; G = V; B = p; break; }
	case 2: { R = p; G = V; B = t; break; }
	case 3: { R = p; G = q; B = V; break; }
	case 4: { R = t; G = p; B = V; break; }
	case 5: { R = V; G = p; B = q; break; }
	}
	// clang-format on

	return std::make_tuple(R, G, B);
}

void calcMandelbrot(Image &line, int size_x, int size_y, int pixelY) {
	const float left = -2.5, right = 1;
	const float bottom = -1, top = 1;

	// scale y pixel into mandelbrot coordinate system
	const float cy = (pixelY / (float)size_y) * (top - bottom) + bottom;
	for (int pixel_x = 0; pixel_x < size_x; pixel_x++) {
		// scale x pixel into mandelbrot coordinate system
		const float cx = (pixel_x / (float)size_x) * (right - left) + left;
		float x = 0;
		float y = 0;
		int num_iterations = 0;

		// Check if the distance from the origin becomes
		// greater than 2 within the max number of iterations.
		while ((x * x + y * y <= 2 * 2) && (num_iterations < max_iterations)) {
			float x_tmp = x * x - y * y + cx;
			y = 2 * x * y + cy;
			x = x_tmp;
			num_iterations += 1;
		}

		// Normalize iteration and write it to pixel position
		double value = fabs((num_iterations / (float)max_iterations)) * 200;

		auto [red, green, blue] = HSVToRGB(value, 1.0, 1.0);

		line[lindex(pixel_x, 0)] = (uint8_t)(red * UINT8_MAX);
		line[lindex(pixel_x, 1)] = (uint8_t)(green * UINT8_MAX);
		line[lindex(pixel_x, 2)] = (uint8_t)(blue * UINT8_MAX);
	}
}

void sendStop(int rank) {
	int stop = -1;
	MPI_Send(&stop, 1, MPI_INT, rank, ASSIGNMENT_TAG, MPI_COMM_WORLD);
}

void sendAssignment(int &sendY, std::vector<int> &lastAssignments, int rank) {
	MPI_Send(&sendY, 1, MPI_INT, rank, ASSIGNMENT_TAG, MPI_COMM_WORLD);
	lastAssignments[rank] = sendY;
	sendY++;
}

int recvData(int &recvY, std::vector<int> &lastAssignments, std::vector<uint8_t> &image, int size_x) {
	MPI_Status status;
	MPI_Probe(MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, &status);
	int rank = status.MPI_SOURCE;
	int y = lastAssignments[rank];
	MPI_Recv(&image[index(y, 0, size_x, 0)], size_x * num_channels, MPI_UINT8_T, rank, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	recvY++;
	return rank;
}

int recvAssignment() {
	int y;
	MPI_Recv(&y, 1, MPI_INT, 0, ASSIGNMENT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	return y;
}

void sendData(std::vector<uint8_t> &line, int size_x) { MPI_Send(&line[0], size_x * num_channels, MPI_UINT8_T, 0, DATA_TAG, MPI_COMM_WORLD); }

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &ranks);

	int size_x = default_size_x;
	int size_y = default_size_y;

	if (argc == 3) {
		size_x = atoi(argv[1]);
		size_y = atoi(argv[2]);
		std::cout << "Using size " << size_x << "x" << size_y << std::endl;
	} else {
		std::cout << "No arguments given, using default size " << size_x << "x" << size_y << std::endl;
	}

	if (rank == 0) {
		auto image = Image(num_channels * size_x * size_y);
		auto lastAssignments = std::vector<int>(ranks);
		int sendY = 0;
		int recvY = 0;

		auto time_start = std::chrono::high_resolution_clock::now();

		for (int r = 1; r < ranks; r++) {
			sendAssignment(sendY, lastAssignments, r);
		}

		while (sendY < size_y) {
			int recvRank = recvData(recvY, lastAssignments, image, size_x);

			sendAssignment(sendY, lastAssignments, recvRank);
		}

		for (int r = 1; r < ranks; r++) {
			sendStop(r);
		}

		while (recvY < size_y) {
			recvData(recvY, lastAssignments, image, size_x);
		}

		auto time_end = std::chrono::high_resolution_clock::now();
		auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

		std::cout << "Mandelbrot set calculation for " << size_x << "x" << size_y << " took: " << time_elapsed << " ms." << std::endl;

		constexpr int stride_bytes = 0;
		stbi_write_png("mandelbrot_mpi.png", size_x, size_y, num_channels, &image[0], stride_bytes);
	} else {
		auto line = Image(num_channels * size_x);
		while (true) {
			int pixelY = recvAssignment();
			if (pixelY < 0) { // -1 = stop
				break;
			}
			calcMandelbrot(line, size_x, size_y, pixelY);
			sendData(line, size_x);
		}
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
