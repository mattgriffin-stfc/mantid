// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <sstream>
#include <iostream>

#include <curand.h>

/**
 * @brief curandGetErrorString  return an error message for a given CURAND
 *                              status.
 *
 * Helper function that mirrors cudaGetErrorString, translates a curandStatus_t
 * enum value into a string error message.
 *
 * @param error  the error code reported
 * @return string representation of the status code.
 */
static const char * curandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "No errors";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "cuRAND header file and linked library version do not match";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "cuRAND generator not initialized";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed";

    case CURAND_STATUS_TYPE_ERROR:
      return "cuRAND generator incorrect type";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "Argument out of range";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "Requested length not a multple of dimension";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "GPU does not have double precision required by MRG32k3a";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "cuRAND kernel launch failure";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "Preexisting failure on cuRAND library entry";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "Initialization of CUDA failed";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "Architecture mismatch, GPU does not support requested feature";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "cuRAND internal library error";
  }

  return "<unknown>";
}

/**
 * Macro for wrapping CURAND calls.
 * Ex: CURAND_GUARD(curandMakeMTGP32Constants(paramSet, params));
 */
#define CURAND_GUARD(ans) { curandAssert((ans), __FILE__, __LINE__); }

/**
 * @brief curandAssert  assert a CURAND runtime invocation result
 *
 * Handles an unknown curandStatus_t in the CURAND runtime environment.
 *
 * @param code  the curandStatus_t code that occurred
 * @param file  file in which the error occurred
 * @param line  line number at which the error occurred
 * @param die   whether to throw an exception on error (default: true).
 */
inline void curandAssert(curandStatus_t code, const char* file, int line,
                         bool die = true) {
  if (code != CURAND_STATUS_SUCCESS) {
      const char* errorMessage = curandGetErrorString(code);

      std::ostringstream error;
      error << "CURAND Error: " << errorMessage
            << " FILE: " << file
            << " LINE: " << line
            << std::endl;

      if (die) {
          throw std::runtime_error(error.str());
      } else {
          std::cerr << error.str();
      }
  }
}
