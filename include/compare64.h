#ifndef COMPARE64_H
#define COMPARE64_H

#include "matrix.h"
#include "query64.h"

// Confronta risultati 64-bit con i file ufficiali
void compare_results_f64(const MatrixF64 *queries,
                         const Neighbor64 *computed,
                         const MatrixI32 *ref_ids,
                         const MatrixF64 *ref_dist,
                         int k);

#endif
