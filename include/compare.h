#ifndef COMPARE_H
#define COMPARE_H

#include "matrix.h"
#include "query.h"

// Confronta i risultati calcolati con quelli forniti
void compare_results(const MatrixF32 *queries,
                     const Neighbor *computed,
                     const MatrixI32 *ref_ids,
                     const MatrixF32 *ref_dist,
                     int k);

#endif
