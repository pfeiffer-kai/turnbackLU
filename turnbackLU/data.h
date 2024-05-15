// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef _DATA_TURNBACK_
#define _DATA_TURNBACK_

#pragma once
#include "typedefs.h"
#include "options.h"
#include <eigenlusol/eigenlusol.h>
#include <eigenlusol/options.h>

// FIXME: can't pass references as rvalues, makes up for some ugly code (see winact() ...)
// for some reason, passing as rvalue f(vec&& a) doesn't work and a is not changed (only locally in the function f

namespace turnback
{
    struct data
    {
        // is this fast?

        // contains all information about the HLSP
        public:
            data(shared_ptr<cpplusol::eigenlusol>& lu_, int nrThreads, options& _opt);

            void reset(int rows, int cols);

            veci b; // subset indices; this is not strict but a guidance to enhance subset search efficiency (column replacements, refactorizations, ...)
            vector<vector<int> > JPerThread;
            vector<int> bsort;
            veci pivotCols;
            veci pivotCols_;
            veci subsetNr;
            veci subsetCard;
            vector<vector<int> > subsetsPerThread;
            vector<vector<int> > jsPerSubset;
            veci subsetActivity;
            mati subsetRg;
            vector<int> pivotcolssort;
            vec rowValsCtr;
            unique_ptr<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> > _Q; // Q from initial LU decomposition
            vector<vec> z;
            vector<vec> u;
            shared_ptr<mat> _Z;
            vector<mat> tmpU2Z;
            turnback::tripletHandler thz;
            vector<turnback::tripletHandler> th;
            vector<vec> tmp;
            vector<bool> xeidFacFresh;
            mat A_partial;
            mat AZ;
            vec AZerr;
            vector<int> AZerrsort;

            shared_ptr<cpplusol::eigenlusol> lu0; // the big one to compute the whole factorization of the (possibly large) incoming matrix
            vector<shared_ptr<cpplusol::eigenlusol> > lu_omp; // smaller ones for parallel computation

            shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > > qr;
            vector<shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > > > qr_omp;

            vector<veci> elimCols_omp; // columns of A that have been used in the LU decomposition
            veci chosenCols; // pivot columns that already have been used
            vector<veci> chosenColsP; // permuted pivot columns that already have been used
            vector<veci> chosenColsPinv; // permuted pivot columns that already have been used

            options opt;


#if TIMEMEASUREMENTS
            vector<turnback::time> times;
#endif

            void print()
            {
            };
    };
}

#endif // _DATA_TURNBACK_
