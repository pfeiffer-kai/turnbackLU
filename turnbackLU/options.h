// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef _OPTIONS_TURNBACK_
#define _OPTIONS_TURNBACK_

#pragma once

// FIXME: everything constant?

// this file contains the option settings for 
// turnback

namespace turnback
{
    struct options
    {
        // 0: only do initial two factorizations (whole and first submatrix)
        // 1: new factorization at every turnback step
        // the lower (up to 0) the less whole factorizations are done
        double olThres = 0.85; 
        double leftColDiscount = 1.; // < 1: consider the columns on the left that need to be removed and can be discounted
        int maxmn = 1e5; // for vector initialization
        int maxn = 1e5; // for vector initialization
        double linearDependency = 1e-12; // linear dependency in rank revealing qr decomposition
        double condThres = 1e10; // threshold for pseudo conditioning of NS, set large to disable, max < 1e153, due to rankCheck in same function
        bool colUpdates = false; // FIXME: true: NI

        double nserrThres = 2e-10;
        double nserrThresFatal = 1e-1;

        // factorization method
        // 0: LUSOL
        // 1: QR
        int facM = 0;

        // if the blocks have known structure, the turnback algorithm can be accelerated
        bool pattern = false;
        // XEID
        // 0: the pivot is picked by hand (good accuracy ~1e-11; fast; provably full rank (unless there is some implementation mistake))
        // 1: the pivot is picked by LUSOL, by avoiding LUSOL pivots within the given range (can have higher accuracy for well posed problems ~1e-12; not necessarily numerically stable for badly posed problems)
        // 2: the pivot is picked by LUSOL, by conducting a RR of the range and choosing the nu weakest elements (not necessarily numerically stable)
        // 3: the pivot is picked by LUSOL, by having no pivot requirements, all z's are computed (higher accuracy with higher computational effort ~1e-13; this basis does not necessarily have full rank)
        // 4: the pivot is picked by LUSOL, by having no pivot requirements, z's are computed one by one and accepted if error below threshold (higher accuracy with higher computational effort; this basis does not necessarily have full rank)
        int pivotM = 0;

        int verbose = NONE; // NONE, BASIC, CONV, VAR, MAT
        int verboseTime = NONE;
        int tid = 0;

    };
} // namespace turnback
#endif // _OPTIONS_TURNBACK_
