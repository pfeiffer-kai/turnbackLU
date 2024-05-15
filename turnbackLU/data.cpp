// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "data.h"

namespace turnback
{
    data::data(shared_ptr<cpplusol::eigenlusol>& lu_, int nrThreads, options& _opt) :
        lu0(lu_),
        opt(_opt)
    {
        lu0->reset();
        lu0->getOpt().pivot = cpplusol::TCP; // TCPNM
        lu0->getOpt().keepLU = 0; // L and U are only needed in the update case (?)

        th.resize(nrThreads); for (tripletHandler& thh : th) { thh = tripletHandler(opt.maxmn); }
        thz = turnback::tripletHandler(opt.maxmn);

        b = veci::Zero(2*opt.maxn);
        bsort.resize(2*opt.maxn); std::fill(bsort.begin(), bsort.end(), 0);
        pivotcolssort.resize(2*opt.maxn); std::fill(pivotcolssort.begin(), pivotcolssort.end(), 0);
        rowValsCtr = vec::Zero(2*opt.maxn);
        pivotCols = veci(2*opt.maxn); pivotCols.setConstant(0);
        pivotCols_ = veci(2*opt.maxn); pivotCols_.setConstant(0);
        subsetNr = veci(2*opt.maxn); subsetNr.setConstant(-1);
        subsetCard = veci(2*opt.maxn); subsetCard.setZero();
        subsetRg = mati(2,2*opt.maxn); subsetRg.setConstant(-1);
        subsetsPerThread.resize(nrThreads);
        jsPerSubset.resize(2*opt.maxn);
        subsetActivity.resize(2*opt.maxn); subsetActivity.setZero();
        xeidFacFresh.resize(2*opt.maxn); std::fill(xeidFacFresh.begin(), xeidFacFresh.end(), false);

        _Z = make_unique<mat>(0,0);

        u.resize(nrThreads);
        z.resize(nrThreads); for (vec& zz : z) zz = vec::Zero(2*opt.maxn);
        JPerThread.resize(nrThreads);
        if (opt.facM == 0)
        {
            lu_omp.resize(nrThreads);
            if (nrThreads == 1)
            {
                lu_omp[0] = lu0;
            }
            else
            {
                for (shared_ptr<cpplusol::eigenlusol>& lu_omp_ : lu_omp) lu_omp_ = make_shared<cpplusol::eigenlusol>();
            }
        }
        else
        {
            A_partial = mat(opt.maxmn, opt.maxmn);
            qr = make_shared<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >();
            qr_omp.resize(nrThreads);
            for (shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr_omp_ : qr_omp) qr_omp_ = make_shared<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >();
        }
        elimCols_omp.resize(nrThreads); for (veci& ec : elimCols_omp) { ec = veci(2*opt.maxn); ec.setZero(); }
        chosenCols = veci(2*opt.maxn); chosenCols.setZero();
        chosenColsP.resize(nrThreads); for (veci& cc : chosenColsP) { cc = veci(2*opt.maxn); cc.setZero(); }
        chosenColsPinv.resize(nrThreads); for (veci& cc : chosenColsPinv) { cc = veci(2*opt.maxn); cc.setZero(); }

        if (opt.pivotM == 0 || opt.pivotM == 3)
        {
            AZ = mat(opt.maxmn, opt.maxmn);
            AZerr = vec(opt.maxmn); AZerr.setZero();
            AZerrsort.resize(opt.maxmn); std::fill(AZerrsort.begin(), AZerrsort.end(), 0);
            tmpU2Z.resize(nrThreads); for (mat& m : tmpU2Z) { m = mat(opt.maxmn,opt.maxmn); }
        }

        tmp.resize(nrThreads); std::fill(tmp.begin(), tmp.end(), vec::Zero(opt.maxmn));

#if SAFEGUARD
        qr = make_shared<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >();
#endif // SAFEGUARD

    }

    void data::reset(int rows, int cols)
    {
        lu0->reset();
        thz.reset();
        for (int tid = 0; tid < JPerThread.size(); tid++)
        {
            elimCols_omp[tid].head(cols).setZero();
            chosenCols.head(cols).setZero();
        }

        subsetNr.head(cols).setConstant(-1);
        subsetCard.head(cols).setZero();
        subsetRg.leftCols(cols).setConstant(-1);
        subsetActivity.head(cols).setZero();
        for (vector<int>& jps : jsPerSubset)
            jps.clear();
    }
} // namespace nipmhlsp
