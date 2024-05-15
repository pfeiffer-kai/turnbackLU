// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef _TURNBACKLU_
#define _TURNBACKLU_

#pragma once

#include "typedefs.h"
#include "options.h"
#include "data.h"
#include <eigenlusol/eigenlusol.h>

// run time is sensitive to zero rows since lusol does not consider sparsity during column updates
// data should be filtered before such that there are no zero rows

namespace turnback
{
    template <typename T> void print(string name, T* a, int len);

    struct turnbackIndices // does this make sense? are there copies?
    {
        int j = 0;
        int piv = 0; // pivot column
        int pivU = 0; // pivot column in local LU column system
        int jrep = 0; // current replacement column
        int bprev = 0; // subset start previous
        int bcur = 0; // subset start current
        int bnext = 0; // subset start next
        int subsetNr = -1;
        int leftI = -1;
        int rightI = -1;
        int leftsubsetnr = -1;
        int rightsubsetnr = -1;
        double ol = 0;
        int mar = 0; // maximally achievable rank
        int colRg = 0;
        bool fatalError = false;
        // the first turnback iteration needs to be recomputing, otherwise dense (since we dont remove columns from the right)
        bool anew = true;
        int nrCols = 0; // number of columns in the current decomposition
        string subsetDir = "left"; // "right"

        options opt;

        void print()
        {
            cout << "piv " << piv << " subset " << subsetNr << " bcur " <<  bcur << " bprev " << bprev << " bnext " << bnext << " jrep " << jrep << " mar " << mar << " col overlap " << ol << " <  " << opt.olThres << "? anew " << anew << " colRg " << colRg << endl;
        }
    };

    struct xeidIndices
    {
        // in case that the structure corresponds to dynamics with variables [tau f q dq] := [u x] with u := [tau f] and x := [q dq] and nf = dim(f) 
        xeidIndices(const std::tuple<int,int,int,int> pattern_)
        {
            pattern = pattern_; 
            nx = std::get<0>(pattern);
            nt = std::get<1>(pattern);
            nf = std::get<2>(pattern);
            T = std::get<3>(pattern);
            nua = nx/2 - nt; // underactuation
            nu = nt + nf;
        }

        bool checkPattern()
        {
            // check pattern
            if (nx <= 0)
            {
                cout << "turnbackLU: WARNING: nx " << nx << " value (<= 0) does not indicate that there is a valid pattern. Proceeding without pattern." << endl;
                return false;
            }
            if (nx % 2 != 0)
            {
                cout << "turnbackLU: WARNING: nx " << nx << " value (uneven) does not indicate that there is a valid pattern. Proceeding without pattern." << endl;
                return false;
            }
            if (T <= 0)
            {
                cout << "turnbackLU: WARNING: T " << T << " value (<= 0) does not indicate that there is a valid pattern. Proceeding without pattern." << endl;
                return false;
            }
            if (nua >= (int)(nx/2))
            {
                cout << "turnbackLU: WARNING: nua " << nua << " is larger equal than the threshold nq " << nx/2 << ". Proceeding without pattern." << endl;
                return false;
            }
            return true;
        }

        std::tuple<int,int,int,int> pattern = {0,0,0,1};
        int nx, nu, nt, nf, nua, T=1;
    };

    class turnbackLU
    {
        // computes banded NS basis of A by using lusol LU
        public:
            turnbackLU(shared_ptr<cpplusol::eigenlusol> lu_ = NULL, options opt_ = options());

            // computes the explicit NS of the matrix A
            void computeNS(const mat& A, shared_ptr<mat> Zio = NULL, const std::tuple<int,int,int,int> pattern_ = {0,0,0,1}, const int nrEcs = 0, const bool eqProj=false);

            int& rank() { return rank0; };

            inline const mat& Z() const { if (!ws->_Z) { cout << "turnback::Z(): not computed" << endl; throw; } return *ws->_Z; }

            inline const Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>& Q() const
            { 
                if (!ws->_Q)
                { 
                    cout << "turnback::Q(): not computed" << endl;
                    throw;
                }
                else
                    return *ws->_Q;
            }

            // for info
            int nrAddCols = 0, nrRemCols = 0, nrFac = 0, nrFac_bin = 0;

            void printInfo(const mat& A = mat(0,0));
            void printTime();

            shared_ptr<cpplusol::eigenlusol> get_lu() { return ws->lu_omp[0]; };

            bool bInitialized = false;

        private:
            int cols, rows;
            int rank0, rankZ;
            int nrSubsets = 0;
            int dimNS = 0;

            bool useXEID = false;
            std::shared_ptr<xeidIndices> xeidIdx = NULL;
            int nodes = 1;

            void reset(const mat& A, const std::tuple<int,int,int,int> pattern_);

            void getOB(const mat& A, const int nrEcs);
            // finds the pivot columns and the subset indicators
            // assumes the variable structure tau | f | q | dq
            // (XEID: explicit euler inverse dynamics, where the dynamics are unit on tau)
            // pivot elements are put on f and dq
            // the linearly independent subsets are on tau and q
            // note that if (dim(tau) != dim(dq) (underactuation), the subsets are also sourced from dq (and f)
            // the dynamics should be equalized, such that the subset dynamic matrices are leading in magnitude and will be chosen first by eigenlusol::TCPNM (threshold complete pivoting with no Markovitz)
            void findBX(const mat& A);
            // from scratch using rank revealing LU decomposition
            void findB();
            int count2b();
            void fixb();
            void initTBIndices(unique_ptr<turnbackIndices>& tbx, int tid);
            // inline since we do this (rankZ) times
            // FIXME: this is quite slow and not well understood
            inline void compOverlap(int tid, unique_ptr<turnbackIndices>& tbx, const mat& A);
            inline void getIdx(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const mat& A);
            inline bool getIdxXEID(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const mat& A);
            inline bool getIdxXEID(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, unique_ptr<turnbackIndices>& tbx, const mat& A);
            inline int getIdxFromPiv(shared_ptr<cpplusol::eigenlusol>& lu, const unique_ptr<turnbackIndices>& tbx);
            inline void getMAR(int tid, const mat& A, unique_ptr<turnbackIndices>& tbx); // calculate maximum achievable rank
            inline void initialFac(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx);
            inline void initialFac_rg(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx); // range
            inline void initialFac_bin(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A); // based on elimcol
            inline void initialFac_XEID(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx); // based on custom made elimCol
            inline void initialFac_XEID(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const mat& A, unique_ptr<turnbackIndices>& tbx); // based on custom made elimCol
            inline void stage1(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx);
            inline int addCol(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, const int colidx, const int mode2=1);
            inline int addCol(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const mat& A, const int colidx, const int mode2=1);
            inline bool addSubsetsUntilRup(int tid, const mat& A, unique_ptr<turnbackIndices>& tbx);
            inline int addSubset(int tid, unique_ptr<turnbackIndices>& tbx, string dir="right");
            inline void remCol(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const int colidx, const mat& A);
            inline bool remCols(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const string dir, const mat& A);
            inline int computez(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const int j, unique_ptr<turnbackIndices>& tbx, const mat& A, int testAzCtr=-1);
            inline int computez(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const int j, unique_ptr<turnbackIndices>& tbx, const mat& A, int testAzCtr=-1);

            int nrThreads = 1;
            int nrThreadsUsed = 1;

            // order
            // 0: b indicates first nnz in each column of Z, a right looking turnback algorithm is conducted
            // 1: b indicates last nnz in each column of Z, a left looking turnback algorithm is conducted
            int order = 0; 
            int curNrCols = 0;

            shared_ptr<data> ws;

            turnback::options opt;
            int l;

            double nserrThres = 1e-12;

            void printOB(const mat& A);
            void printIter(int tid, int j, const unique_ptr<turnbackIndices>& tbx);

#if SAFEGUARD
            void computeLUNS(const mat& A);
            matd _A;
            vector<vec> zcollected;
#endif

    };

    inline void turnbackLU::compOverlap(int tid, unique_ptr<turnbackIndices>& tbx, const mat& A)
    {
        if (xeidIdx)
        {
            tbx->mar = 3*xeidIdx->nx + 2*xeidIdx->nf; // subset range, more stable 
        }
        else
        {
            getMAR(tid, A, tbx);
            tbx->mar = max(max(tbx->mar, abs(tbx->piv - tbx->bcur) + 1), abs(tbx->bnext - tbx->bcur));
        }
        if (order == 0) tbx->mar = max(tbx->jrep - tbx->bcur, tbx->mar); // keep columns that have been used on the right already anyways
        else tbx->mar = max(tbx->bcur + tbx->jrep, tbx->mar); // keep columns that have been used on the right already anyways
        // if mar is shorter or longer
        // considers the columns on the left that need to be removed and can be discounted
        int colOl = tbx->mar;
        if (tbx->mar > abs(tbx->jrep - tbx->bcur))
            colOl = abs(tbx->jrep - tbx->bcur);
        else if (tbx->mar < abs(tbx->jrep - tbx->bcur))
            colOl = tbx->mar;
        // colOl = abs(tbx->jrep - tbx->bcur); // this discounts columns on the right that are already there and have destroyed the sparsity anyways
        tbx->ol = (double)(colOl) / ((double)(abs(tbx->jrep - tbx->bcur) + opt.leftColDiscount * (double)(abs(tbx->bcur - tbx->bprev))));
        if (tbx->ol < opt.olThres) tbx->anew = true;
        if (order == 0)
        {
            tbx->colRg = min(cols-tbx->bcur,max(1,tbx->mar));
        }
        else if (order == 1) tbx->colRg = min(tbx->bcur+1,max(1,tbx->mar));
    }

    inline void turnbackLU::getMAR(int tid, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
        // mar: maximally achievable rank identified by mock RR decomp
        // pivot column needs to be within the decomposed block
        // take at least the block reaching to brext but excluding it
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        // find the maximally achievable rank starting from tbx->bcur by using a mock RR decomp
        ws->tmp[tid].head(rows).setConstant(-1);
        tbx->mar = 0;
        if (order == 0)
        {
            for (int j = tbx->bcur; j < cols; j++)
            {
                bool found = false;
                for (mat::InnerIterator jt(A,j); jt; ++jt)
                {
                    if (ws->tmp[tid][jt.row()] == -1) { ws->tmp[tid][jt.row()] = j; found = true; break; }
                }
                if (!found) { tbx->mar = j - tbx->bcur; return; }
            }
            tbx->mar = cols - tbx->bcur;
        }
        else if (order == 1)
        {
            for (int j = tbx->bcur; j >= 0; j--)
            {
                bool found = false;
                for (mat::InnerIterator jt(A,j); jt; ++jt)
                {
                    if (ws->tmp[tid][jt.row()] == -1) { ws->tmp[tid][jt.row()] = j; found = true; break; }
                }
                if (!found) { tbx->mar = tbx->bcur - j; return; }
            }
            tbx->mar = tbx->bcur;
        }
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::getMAR"));
        if (opt.verbose >= CONV) tm1.stop("getMAR");
#endif
    }

    inline void turnbackLU::initialFac(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
#if TIMEMEASUREMENTS
        turnback::timer tm3;
#endif
        tbx->jrep = 0;
        ws->elimCols_omp[tid].head(cols).setConstant(1);
        lu->reset();
#if TIMEMEASUREMENTS
        tm3.stop(); times.push_back(turnback::time(tm3.time, "turnbackLU::initialFac: start"));
#endif
#if TIMEMEASUREMENTS
        turnback::timer tm2;
#endif
        turnback::timer tm2;
        lu->factorize(A, false);
        if (opt.verbose >= CONV && tid == opt.tid) tm2.stop("[thread-" + std::to_string(tid) + "] initialFac with " + std::to_string(lu->U().nonZeros()) + " nnz's in U");
        ws->_Q = make_unique<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> >(cols);
        *ws->_Q = lu->Q();
        ws->elimCols_omp[tid].head(cols).setZero();
        nrAddCols += cols;
        curNrCols = cols;
#if TIMEMEASUREMENTS
        tm2.stop(); times.push_back(turnback::time(tm2.time, "turnbackLU::initialFac: factorize"));
        if (opt.verbose >= CONV) tm2.stop("initialFac: factorize");
#endif
#if SAFEGUARD
        _A = (matd)A;
#endif
        if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxxxxxxxx turnbackLU::initialFac of " << rows << " X " << cols << " with tbx->bcur " << tbx->bcur << " and range " << tbx->colRg << ": rank " << lu->rank() << ", NS dim " << cols-lu->rank() << endl;
        nrFac++;
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::initialFac"));
        if (opt.verbose >= CONV) tm1.stop("initialFac");
#endif
    }

    inline void turnbackLU::initialFac_rg(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
#if TIMEMEASUREMENTS
        turnback::timer tm3;
#endif
        tbx->jrep = 0;
        ws->elimCols_omp[tid].head(cols).setConstant(1);
        lu->reset();
#if TIMEMEASUREMENTS
        tm3.stop(); times.push_back(turnback::time(tm3.time, "turnbackLU::initialFac_rg: start"));
#endif
#if TIMEMEASUREMENTS
        turnback::timer tm2;
#endif
        if (opt.verbose >= MAT)
        {
            turnback::tripletHandler th(1e5);

            for (int c = tbx->bcur; c < tbx->bcur+tbx->colRg; c++)
            {
                for (mat::InnerIterator it(A,c); it; ++it)
                {
                    th.setTriplet(it.row(),c,it.value());
                }
            }
            mat Ashow = mat(A.rows(), A.cols());
            th.setFromTriplets(Ashow);
            cout << "Ashow (" << Ashow.rows() << " " << Ashow.cols() << "\n" << (matd)Ashow << endl;
        }
        nrAddCols += tbx->colRg;
        curNrCols = tbx->colRg;
        turnback::timer tm2;
        if (order == 0) 
        {
            lu->factorize_partial(A, tbx->bcur, tbx->colRg, false);
            if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxxxxx turnbackLU::initialFac_rg: RECALC fac from col " << tbx->bcur << " to col " << tbx->bcur + tbx->colRg-1 << " with rank " << lu->rank() << endl;
            tbx->jrep = tbx->bcur + tbx->colRg;
            ws->elimCols_omp[tid].head(cols).setConstant(1.);
            ws->elimCols_omp[tid].segment(tbx->bcur, tbx->colRg).setZero();
#if SAFEGUARD
            _A.setZero();
            _A.block(0, tbx->bcur, rows, tbx->colRg) = (matd)A.block(0, tbx->bcur, rows, tbx->colRg);
#endif
        }
        else if (order == 1) 
        { 
            if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxxxx turnbackLU::initialFac_rg: RECALC fac from col " << tbx->bcur-tbx->colRg+1 << " to col " << tbx->bcur << endl;
            lu->factorize_partial(A, tbx->bcur-tbx->colRg+1, tbx->colRg, false);
            tbx->jrep = tbx->bcur - tbx->colRg;
            ws->elimCols_omp[tid].head(cols).setConstant(1.);
            ws->elimCols_omp[tid].segment(tbx->bcur-tbx->colRg+1, tbx->colRg).setZero();
#if SAFEGUARD
            _A.setZero();
            _A.block(0, tbx->bcur-tbx->colRg+1, rows, tbx->colRg) = (matd)A.block(0, tbx->bcur-tbx->colRg+1, rows, tbx->colRg);
#endif
        }
        if (opt.verbose >= BASIC && tid == opt.tid) tm2.stop("[thread-" + std::to_string(tid) + "] initialFac_rg with " + std::to_string(lu->U().nonZeros()) + " nnz's in U");
#if TIMEMEASUREMENTS
        tm2.stop(); times.push_back(turnback::time(tm2.time, "turnbackLU::initialFac_rg: factorize_partial"));
        if (opt.verbose >= CONV && tid == opt.tid) tm2.stop("[thread-" << tid << "] initialFac_rg: factorize_partial");
#endif
        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::initialFac_rg of " << rows << " X " << cols << " with tbx->bcur " << tbx->bcur << " and range " << tbx->colRg << ": rank " << lu->rank() << ", NS dim " << cols-lu->rank() << endl;
        if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::initialFac_rg U nnz " << lu->U().nonZeros() << endl;
        nrFac++;
#if TIMEMEASUREMENTS
        tm1.stop("[thread-" + std::to_string(tid) + "] initialFac_rg");
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::initialFac_rg"));
        if (opt.verbose >= CONV) tm1.stop("initialFac_rg");
#endif
    }

    inline void turnbackLU::initialFac_bin(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        // refactorize columns in ws->elimCols_omp[tid]
        // tbx->jrep etc is not touched
#if SAFEGUARD
        // compare with qr
        int nrCols_ = 0;
        ws->th.reset();
        for (int c = 0; c < A.cols(); c++)
        {
            if (ws->elimCols_omp[tid][c] == 0)
            {
                for (mat::InnerIterator it(A,c); it; ++it)
                {
                    th.setTriplet(it.row(),nrCols_,it.value());
                }
                nrCols_++;
            }
        }
        mat Acond = mat(A.rows(), nrCols_);
        th.setFromTriplets(Acond);
        ws->qr = make_shared<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >();
        ws->qr->compute(Acond);
        cout << "QR\n" << ws->qr->matrixR().diagonal().transpose() << endl;
#endif // SAFEGUARD

        lu->reset();
        turnback::timer tm2;
        int nrRows;
        int nrCols = lu->factorize_bin(A, ws->elimCols_omp[tid], nrRows, false);
        if (opt.verbose >= BASIC && tid == opt.tid) tm2.stop("[thread-" + std::to_string(tid) + "] initialFac_bin with " + std::to_string(lu->U().nonZeros()) + " nnz's in U");
        if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxx initialFac_bin " << nrFac_bin+1 << " with " << nrCols << " columns and new condition number " << lu->getUcond() << endl;
        if (opt.verbose >= VAR) cout << "initialFac_bin with elimcols " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
        curNrCols = nrCols;
        nrAddCols += nrCols;
        nrFac_bin++;
        if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::initialFac_bin U nnz " << lu->U().nonZeros() << endl;
#if TIMEMEASUREMENTS
        tm1.stop("[thread-" + std::to_string(tid) + "] initialFac_bin");
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::initialFac_bin"));
        if (opt.verbose >= CONV) tm1.stop("initialFac_bin");
#endif
    }

    inline void turnbackLU::initialFac_XEID(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        // refactorize columns according to XEID
        // tbx->jrep etc is not touched

        // assemble the elimination vector
        int addedCols = 0;
        ws->elimCols_omp[tid].head(cols).setOnes();
        if (!xeidIdx || opt.pivotM == 0 || opt.pivotM == 2)
        {
            for (int c = tbx->leftI; c <= tbx->rightI; c++)
            {
                if (ws->pivotCols_[c] == 0) ws->elimCols_omp[tid][c] = 0; // non-pivot columns
                else if (ws->subsetNr(c) > tbx->subsetNr) ws->elimCols_omp[tid][c] = 0; // pivot columns only on the right
            }
        }
        else
        {
            // this is not thread safe since chosencols is fluent
            for (int c = tbx->leftI; c <= tbx->rightI; c++)
            {
                if (c >= tbx->subsetNr * (xeidIdx->nx+xeidIdx->nu)) ws->elimCols_omp[tid][c] = 0; // only columns to the right
                else if (c < tbx->subsetNr * (xeidIdx->nx+xeidIdx->nu) && ws->chosenCols[c] == 0) ws->elimCols_omp[tid][c] = 0; // columns to the left that have not been chosen as pivots yet
            }
        }
        if (opt.verbose >= BASIC/* && tid == opt.tid*/) cout << "[thread-" << tid << "] initialFac_XEID with elimcols with " << addedCols << " entries\n" << ws->elimCols_omp[tid].head(cols).transpose() << endl;

        if (opt.verbose >= MAT)
        {
            turnback::tripletHandler th(1e5);

            for (int c = 0; c < A.cols(); c++)
            {
                if (ws->elimCols_omp[tid][c] == 0)
                {
                    for (mat::InnerIterator it(A,c); it; ++it)
                    {
                        th.setTriplet(it.row(),c,it.value());
                    }
                }
            }
            mat Ashow = mat(A.rows(), A.cols());
            th.setFromTriplets(Ashow);
            cout << "Ashow (" << Ashow.rows() << " " << Ashow.cols() << "\n" << (matd)Ashow << endl;
        }

        lu->reset();
        turnback::timer tm2;
        int nrRows;
        if (!xeidIdx || opt.pivotM == 0 || opt.pivotM == 2) tbx->nrCols = lu->factorize_bin(A, ws->elimCols_omp[tid], nrRows, false );
        else if (!xeidIdx && opt.pivotM == 1) tbx->nrCols = lu->factorize_bin(A, ws->elimCols_omp[tid], nrRows, false, tbx->subsetNr * (xeidIdx->nx+xeidIdx->nu)+1 /* lusol format */, (tbx->subsetNr+1) * (xeidIdx->nx+xeidIdx->nu), xeidIdx->nu);
        else if (!xeidIdx && opt.pivotM == 3) tbx->nrCols = lu->factorize_bin(A, ws->elimCols_omp[tid], nrRows, false, 0, cols, xeidIdx->nu);
        if (opt.verbose >= BASIC && tid == opt.tid) tm2.stop("[thread-" + std::to_string(tid) + "] initialFac_XEID of subset " +std::to_string(tbx->subsetNr) + " with " + std::to_string(lu->U().nonZeros()) + " nnz's in U");
        if (opt.verbose >= BASIC /*&& tid == opt.tid*/) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxx initialFac_XEID " << nrFac_bin+1 << " of subset " << tbx->subsetNr << " with " << tbx->nrCols << " columns and new condition number " << lu->getUcond() << " and rank " << lu->rank() << " nrRows " << nrRows << " nrCols " << tbx->rightI - tbx->leftI << " minus pivots " << tbx->nrCols << " in range [" << tbx->leftI << ", " << tbx->rightI << "] rank " << lu->rank() << " ns " << tbx->nrCols - lu->rank() << " dimNS " << dimNS << endl;
        curNrCols = tbx->nrCols;
        nrAddCols += tbx->nrCols;
        nrFac_bin++;
        tbx->jrep = tbx->rightI;
#if TIMEMEASUREMENTS
        tm1.stop("[thread-" + std::to_string(tid) + "] initialFac_XEID");
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::initialFac_XEID"));
        if (opt.verbose >= CONV) tm1.stop("initialFac_XEID");
#endif
        ws->xeidFacFresh[tid] = true;    
    }

    inline void turnbackLU::initialFac_XEID(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        // refactorize columns according to XEID
        // tbx->jrep etc is not touched

        // assemble the elimination vector
        ws->elimCols_omp[tid].head(cols).setOnes();
        for (int c = tbx->leftI; c <= tbx->rightI; c++)
        {
            if (ws->pivotCols_[c] == 0) ws->elimCols_omp[tid][c] = 0; // non-pivot columns
            else if (ws->subsetNr(c) != tbx->subsetNr) ws->elimCols_omp[tid][c] = 0;
        }

        int nrRows;
        int nrCols = 0;
        ws->th[tid].reset();
        for (int c = 0; c < A.cols(); c++)
        {
            if (ws->elimCols_omp[tid][c] == 0)
            {
                for (mat::InnerIterator it(A,c); it; ++it)
                {
                    ws->th[tid].setTriplet(it.row(),c,it.value());
                }
                nrCols++;
            }
        }
        ws->A_partial.conservativeResize(A.rows(), A.cols());
        ws->th[tid].setFromTriplets(ws->A_partial);
        turnback::timer tm2;
        qr->compute(ws->A_partial);
        if (opt.verbose >= BASIC && tid == opt.tid) tm2.stop("[thread-" + std::to_string(tid) + "] initialFac_XEID of subset " +std::to_string(tbx->subsetNr) + " with " + std::to_string(qr->matrixR().nonZeros()) + " nnz's in U");
        if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] xxxxxxxxxxxxxxx initialFac_XEID " << nrFac_bin+1 << " of subset " << tbx->subsetNr << " with " << nrCols << " columns and rank " << qr->rank() << " nrRows " << nrRows << " nrCols " << tbx->rightI - tbx->leftI << " minus pivots " << nrCols << " in range [" << tbx->leftI << ", " << tbx->rightI << "] rank " << qr->rank() << " ns " << nrCols - qr->rank() << endl;
        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] initialFac_XEID with elimcols " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
        curNrCols = nrCols;
        nrAddCols += nrCols;
        nrFac_bin++;
        tbx->jrep = tbx->rightI;
#if TIMEMEASUREMENTS
        tm1.stop("[thread-" + std::to_string(tid) + "] initialFac_XEID");
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::initialFac_XEID"));
        if (opt.verbose >= CONV) tm1.stop("initialFac_XEID");
#endif
        ws->xeidFacFresh[tid] = true;    
    }

    inline int turnbackLU::computez(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const int j, unique_ptr<turnbackIndices>& tbx, const mat& A, int testAzCtr)
    {
#if TIMEMEASUREMENTS
        turnback::timer tallz;
#endif

        if (!xeidIdx || opt.pivotM == 0 || opt.pivotM == 2)
        {
#if TIMEMEASUREMENTS
            turnback::timer t0;
#endif
            if (opt.colUpdates) getIdx(tid, lu, tbx, A);
            else
            {
                if (!getIdxXEID(tid, lu, tbx, A))
                    return 0;
            }
            ws->u[tid] = vec::Zero(lu->rank());
#if TIMEMEASUREMENTS
            t0.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::computezinitial"));
            if (opt.verbose >= CONV) t0.stop("computezinitial");
#endif
#if SAFEGUARD
            mat Upartprev;
#endif
            if (!opt.colUpdates)
            {
#if SAFEGUARD
                Upartprev = lu->U();
#endif
                if (ws->xeidFacFresh[tid])
                {
                    lu->constructu_partial(ws->u[tid]);
                    if (!lu->constructU_partial(true))
                    {
                        cout << "turnbackLU::computez: ERROR in constructU_partial, probably memory exceeded" << endl; 
                        throw;
                    }
                    ws->xeidFacFresh[tid] = false;
                }
                else
                {
                    lu->constructQ();
                    lu->constructu_partial(ws->u[tid]);
                }
#if SAFEGUARD
                if (Upartprev.cols() == lu->U().cols() && Upartprev.rows() == lu->U().rows())
                {
                    if ((Upartprev - lu->U()).norm() > 1e-12)
                    {
                        cout << "difference between Upartial now and prev" << endl;
                        cout << (matd)(Upartprev - lu->U()) << endl;
                        throw;
                    }
                }
#endif
            }
            else
            {
                if (!lu->constructU_partial(true, tbx->pivU, &(ws->u[tid]))) 
                {
                    cout << "turnbackLU::computez: error in constructUpartial, refactorize" << endl;
                    ws->chosenCols[lu->qq()[tbx->pivU]-1]--;
                    initialFac_bin(tid, lu, A);
                    getIdx(tid, lu, tbx, A);

                    lu->constructU_partial(true, tbx->pivU, &(ws->u[tid])); 
                }
            }
            if (opt.verbose >= VAR && tid == opt.tid)
            {
                cout << "[thread-" << tid << "] upart with " << ws->u[tid].rows() << " rows\n" << ws->u[tid].transpose() << endl;
                cout << "[thread-" << tid << "] Upart diag with " << lu->U().cols() << " cols\n" << lu->U().diagonal().transpose() << endl;
                if (opt.verbose >= MAT) cout << "[thread-" << tid << "] Upart with " << lu->U().cols() << " cols\n" << (matd)lu->U() << endl;
            }

#if TIMEMEASUREMENTS
            t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::cal Upartial"));
            if (opt.verbose >= CONV) t0.stop("cal U");
            t0 = turnback::timer();
#endif
            ws->z[tid].head(cols).setZero();
            if (lu->rank() > 0)
            {
#if TIMEMEASUREMENTS
                timer t1 = turnback::timer();
#endif
                for (int rr = 0; rr < lu->rank(); rr++)
                {
                    if (abs(lu->U().coeffRef(rr,rr)) < 1e-12)
                    {
                        cout << "[thread-" << tid << "] turnbackLU::computez: FATAL ERROR: diagonal of U not valid with value " << lu->U().coeffRef(rr,rr) << " on position " << rr << ", probably memory issue" << endl;
                        tbx->fatalError = true;
                        return 0;
                    }
                }
                if (lu->rank() != ws->u[tid].rows())
                {
                    cout << "[thread-" << tid << "] turnbackLU::computez: FATAL ERROR1: rows do not match, probably memory issue" << endl;
                    tbx->fatalError = true;
                    return 0;
                }
                if (lu->U().rows() != ws->u[tid].rows())
                {
                    cout << "[thread-" << tid << "] turnbackLU::computez: FATAL ERROR2: rows do not match, probably memory issue" << endl;
                    tbx->fatalError = true;
                    return 0;
                }
                ws->z[tid].head(lu->rank()) = lu->U().triangularView<Eigen::Upper>().solve(ws->u[tid]); // sparse solveinplace does not accept references for u 
#if TIMEMEASUREMENTS
                t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::solve Uz=u"));
                if (opt.verbose >= CONV) t1.stop("solveu");
#endif
            }
            ws->z[tid].head(cols).applyOnTheLeft(lu->Q());
            ws->z[tid][tbx->piv] = 1;
            ws->chosenCols[tbx->piv]++;
#if TIMEMEASUREMENTS
            t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::calc z"));
            if (opt.verbose >= CONV) t0.stop("createz");
#endif
            if (opt.verbose >= VAR && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::computez ws->z[tid].head(cols)\n" << ws->z[tid].head(cols).transpose() << endl;
#if SAFEGUARD
            if ((Ufull * lu->Q().transpose() * z.head(cols)).norm() > 1e-3)
            {
                // cout << "turnbackLU::computz: ERROR ||UfullQT z|| " << (Ufull * lu->Q().transpose()*z.head(cols)).norm() << endl;
                if ((Ufull * lu->Q().transpose() * z.head(cols)).norm() > 1e-1)
                {
                    cout << "Udiag\n" << lu->U().diagonal().transpose() << endl;
                    cout << "upart\n" << u.transpose() << endl;
                    throw;
                }
            }
#endif // SAFEGUARD
            // timer t1;
            // double nserrA = (A * ws->z[tid].head(cols)).norm();
            prodAbColWise(A, ws->z[tid], ws->elimCols_omp[tid], ws->tmp[tid]); // there is only one tmp
            double nserrA = ws->tmp[tid].head(rows).norm();
            // t1.stop("nserrA");
            if (testAzCtr > -1 && nserrA > nserrThres && !(tbx->leftI == 0 && tbx->rightI == cols-1))
            {
                if (opt.verbose >= BASIC) cout << "[thread-" << tid << "] turnbackLU::computez error turnback iteration " << j << " with error NS test Az " << nserrA << " and threshold " << nserrThres << endl;

                if (!xeidIdx)
                {
                    if (testAzCtr % 2 == 0 || tbx->rightI == cols-1)
                    {
                        if (tbx->leftsubsetnr >= 0)
                        {
                            addSubset(tid, tbx, "left");
                            return 0;
                        }
                    }
                    if (testAzCtr % 2 == 1 || tbx->leftI == 0)
                    {
                        if (tbx->rightsubsetnr < nrSubsets)
                        {
                            ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += addSubset(tid, tbx, "right");
                            return 0;
                        }
                        else
                        {
                            int diffRg = abs(ws->subsetRg.col(nrSubsets-2)(0) - ws->subsetRg.col(nrSubsets-1)(0));
                            tbx->rightI = min(cols-1, tbx->rightI + diffRg);
                            ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += diffRg;
                            return 0;
                        }
                    }
                    if (opt.verbose >= CONV && tid == opt.tid) cout << "subsetnr left right " << tbx->leftsubsetnr << " " << tbx->rightsubsetnr << " nrSubsets " << nrSubsets << endl;
                }
                else
                {
                    if (testAzCtr % 2 == 0 || tbx->rightI == cols-1)
                    {
                        if (tbx->leftI >= 0)
                        {
                            if (opt.verbose >= CONV && tid == opt.tid) cout << "computez nx > 0: add subset on the LEFT" << endl;
                            tbx->leftI = max(0, tbx->leftI - (xeidIdx->nx+xeidIdx->nu));
                            return 0;
                        }
                    }
                    if (testAzCtr % 2 == 1 || tbx->leftI == 0)
                    {
                        if (tbx->rightI < cols-1)
                        {
                            if (opt.verbose >= CONV && tid == opt.tid) cout << "computez nx > 0: add subset on the RIGHT" << endl;
                            tbx->rightI = min(cols-1, tbx->rightI + (xeidIdx->nx+xeidIdx->nu));
                            ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += xeidIdx->nx+xeidIdx->nu;
                            return 0;
                        }
                    }
                }
                if (tbx->leftsubsetnr < 0 && tbx->rightI >= cols)
                {
                    cout << "[thread-" << tid << "] computez: FATAL ERROR: RAN OUT OF SUBSETS TO ADD with range [" << tbx->leftI << ", " << tbx->rightI << "] with matrix " << rows << " x " << cols << " of rank " << rank0 << endl;
                    tbx->fatalError = true;
                    return 0;
                }
                cout << "turnbackLU::computez: FATAL ERROR: no subset added!!!" << endl;
                tbx->fatalError = true;
                return 0;
            }
            else if (nserrA > nserrThres)
            {
                if (nserrA < opt.nserrThresFatal) cout << "[thread-" << tid << "] turnbackLU::computez: z with error " << nserrA << " but accepted nonetheless due to LUSOL numerical difficulties!!!" << endl;
                else
                {
                    cout << "[thread-" << tid << "] turnbackLU::computez: z with FATAL error " << nserrA << " > " << opt.nserrThresFatal << " due to LUSOL numerical difficulties!!!" << endl;
                    tbx->fatalError = true;
                    return 0;
                }
                // if (nx > 0) throw;
            }
            else if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::computez: z clean with error " << nserrA << endl;
#if TIMEMEASUREMENTS
            t0 = turnback::timer();
#endif
#if MULTITHREADING
#pragma omp critical
#endif // MULTITHREADING
            {
                ws->thz.getTripletsVec(ws->z[tid], cols, 0, j); // this is threadded since there is only one thz for all threads
            }
#if TIMEMEASUREMENTS
            t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::getTripletsVec"));
            if (opt.verbose >= CONV && tid = opt.tid) t0.stop("getTripletsVec");
#endif
        }
        else if (opt.pivotM == 1 || opt.pivotM == 3)
        {
            if (tbx->nrCols - lu->rank() < min(rankZ-tbx->subsetNr * (xeidIdx->nu+xeidIdx->nf), xeidIdx->nu+xeidIdx->nf)) // insufficient nullspace
            {
                if (opt.verbose >= BASIC) cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: the nullspace of the current decomposition is not sufficient with tbx->nrCols " << tbx->nrCols << " rank " << lu->rank() << " remaining dimNS " << min(rankZ-dimNS, xeidIdx->nu+xeidIdx->nf) << ", need to add further columns on the left and right" << endl;
                addSubsetsUntilRup(tid, A, tbx);
                tbx->anew = true;
                return 0;
            }
            else
            {
                // get U1 and U2
                ws->chosenColsP[tid].head(cols).setConstant(-1);
                if (opt.pivotM == 1)
                {
                    ws->tmpU2Z[tid].conservativeResize(lu->rank(), xeidIdx->nu);
                    ws->tmpU2Z[tid].setZero();
                    lu->constructU_partial(ws->tmpU2Z[tid], tbx->subsetNr * (xeidIdx->nx+xeidIdx->nu), (tbx->subsetNr+1) * (xeidIdx->nx+xeidIdx->nu), xeidIdx->nu, ws->elimCols_omp[tid], ws->chosenColsP[tid], ws->chosenColsPinv[tid], tid);
                }
                else if (opt.pivotM == 3)
                {
                    ws->tmpU2Z[tid].conservativeResize(lu->rank(), tbx->rightI - tbx->leftI - lu->rank() + 1);
                    ws->tmpU2Z[tid].setZero();
                    lu->constructU_partial(ws->tmpU2Z[tid], tbx->leftI, tbx->rightI, xeidIdx->nu, ws->elimCols_omp[tid], ws->chosenColsP[tid], ws->chosenColsPinv[tid], tid);
                }
                ws->th[tid].reset();
                lu->U().triangularView<Eigen::Upper>().solveInPlace(ws->tmpU2Z[tid]);
                ws->th[tid].getTriplets(-ws->tmpU2Z[tid]);
                int maxCC = -1;
                for (int c = 0; c < cols; c++)
                {
                    if (ws->chosenColsP[tid][c] > -1)
                    {
                        ws->th[tid].setTriplet(c, ws->chosenColsP[tid][c], 1);
                        if (ws->chosenColsP[tid][c] > maxCC) maxCC = ws->chosenColsP[tid][c];
                    }
                }
                if (opt.pivotM == 1) ws->tmpU2Z[tid].conservativeResize(cols, xeidIdx->nu);
                else if (opt.pivotM == 3) ws->tmpU2Z[tid].conservativeResize(cols, tbx->rightI - tbx->leftI - lu->rank() + 1);
                ws->tmpU2Z[tid].setZero();
                ws->th[tid].setFromTriplets(ws->tmpU2Z[tid]);
                for (int c = 0; c < cols; c++)
                {
                    if (ws->chosenColsP[tid][c] == -1 && c-tbx->subsetNr * (xeidIdx->nx+xeidIdx->nu) >= lu->rank() && lu->qq()[c] <= (tbx->subsetNr+1) * (xeidIdx->nx+xeidIdx->nu))
                    {
                        maxCC++;
                        ws->th[tid].setTriplet(c, maxCC, 1);
                    }
                }
                // consider zero columns
                ws->th[tid].setFromTriplets(ws->tmpU2Z[tid]);
                lu->Q().applyThisOnTheLeft(ws->tmpU2Z[tid]);
                ws->AZ.conservativeResize(A.rows(), ws->tmpU2Z[tid].cols());
                ws->AZ = A * ws->tmpU2Z[tid];
                ws->AZ.makeCompressed();
                if (opt.pivotM == 3)
                {
                    ws->AZerr.head(ws->tmpU2Z[tid].cols()).setZero();
                    for (int c = 0; c < ws->tmpU2Z[tid].cols(); c++)
                    {
                        for (mat::InnerIterator it(ws->AZ,c); it; ++it) ws->AZerr[c] += pow(it.value(),2);
                        ws->AZerr[c] = sqrt(ws->AZerr[c]);
                    }
                    for (int c = 0 ; c < ws->tmpU2Z[tid].cols(); c++)
                    {
                        ws->AZerrsort[c] = c;
                    }
                    sort(ws->AZerrsort.begin(), ws->AZerrsort.begin() + ws->tmpU2Z[tid].cols(),
                            [&](const int& i1, const int& i2) {
                            return (ws->AZerr.data()[i1] < ws->AZerr.data()[i2]);
                            }
                        );
                }

                if (ws->AZ.norm() > 1e-11)
                {
                    cout << "Z\n" << (matd)ws->tmpU2Z[tid] << endl;
                    cout << "NS corrupted with AZ " << ws->AZ.norm() << endl;
                    throw;
                }
                else
                {
                    if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] Z of subset " << tbx->subsetNr << " clean with " << (A * ws->tmpU2Z[tid]).norm() << endl;
                    if (opt.pivotM == 1 && dimNS + ws->tmpU2Z[tid].cols() > rankZ)
                    {
                        cout << "dimNS + ws->tmpU2Z[tid].cols() > rankZ " << dimNS + ws->tmpU2Z[tid].cols() << " " << rankZ << endl;
                        cout << "Zhard\n" << (matd)ws->tmpU2Z[tid] << endl;
                        throw;
                    }
#if MULTITHREADING
#pragma omp critical
#endif // MULTITHREADING
                    {
                        if (opt.pivotM == 1)
                        {
                            for (int c = 0; c < ws->tmpU2Z[tid].cols(); c++)
                            {
                                ws->thz.getTripletsCol(ws->tmpU2Z[tid], c, dimNS);
                                dimNS++;
                                if (ws->chosenColsPinv[tid][c] > -1) ws->chosenCols[lu->qq()[ws->chosenColsPinv[tid][c]]-1] = 1;
                            }
                        }
                        else
                        {
                            // z vector from current subrange
                            int zctr = 0;
                            bool fin = false;
                            for (int c_ = 0; c_ < ws->tmpU2Z[tid].cols(); c_++)
                            {
                                int c = ws->AZerrsort[c_];
                                int cc = ws->chosenColsPinv[tid][c];
                                if (cc > -1)
                                {
                                    if (ws->chosenCols[lu->qq()[cc]-1] == 0 && 
                                            (tbx->subsetNr) * (xeidIdx->nx+xeidIdx->nu) <= lu->qq()[cc]-1  && lu->qq()[cc]-1 < (tbx->subsetNr+1) * (xeidIdx->nx+xeidIdx->nu))
                                        //&& ws->AZerr[c] < 1e-13)
                                    {
                                        ws->chosenCols[lu->qq()[cc]-1]++;
                                        ws->thz.getTripletsCol(ws->tmpU2Z[tid], c, dimNS);
                                        if (opt.verbose >= CONV) cout << "z[" << dimNS << "] in range with error " << ws->AZerr[c] << endl;
                                        dimNS++;
                                        zctr++;
                                        if (zctr >= xeidIdx->nu)
                                        {
                                            fin = true;
                                            break;
                                        }
                                    }
                                }
                                if (fin) break;
                            }
                            if (zctr < xeidIdx->nu)
                            {
                                for (int c_ = 0; c_ < ws->tmpU2Z[tid].cols(); c_++)
                                {
                                    int c = ws->AZerrsort[c_];
                                    int cc = ws->chosenColsPinv[tid][c];
                                    if (cc > -1)
                                    {
                                        if (ws->chosenCols[lu->qq()[cc]-1] == 0)
                                        {
                                            ws->chosenCols[lu->qq()[cc]-1]++;
                                            ws->thz.getTripletsCol(ws->tmpU2Z[tid], c, dimNS);
                                            dimNS++;
                                            if (opt.verbose >= CONV) cout << "z[" << dimNS << " not in range with error " << ws->AZerr[c] << endl;
                                            zctr++;
                                            if (zctr >= xeidIdx->nu)
                                            {
                                                fin = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (fin) break;
                                }
                            }
                            if (zctr < xeidIdx->nu)
                            {
                                for (int c_ = 0; c_ < ws->tmpU2Z[tid].cols(); c_++)
                                {
                                    int c = ws->AZerrsort[c_];
                                    int cc = ws->chosenColsPinv[tid][c];
                                    if (cc > -1)
                                    {
                                        ws->thz.getTripletsCol(ws->tmpU2Z[tid], c, dimNS);
                                        if (opt.verbose >= NONE) cout << "turnbackLU::computez: WARNING, double pivot, Z might be rank deficient" << endl;
                                        ws->chosenCols[lu->qq()[cc]-1]++;
                                        dimNS++;
                                        zctr++;
                                        if (zctr >= xeidIdx->nu)
                                        {
                                            fin = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // cout << "ws->chosenCols after " << ws->chosenCols.head(cols).transpose() << endl;
                }
#if TIMEMEASUREMENTS
                t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::solve Uz=u"));
                if (opt.verbose >= CONV) t1.stop("solveu");
#endif
            }
        }
        else if (opt.pivotM == 4)
        {
            if (tbx->nrCols - lu->rank() < min(rankZ-tbx->subsetNr * (xeidIdx->nu+xeidIdx->nf), xeidIdx->nu+xeidIdx->nf)) // insufficient nullspace
            {
                if (opt.verbose >= BASIC) cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: the nullspace of the current decomposition is not sufficient with tbx->nrCols " << tbx->nrCols << " rank " << lu->rank() << " remaining dimNS " << min(rankZ-dimNS, xeidIdx->nu+xeidIdx->nf) << ", need to add further columns on the left and right" << endl;
                addSubsetsUntilRup(tid, A, tbx);
                tbx->anew = true;
                return 0;
            }
            else
            {
                // get U1 and U2
                ws->chosenColsP[tid].head(cols).setConstant(-1);
                ws->tmpU2Z[tid].conservativeResize(lu->rank(), tbx->rightI - tbx->leftI - lu->rank() + 1);
                ws->tmpU2Z[tid].setZero();
                lu->constructU_partial(ws->tmpU2Z[tid], tbx->leftI, tbx->rightI, xeidIdx->nu, ws->elimCols_omp[tid], ws->chosenColsP[tid], ws->chosenColsPinv[tid], tid);
                ws->th[tid].reset();
                for (int c = 0; c < ws->tmpU2Z[tid].cols(); c++)
                {

                }
            }
        }
#if TIMEMEASUREMENTS
        tallz.stop(); times.push_back(turnback::time(tallz.time, "turnbackLU::allz"));
#endif
        return 1;
    }

    inline int turnbackLU::computez(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const int j, unique_ptr<turnbackIndices>& tbx, const mat& A, int testAzCtr)
    {
#if TIMEMEASUREMENTS
        turnback::timer tallz;
#endif

#if TIMEMEASUREMENTS
        turnback::timer t0;
#endif
        ws->u[tid] = vec::Zero(qr->rank());
        if (!getIdxXEID(tid, qr, tbx, A))
            return 0;
#if TIMEMEASUREMENTS
        t0.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::computezinitial"));
        if (opt.verbose >= CONV) t0.stop("computezinitial");
#endif
#if SAFEGUARD
        mat Upartprev;
#endif
        mat R = mat(qr->rank(), qr->rank());
        if (true || ws->xeidFacFresh[tid])
        {
            ws->th[tid].reset();
            // construct R
            for (int c = 0; c < qr->rank(); c++)
            {
                for (mat::InnerIterator it(qr->matrixR(),c); it; ++it)
                {
                    if (it.row() < qr->rank()) ws->th[tid].setTriplet(it.row(),c,it.value());
                }
            }
            ws->th[tid].setFromTriplets(R);

            ws->xeidFacFresh[tid] = false;
        }
        if (opt.verbose >= VAR && tid == opt.tid)
        {
            cout << "[thread-" << tid << "] t with " << ws->u[tid].rows() << " rows\n" << ws->u[tid].transpose() << endl;
            cout << "[thread-" << tid << "] R diag with " << R.cols() << " cols\n" << R.diagonal().transpose() << endl;
            if (opt.verbose >= MAT) cout << "[thread-" << tid << "] R with " << R.cols() << " cols\n" << (matd)R << endl;
        }

#if TIMEMEASUREMENTS
        t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::cal Upartial"));
        if (opt.verbose >= CONV) t0.stop("cal U");
        t0 = turnback::timer();
#endif
        ws->z[tid].head(cols).setZero();
        if (qr->rank() > 0)
        {
#if TIMEMEASUREMENTS
            timer t1 = turnback::timer();
#endif
            for (int rr = 0; rr < qr->rank(); rr++)
            {
                if (abs(R.coeffRef(rr,rr)) < 1e-12)
                {
                    cout << "[thread-" << tid << "] turnbackLU::computez: FATAL ERROR: diagonal of U not valid with value " << R.coeffRef(rr,rr) << " on position " << rr << ", probably memory issue" << endl;
                    tbx->fatalError = true;
                    return 0;
                }
            }
            ws->z[tid].head(qr->rank()) = -1 * R.triangularView<Eigen::Upper>().solve(ws->u[tid]); // sparse solveinplace does not accept references for u 
#if TIMEMEASUREMENTS
            t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::solve Uz=u"));
            if (opt.verbose >= CONV) t1.stop("solveu");
#endif
        }
        ws->z[tid].head(cols).applyOnTheLeft(qr->colsPermutation());
        ws->z[tid][tbx->piv] = 1;
#if TIMEMEASUREMENTS
        t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::calc z"));
        if (opt.verbose >= CONV) t0.stop("createz");
#endif
        if (opt.verbose >= VAR && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::computez ws->z[tid].head(cols)\n" << ws->z[tid].head(cols).transpose() << endl;
        double nserrA = (A * ws->z[tid].head(cols)).norm();
        if (testAzCtr > -1 && nserrA > nserrThres && !(tbx->leftI == 0 && tbx->rightI == cols-1))
        {
            if (opt.verbose >= BASIC) cout << "[thread-" << tid << "] turnbackLU::computez error turnback iteration " << j << " with error NS test Az " << nserrA << endl;
            throw;

            if (!xeidIdx)
            {
                if (testAzCtr % 2 == 0 || tbx->rightI == cols-1)
                {
                    if (tbx->leftsubsetnr >= 0)
                    {
                        addSubset(tid, tbx, "left");
                        return 0;
                    }
                }
                if (testAzCtr % 2 == 1 || tbx->leftI == 0)
                {
                    if (tbx->rightsubsetnr < nrSubsets)
                    {
                        ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += addSubset(tid, tbx, "right");
                        return 0;
                    }
                    else
                    {
                        int diffRg = abs(ws->subsetRg.col(nrSubsets-2)(0) - ws->subsetRg.col(nrSubsets-1)(0));
                        tbx->rightI = min(cols-1, tbx->rightI + diffRg);
                        ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += diffRg;
                        return 0;
                    }
                }
                if (opt.verbose >= CONV && tid == opt.tid) cout << "subsetnr left right " << tbx->leftsubsetnr << " " << tbx->rightsubsetnr << " nrSubsets " << nrSubsets << endl;
            }
            else
            {
                if (testAzCtr % 2 == 0 || tbx->rightI == cols-1)
                {
                    if (tbx->leftI >= 0)
                    {
                        if (opt.verbose >= CONV && tid == opt.tid) cout << "computez nx > 0: add subset on the LEFT" << endl;
                        tbx->leftI = max(0, tbx->leftI - (xeidIdx->nx+xeidIdx->nu));
                        return 0;
                    }
                }
                if (testAzCtr % 2 == 1 || tbx->leftI == 0)
                {
                    if (tbx->rightI < cols-1)
                    {
                        if (opt.verbose >= CONV && tid == opt.tid) cout << "computez nx > 0: add subset on the RIGHT" << endl;
                        tbx->rightI = min(cols-1, tbx->rightI + (xeidIdx->nx+xeidIdx->nu));
                        ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += xeidIdx->nx+xeidIdx->nu;
                        return 0;
                    }
                }
            }
            if (tbx->leftsubsetnr < 0 && tbx->rightI >= cols)
            {
                cout << "[thread-" << tid << "] computez: FATAL ERROR: RAN OUT OF SUBSETS TO ADD with range [" << tbx->leftI << ", " << tbx->rightI << "] with matrix " << rows << " x " << cols << " of rank " << rank0 << endl;
                tbx->fatalError = true;
                return 0;
            }
            cout << "turnbackLU::computez: FATAL ERROR: no subset added!!!" << endl;
            tbx->fatalError = true;
            return 0;
        }
        else if (nserrA > nserrThres)
        {
            if (nserrA < opt.nserrThresFatal) cout << "[thread-" << tid << "] turnbackLU::computez: z with error " << nserrA << " but accepted nonetheless due to LUSOL numerical difficulties!!!" << endl;
            else
            {
                cout << "[thread-" << tid << "] turnbackLU::computez: z with FATAL error " << nserrA << " > " << opt.nserrThresFatal << " due to LUSOL numerical difficulties!!!" << endl;
                tbx->fatalError = true;
                return 0;
            }
            // if (nx > 0) throw;
        }
        else if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::computez: z clean with error " << nserrA << endl;
#if TIMEMEASUREMENTS
        t0 = turnback::timer();
#endif
#if MULTITHREADING
#pragma omp critical
#endif // MULTITHREADING
        {
            ws->thz.getTripletsVec(ws->z[tid], cols, 0, j); // this is threadded since there is only one thz for all threads
        }
#if TIMEMEASUREMENTS
        t0.stop(); times.push_back(turnback::time(t0.time, "turnbackLU::getTripletsVec"));
        if (opt.verbose >= CONV && tid = opt.tid) t0.stop("getTripletsVec");
#endif
#if TIMEMEASUREMENTS
        tallz.stop(); times.push_back(turnback::time(tallz.time, "turnbackLU::allz"));
#endif
        return 1;
    }

    inline bool turnbackLU::getIdxXEID(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const mat& A)
    {
        if (ws->elimCols_omp[tid][tbx->piv] == 0)
        {
            cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: ws->elimCols_omp[tid] " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
            cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: tbx->pivot column " << tbx->piv << " is already in lu" << endl;
            throw;
        }
        int status = addCol(tid, lu, A, tbx->piv, 3);
        if (status == 1) // rank increase!
        {
            if (opt.verbose >= BASIC && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: adding the tbx->pivot column " << tbx->piv << " lead to rank increase to " << lu->rank()+1 << ", readd tbx->pivot column " << tbx->piv << " and then need to add further columns on the left and right" << endl;
            addSubsetsUntilRup(tid, A, tbx);
            tbx->anew = true;
            return false;
        }

        ws->elimCols_omp[tid](tbx->piv) = 0;
        tbx->pivU = getIdxFromPiv(lu, tbx);
        return true;
    }

    inline bool turnbackLU::getIdxXEID(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, unique_ptr<turnbackIndices>& tbx, const mat& A)
    {
        if (ws->elimCols_omp[tid][tbx->piv] == 0)
        {
            cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: ws->elimCols_omp[tid] " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
            cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: tbx->pivot column " << tbx->piv << " is already in lu" << endl;
            throw;
        }
        int status = addCol(tid, qr, A, tbx->piv, 3);
        if (status == 1) // rank increase!
        {
            if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::getIdxXEID: adding the tbx->pivot column " << tbx->piv << " lead to rank increase to " << qr->rank()+1 << ", readd tbx->pivot column " << tbx->piv << " and then need to add further columns on the left and right" << endl;
            addSubsetsUntilRup(tid, A, tbx);
            tbx->anew = true;
            return false;
        }

        ws->elimCols_omp[tid](tbx->piv) = 0;
        return true;
    }

    inline void turnbackLU::getIdx(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const mat& A)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        double cond = lu->getUcond();
        double condTol = max(cond, opt.condThres);
        // double cond = 0; //lu->getUcond();
        // if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: The approximate condition number of U is " << cond << endl;
        if (cond > opt.condThres)
        {
            initialFac_bin(tid, lu, A);
            condTol = max(condTol, 10*lu->getUcond());
            if (opt.verbose >= CONV && tid == opt.tid)
            {
                cout << "[thread-" << tid << "] getIdx: After refactorization, the approximate condition number of U is " << lu->getUcond() << " before: " << cond << endl;
                cond = lu->getUcond();
            }
        }
        bool found = false;
        int jrepm = tbx->bcur-1;
        if (order == 1) jrepm = tbx->bcur+1;
        bool leftfinished = false;
        bool rightfinished = false;
        if (opt.verbose >= VAR)
        {
            cout << "[thread-" << tid << "] turnbackLU::getIdx: tbx->piv " << tbx->piv << " tbx->jrep " << tbx->jrep << " jrepm " << jrepm << " tbx->bcur " << tbx->bcur << " tbx->bnext " << tbx->bnext <<" lu->rank() " << lu->rank() << endl;
        }
        if (opt.verbose >= VAR)
        {
            cout << "[thread-" << tid << "] turnbackLU::getIdx: elimcols  : " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
            cout << "[thread-" << tid << "] turnbackLU::getIdx: chosencols: " << ws->chosenCols.head(cols).transpose() << endl;
        }

        // Do some checks for good measure
        if (opt.verbose >= MAT)
        {
            lu->_Q = NULL; lu->_U = NULL; lu->_P = NULL;
            cout << "[thread-" << tid << "] turnbackLU::getIdx: U\n" << (matd)lu->U() << endl;
        }
        if (lu->rank() >= cols) 
        {
            // something went wrong with the rank in updating
            cout << "[thread-" << tid << "] turnbackLU::getIdx: lu->rank() == cols; something went wrong with the rank in updating" << endl;
            throw;
        }
        if (ws->chosenCols[tbx->piv] != 0)
        {
            cout << "[thread-" << tid << "] turnbackLU::getIdx: we should not be here, this tbx->pivot " << tbx->piv << " has already been used before" << endl;
            throw;
        }
        if (ws->elimCols_omp[tid][tbx->piv] != 0)
        {
            cout << "[thread-" << tid << "] ws->elimCols_omp[tid] " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
            cout << "[thread-" << tid << "] turnbackLU::getIdx: tbx->pivot column " << tbx->piv << " is not in lu" << endl;
            // addCol(A, tbx->piv);
            throw;
        }

        // find the column index idx in permuted U = [U1 U2] (UQ) that corrsponds to a column in A(:,tbx->bcur:tbx->bnext-1) and that is not eliminated and has not been chosen before
        // this column is then used as u (compared to Upartial, see computez)
        // in the composed z, we set z[idx] = 1
        // this is for full rank guarantuee of Z
        // Z = [1 a b ...; 0 1 c ...; 
        // ws->elimCols_omp[tid] and ws->chosenCols_omp[tid] are in the original column reference of A
        // invqq and invpp are not updated in lusol updates, need to do by hand
        int getIdxIter = 0;
        while (!found)
        {
            // check tbx->pivot column
            if (opt.verbose >= VAR)
            {
                cout << "< < < < < turnbackLU::getidx iteration " << getIdxIter << " with tbx->piv " << tbx->piv << " luinvqq " << lu->invqq()[tbx->piv]-1 << " ws->elimCols_omp[tid][tbx->piv] " << ws->elimCols_omp[tid][tbx->piv] << " ws->chosenCols_omp[tid][tbx->piv] " << ws->chosenCols[tbx->piv] << endl;
                getIdxIter++;
            }

            tbx->pivU = getIdxFromPiv(lu, tbx);
            if (tbx->pivU >= lu->rank()) 
            {
                found = true;
                // The identified idx is already in U2 (with U1 in R(m X lu->rank()))
                if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::getIdx: the tbx->pivot column " << tbx->piv << " is already in U2 (pivU: " << tbx->pivU << ", lu->qq()[pivU]: " << lu->qq()[tbx->pivU]-1 << ", tbx->piv: " << tbx->piv << " lu->rank(): " << lu->rank()<< ", curNrCols " << curNrCols << ")" << endl;
                double cond = lu->getUcond();
                // double cond = 0; //lu->getUcond();
                if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: after detecting tbx->piv in u2, condition number of U is " << cond << endl;
                if (!(leftfinished && rightfinished) && cond > condTol)
                {
                    if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: !(leftfinished && rightfinished) && cond > " << condTol << ", refactorize" << endl;
                    initialFac_bin(tid, lu, A);
                    condTol = max(condTol, 10*lu->getUcond());
                    if (opt.verbose >= CONV && tid == opt.tid)
                    {
                        cout << "[thread-" << tid << "] getIdx: After refactorization, the approximate condition number of U is " << lu->getUcond() << " before: " << cond << endl;
                        cond = lu->getUcond();
                    }
                    tbx->pivU = getIdxFromPiv(lu, tbx);
                    if (tbx->pivU < lu->rank()) found = false;
                }
                else break;
            }
            if (tbx->pivU < lu->rank())
            {
                // The tbx->pivot column idx is in U1 (with U1 in R(m X lu->rank()))
                // we need to remove it from U1 and add it to U2
                // get length of the tbx->pivot row
                bool foundnnz = false;
                if (lu->lenRow(tbx->pivU) > 0) 
                {
                    if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: the actual tbx->pivot column " << tbx->piv << " has nnz" << endl;
                    foundnnz = true;
                }
                else
                    if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: the actual tbx->pivot column " << tbx->piv << " at local index " << tbx->pivU << " has no nnz with length " << lu->lenRow(tbx->pivU) << "!!!" << endl;
                if (foundnnz || lu->getUcond() > condTol)
                {
                    // remove the column from U1 and add to U2 with repCol mode3 (non-permuting)
                    // this also, for good measure, checks the actual linear dependency between the tbx->pivot column and U2
                    // remove column
                    int curRank = lu->rank();
                    remCol(tid, lu, tbx->piv, A);
                    if (lu->rank() != curRank)
                    {
                        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::getIdx: removing the tbx->pivot column " << tbx->piv << " lead to rank decrease or increase from " << curRank << " to " << lu->rank() << ", readd tbx->pivot column " << tbx->piv << " and then need to add further columns on the left and right" << endl;
                        // readd column
                        addCol(tid, lu, A, tbx->piv);
                    }
                    else
                    {
                        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::getIdx: removing the tbx->pivot column " << tbx->piv << " didnt change the rank " << lu->rank() << ", readd column tbx->piv " << tbx->piv << " in lu8rpc mode2: 3" << endl;
                        if (opt.verbose >= CONV && tid == opt.tid && !foundnnz)
                        { 
                            lu->_Q = NULL; lu->_U = NULL; lu->_P = NULL;
                            cout << "foundnnz is false with condition number " << lu->getUcond() << endl;
                            cout << "lu->U(),idx " << tbx->pivU << "\n" << lu->U().row(tbx->pivU) << endl;
                            if (opt.verbose >= MAT) cout << "lu->U()\n" << (matd)lu->U() << endl;
                            // throw;
                        }
                        // add this column to U2 without tbx->pivoting (so it is on the right in U2 and can be chosen as u)
                        addCol(tid, lu, A, tbx->piv, 3);
                        if (lu->rank() > curRank)
                        {
                            initialFac_bin(tid, lu, A);
                            condTol = max(condTol, 10*lu->getUcond());
                            if (opt.verbose >= CONV && tid == opt.tid)
                            {
                                double cond = lu->getUcond();
                                cout << "[thread-" << tid << "] getIdx: After readding tbx->pivot column in mode 1, the rank of U is " << lu->rank() << endl;
                            }
                        }
                        else
                        {
                            // condition number check
                            double cond = lu->getUcond();
                            // double cond = 0; // lu->getUcond();
                            if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: The approximate condition number is " << cond << " leftfinished && rightfinished " << (leftfinished && rightfinished) << endl;
                            if (cond <= condTol || (leftfinished && rightfinished))
                            {
                                tbx->pivU = getIdxFromPiv(lu, tbx);
                                found = true;
                                if (opt.verbose >= VAR) cout << "[thread-" << tid << "] turnbackLU::getIdx: the tbx->pivot column " << tbx->piv << " is idx " << tbx->pivU << ", lu->qq()[idx]-1: " << lu->qq()[tbx->pivU]-1 << " lu->rank(): " << lu->rank()<< ")" << endl;
                                break;
                            }
                            else
                            {
                                if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] getIdx: bad conditioning, keep adding columns" << endl;
                                remCol(tid, lu, tbx->piv, A);
                                addCol(tid, lu, A, tbx->piv);
                                if (opt.verbose >= CONV && tid == opt.tid)
                                {
                                    double cond = lu->getUcond();
                                    cout << "[thread-" << tid << "] getIdx: After readding tbx->pivot column in mode 1, the approximate condition number of U is " << cond << endl;
                                }
                            }
                        }
                    }
                }
            }
            if (opt.verbose >= VAR) cout << "[thread-" << tid << "] turnbackLU::getIdx: idx in local U frame: " << tbx->pivU << ", lu->rank() " << lu->rank() << " idx in global A frame (tbx->piv): " << lu->qq()[tbx->pivU]-1 << endl;
            // the chosen tbx->pivot column is within U1
            // removing it from there lead to rank decrease
            // add columns until we can remove the tbx->pivot column without rank decrease of lu
            if (!found)
            {
                if (opt.verbose >= VAR) cout << "[thread-" << tid << "] turnbackLU::getIdx: add column until tbx->pivot column is linear dependent" << endl;
                // interchangeably add columns from the left and right (in order to preserve as much block sparsity as possible)
                if (leftfinished && rightfinished)
                {
                    int colRg;
                    if (order == 0) colRg = cols-tbx->bcur;
                    else if (order == 1) colRg = tbx->bcur+1;
                    cout << "turnbackLU::getIdx: FATAL ERROR: maximum number of columns reached with tbx->jrep " << tbx->jrep << " and jrepm " << jrepm << "; most likely something went wrong during the updating; REFACTORIZE MATRIX OF RANGE " << colRg << "!!!!!!!!!!!!!" << endl;
                    tbx->pivU = -1;
                    tbx->fatalError = true;
                }
                else
                {
                    if (!leftfinished && rightfinished) // if right side is finished, add columns from the left
                    {
                        // add column on the left
                        // note that these columns will be removed again in the main loop (before stage1)
                        while ((jrepm < cols && jrepm >= 0) && (ws->pivotCols_[jrepm] > 0 || ws->elimCols_omp[tid][jrepm] == 0)) // only add non-tbx->pivot columns for rank criterion
                        {
                            if (order == 0) jrepm--;
                            else if (order == 1) jrepm++;
                        }
                        if (jrepm > cols-1 || jrepm < 0) leftfinished = true;
                        else
                        {
                            if (opt.verbose >= VAR) cout << "turnbackLU::getIdx: add column on the left " << jrepm << endl;
                            addCol(tid, lu, A, jrepm);
                            if (order == 0) jrepm--;
                            else if (order == 1) jrepm++;
                        }
                    }
                    if (!rightfinished) // do right side first
                    {
                        if (tbx->jrep > cols-1 || tbx->jrep < 0) rightfinished = true;
                        else
                        {
                            // add column on the right
                            if (opt.verbose >= VAR) cout << "turnbackLU::getIdx: add column on the right " << tbx->jrep << endl;
                            addCol(tid, lu, A, tbx->jrep);
                            if (order == 0) tbx->jrep++;
                            else if (order == 1) tbx->jrep--;
                        }
                    }
                }
            }
        }
        if (opt.verbose >= VAR) cout << "turnbackLU::getIdx: idx in local U frame: " << tbx->pivU << ", idx in global A frame (tbx->piv): " << lu->qq()[tbx->pivU]-1 << " tbx->jrep " << tbx->jrep << endl;
        ws->chosenCols[tbx->piv]++;
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::getIdx"));
        if (opt.verbose >= CONV && tid == opt.tid) tm1.stop("getIdx");
#endif
    }

    inline int turnbackLU::getIdxFromPiv(shared_ptr<cpplusol::eigenlusol>& lu, const unique_ptr<turnbackIndices>& tbx)
    {
        lu->updateInvpq();
        return lu->invqq()[tbx->piv]-1;
    }

    inline bool turnbackLU::remCols(int tid, shared_ptr<cpplusol::eigenlusol>& lu, unique_ptr<turnbackIndices>& tbx, const string dir, const mat& A)
    {
        if (opt.verbose >= MAT) cout << "remCols order " << order << " ws->elimCols_omp[tid] " << ws->elimCols_omp[tid].head(cols).transpose() << endl;
        bool acted = false;
        if (order == 0)
        {
            if (dir == "left")
            {
                if (opt.verbose >= VAR) cout << "lurank " << lu->rank() << " range " << abs(tbx->jrep - tbx->bcur) << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << endl;
                while (lu->rank() > 1 && lu->rank() < tbx->jrep - tbx->bcur && tbx->jrep >= 0)
                {
                    if (ws->elimCols_omp[tid][tbx->jrep] == 0) remCol(tid, lu, tbx->jrep, A);
                    tbx->jrep--;
                    acted = true;
                    if (opt.verbose >= VAR) cout << "lurank " << lu->rank() << " range " << abs(tbx->jrep - tbx->bcur) << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << endl;
                }
                if (acted) tbx->jrep++;
            }
            else if (dir == "right")
            {
                while (tbx->jrep < tbx->bcur)
                {
                    {
                        if (ws->elimCols_omp[tid][tbx->jrep] == 0) remCol(tid, lu, tbx->jrep, A);
                        acted = true;
                    }
                    tbx->jrep++;
                }
                if (acted) tbx->jrep--;
            }
        }
        else if (order == 1)
        {
            if (dir == "right")
            {
                if (opt.verbose >= VAR) cout << "lurank " << lu->rank() << " range " << abs(tbx->jrep - tbx->bcur) << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << endl;
                while (lu->rank() > 1 && lu->rank() < tbx->bcur - tbx->jrep && tbx->jrep < cols)
                {
                    if (ws->elimCols_omp[tid][tbx->jrep] == 0) remCol(tid, lu, tbx->jrep, A);
                    tbx->jrep++;
                    acted = true;
                    if (opt.verbose >= VAR) cout << "lurank " << lu->rank() << " range " << abs(tbx->jrep - tbx->bcur) << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << endl;
                }
                if (acted) tbx->jrep--;
            }
            else if (dir == "left")
            {
                while (tbx->jrep > tbx->bcur)
                {
                    if (ws->elimCols_omp[tid][tbx->jrep] == 0) remCol(tid, lu, tbx->jrep, A);
                    tbx->jrep--;
                    acted = true;
                }
                if (acted) tbx->jrep++;
            }
        }
        return acted;
    }

    inline void turnbackLU::stage1(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
        // add cols on the right until rank increase
        int jrep_ = tbx->jrep;
        if (order == 0)
        {
            for (; tbx->jrep < cols; tbx->jrep++)
            {
                if (opt.verbose >= VAR) cout << "[thread-" << tid << "] stage1: current rank " << lu->rank() << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << " range (tbx->jrep-tbx->bcur) " << abs(tbx->jrep - tbx->bcur) << " tbx->piv " << tbx->piv << " ws->elimCols_omp[tid][tbx->piv] " << ws->elimCols_omp[tid][tbx->piv] << endl;
                if (lu->rank() > 0 && lu->rank() < tbx->jrep - tbx->bcur && (curNrCols > 1 || cols == 1) && ws->elimCols_omp[tid][tbx->piv] == 0)
                {
                    if (opt.verbose >= VAR) cout << "[thread-" << tid << "] stage1: add " << tbx->jrep-jrep_ << " cols with offset b " << tbx->bcur << endl;
                    break;
                }
#if TIMEMEASUREMENTS
                turnback::timer t2;
#endif
                if (ws->elimCols_omp[tid][tbx->jrep] == 1) addCol(tid, lu, A, tbx->jrep);
#if TIMEMEASUREMENTS
                t2.stop(); times.push_back(turnback::time(t2.time, "[thread-" << tid << "] turnbackLU::stage1: addCol"));
#endif
            }
        }
        else if (order == 1)
        {
            for (; tbx->jrep >= 0; tbx->jrep--)
            {
                if (opt.verbose >= VAR) cout << "[thread-" << tid << "] stage1: current rank " << lu->rank() << " tbx->jrep " << tbx->jrep << " tbx->bcur " << tbx->bcur << " range (tbx->jrep-tbx->bcur) " << abs(tbx->jrep - tbx->bcur) << endl;
                if (lu->rank() > 0 && lu->rank() < tbx->bcur - tbx->jrep && (curNrCols > 1 || cols == 1) && ws->elimCols_omp[tid][tbx->piv] == 0)
                {
                    if (opt.verbose >= VAR) cout << "[thread-" << tid << "] stage1: add " << jrep_-tbx->jrep << " cols with offset b " << tbx->bcur << endl;
                    break;
                }
#if TIMEMEASUREMENTS
                turnback::timer t2;
#endif
                if (ws->elimCols_omp[tid][tbx->jrep] == 1) addCol(tid, lu, A, tbx->jrep);
#if TIMEMEASUREMENTS
                t2.stop(); times.push_back(turnback::time(t2.time, "[thread-" << tid << "] turnbackLU::stage1: addCol"));
#endif
            }
        }
    }

    inline int turnbackLU::addSubset(int tid, unique_ptr<turnbackIndices>& tbx, string dir)
    {
        int diffRg = 0;
        if (dir == "left")
        {
            // add column subset on the left
            diffRg = tbx->leftI - ws->subsetRg.col(tbx->leftsubsetnr)(0);
            tbx->leftI = ws->subsetRg.col(tbx->leftsubsetnr)(0);
            tbx->leftsubsetnr--;
            if (opt.verbose >= CONV && tid == opt.tid)
            {
                cout << "addSubset: add subset on the LEFT" << endl;
            }
        }
        else if (dir == "right")
        {
            if (opt.verbose >= CONV && tid == opt.tid) cout << "addSubset: add subset on the RIGHT" << endl;
            // add column subset on the right
            bool assigned = false;
            int rightIold = tbx->rightI;
            for (; tbx->rightsubsetnr < nrSubsets; tbx->rightsubsetnr++)
            {
                if (opt.verbose >= CONV && tid == opt.tid) cout << "tbx->subsetNr " << tbx->subsetNr << " rightsubsetnr " << tbx->rightsubsetnr << " nrSubsets " << nrSubsets << " ws->subsetRg.col(tbx->rightsubsetnr)(0) > tbx->rightI " << ws->subsetRg.col(tbx->rightsubsetnr)(0) << " > " << tbx->rightI << endl;
                if (ws->subsetRg.col(tbx->rightsubsetnr)(0) > tbx->rightI)
                {
                    diffRg = ws->subsetRg.col(tbx->rightsubsetnr)(0) - tbx->rightI;
                    tbx->rightI = min(cols-1, tbx->rightI+diffRg);
                    tbx->rightsubsetnr++;
                    assigned = true;
                    break;
                }
            }

            if (!assigned) 
            {
                diffRg = abs(ws->subsetRg.col(nrSubsets-2)(0) - ws->subsetRg.col(nrSubsets-1)(0));
                tbx->rightI = min(cols-1, tbx->rightI+diffRg);
            }
        }
        return diffRg;
    }

    inline bool turnbackLU::addSubsetsUntilRup(int tid, const mat& A, unique_ptr<turnbackIndices>& tbx)
    {
        bool found = false;
        if (opt.verbose >= VAR && tid == opt.tid) cout << "[thread-" << tid << "] turnbackLU::addSubsetsUntilRup: add column until tbx->pivot column is linear dependent" << endl;
        // first add subsets on left, then on right
        if (tbx->leftI == 0 && tbx->rightI == cols-1)
        {
            cout << "[thread-" << tid << "] addSubsetsUntilRup: FATAL ERROR: RAN OUT OF SUBSETS TO ADD with range [" << tbx->leftI << ", " << tbx->rightI << "] with matrix " << rows << " x " << cols << " of rank " << rank0 << endl;
            tbx->fatalError = true;
            return false;
        }
        else
        {
            if (opt.verbose >= VAR && tid == opt.tid)
            {
                cout << "tbx->subsetNr " << tbx->subsetNr << " [leftI rightI]: " << tbx->leftI << " " << tbx->rightI << " leftsubsetnr " << tbx->leftsubsetnr << " rightsubsetnr " << tbx->rightsubsetnr << " nrSubsets " << nrSubsets << endl;
                cout << "subsetRg\n" << ws->subsetRg.leftCols(nrSubsets) << endl;
            }
            if (!xeidIdx)
            {
                // add column subset on the left
                if (tbx->leftsubsetnr >= 0)
                {
                    addSubset(tid, tbx, "left");
                    return true;
                }
                if (tbx->rightsubsetnr < nrSubsets)
                {
                    addSubset(tid, tbx, "right");
                    ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() = tbx->rightI;
                    return true;
                }
                else
                {
                    int diffRg = abs(ws->subsetRg.col(nrSubsets-2)(0) - ws->subsetRg.col(nrSubsets-1)(0));
                    tbx->rightI = min(cols-1, tbx->rightI + diffRg);
                    ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() += diffRg;
                    return true;
                }
            }
            else // nx > 0
            {
                if (tbx->subsetDir == "left")
                {
                    if (opt.verbose >= VAR && tid == opt.tid) cout << "addSubsetsUntilRup: add subset on the LEFT" << endl;
                    tbx->leftI = max(0, tbx->leftI - (xeidIdx->nx+xeidIdx->nu));
                    tbx->subsetDir = "right";
                }
                else if (tbx->subsetDir == "right")
                {
                    if (opt.verbose >= VAR && tid == opt.tid) cout << "addSubsetsUntilRup: add subset on the RIGHT" << endl;
                    tbx->rightI = min(cols-1, tbx->rightI + xeidIdx->nx+xeidIdx->nu);
                    tbx->subsetDir = "left";
                }
                return true;
            }
        }
        return found;
    }

    inline int turnbackLU::addCol(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const mat& A, const int colidx, const int mode2)
    {
        // add column
#if TIMEMEASUREMENTS
        turnback::timer t2;
#endif
        int rankOld = lu->rank();
        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] --- addCol: replace column " << colidx << ", curRank " << lu->rank() << ", curNrCols " << curNrCols << " mode2 " << mode2 << " ws->elimCols_omp[tid][colidx] " << ws->elimCols_omp[tid][colidx] << " tbx->ws->pivotCols_[colidx] " << ws->pivotCols_[colidx];
        if (opt.verbose >= VAR && tid == opt.tid) cout << " with column " << A.col(colidx).transpose();
        if (opt.verbose >= CONV && tid == opt.tid) cout << endl;
        int status = 0;
        if (ws->elimCols_omp[tid][colidx] == 1) 
        {
            status = lu->repCol(A, colidx, colidx, 0, mode2);
            if (mode2 != 3) ws->elimCols_omp[tid](colidx) = 0;
            curNrCols++;
            nrAddCols++;
            if (status == 9)
            {
                cout << "turnback::addcol: FATAL error in clusol, refactorize" << endl;
                initialFac_bin(tid, lu, A);
            }
            else if (lu->rank() > rankOld)
                status = 1;
        } // old column is assumed to be zero in lusol
        else
            cout << "turnbackLU::addCol: WARNING: attempt to add column " << colidx << " which is already in the LU decomposition" << endl; 
#if TIMEMEASUREMENTS
        t2.stop(); times.push_back(turnback::time(t2.time, "turnbackLU::addCol: addCol"));
#endif
        if (opt.verbose >= CONV && tid == opt.tid) cout << ", new rank: lurank " << lu->rank() << " colidx " << colidx << endl;
#if SAFEGUARD
        _A.col(colidx) = (vec)A.col(colidx);
#endif
        return status;
    }

    inline int turnbackLU::addCol(int tid, shared_ptr<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >& qr, const mat& A, const int colidx, const int mode2)
    {
        // add column
#if TIMEMEASUREMENTS
        turnback::timer t2;
#endif
        int rankOld = qr->rank();
        if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] --- addCol: replace column " << colidx << ", curRank " << qr->rank() << ", curNrCols " << curNrCols << " mode2 " << mode2 << " ws->elimCols_omp[tid][colidx] " << ws->elimCols_omp[tid][colidx] << " tbx->ws->pivotCols_[colidx] " << ws->pivotCols_[colidx];
        if (opt.verbose >= VAR && tid == opt.tid) cout << " with column " << A.col(colidx).transpose();
        if (opt.verbose >= CONV && tid == opt.tid) cout << endl;
        int status = 0;
        if (ws->elimCols_omp[tid][colidx] == 1) 
        {
            vec v = vec::Zero(A.rows());
            for (mat::InnerIterator it(A,colidx); it; ++it)
            {
                v[it.row()] = it.value();
            }

            v = qr->matrixQ().transpose() * v;
            if (v.tail(A.rows() - qr->rank()).norm() > 1e-9)
            {
                cout << "QTv " << v.transpose() << endl;
                cout << "The pivot column is not linear independent with tail norm " << v.tail(A.rows() - qr->rank()).norm() << endl;
                status = 1;
            }
            else ws->u[tid] = v.head(qr->rank());
            curNrCols++;
            nrAddCols++;
        } // old column is assumed to be zero in lusol
        else
            cout << "turnbackLU::addCol: WARNING: attempt to add column " << colidx << " which is already in the LU decomposition" << endl; 
#if TIMEMEASUREMENTS
        t2.stop(); times.push_back(turnback::time(t2.time, "turnbackLU::addCol: addCol"));
#endif
        if (opt.verbose >= CONV && tid == opt.tid) cout << ", new rank: lurank " << qr->rank() << " colidx " << colidx << endl;
#if SAFEGUARD
        _A.col(colidx) = (vec)A.col(colidx);
#endif
        return status;
    }

    inline void turnbackLU::remCol(int tid, shared_ptr<cpplusol::eigenlusol>& lu, const int colidx, const mat& A)
    {
#if TIMEMEASUREMENTS
        turnback::timer t2;
#endif
        if (ws->elimCols_omp[tid][colidx] == 0)
        {
            if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] --- RemCol: replace column " << colidx  << " with zero, current rank " << lu->rank() << ", curNrCols " << curNrCols << " ws->elimCols_omp[tid] " << ws->elimCols_omp[tid][colidx];
            if (opt.verbose >= VAR && tid == opt.tid) cout << " with column " << A.col(colidx).transpose();
            int status = lu->repCol(A, 0, colidx, 1, 0); 
            ws->elimCols_omp[tid](colidx) = 1;
            curNrCols--;
            nrRemCols++;
            if (status == 9)
            {
                cout << "turnback::addcol: FATAL error in clusol, refactorize" << endl;
                initialFac_bin(tid, lu, A);
            }
            if (opt.verbose >= CONV && tid == opt.tid) cout << ", new rank: lurank " << lu->rank() << " colidx " << colidx << endl;
#if SAFEGUARD
            _A.col(colidx) = vec::Zero(rows);
#endif
#if TIMEMEASUREMENTS
            t2.stop(); times.push_back(turnback::time(t2.time, "turnbackLU::remCol: remCol"));
#endif
        }
        else
        {
            if (opt.verbose >= CONV && tid == opt.tid) cout << "[thread-" << tid << "] --- RemCol: WARNING: attempt to remove column " << colidx  << " which is not in decomposition" << endl;
        }
    }

} // namespace 
#endif // _TURNBACKLU_
