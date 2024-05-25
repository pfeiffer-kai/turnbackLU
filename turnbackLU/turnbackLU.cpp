// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "turnbackLU.h"
#include <clusol/clusol.h>
#include <omp.h>

namespace turnback
{
    template <typename T> void print(string name, T* a, int len)
    {
        cout << name << ":\n";
        for (int i=0;i<len;i++)
        {
            cout << (a)[i] << " ";
        }
        cout << endl;
    }

    turnbackLU::turnbackLU(shared_ptr<cpplusol::eigenlusol> lu_, options opt_) :
        opt(opt_)
    {
#if MULTITHREADING
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) nrThreads = omp_get_num_threads();
        }
#else
        nrThreads = 1;
        opt.tid = 0;
#endif
        if (opt.verbose >= CONV) cout << "turnbackLU: number of threads available: " << nrThreads << endl;

        // set workspace
        ws = make_shared<data>(lu_, nrThreads, opt); // FIXME: proper dimensioning of eigenlusol

        nserrThres = opt.nserrThres;
    }

    void turnbackLU::reset(const mat& A, const std::tuple<int,int,int,int> pattern_)
    {
        rows = A.rows();
        cols = A.cols();
        dimNS = 0;
        curNrCols = 0;
        nrAddCols = 0;
        nrRemCols = 0;
        nrFac = 0;
        nrFac_bin = 0;

        ws->reset(rows, cols);
        for (shared_ptr<cpplusol::eigenlusol>& lu_omp_ : ws->lu_omp) lu_omp_->reset();

        if (opt.pattern)
        {
            xeidIdx = std::make_shared<xeidIndices>(pattern_);
            if (!xeidIdx->checkPattern())
            {
                if (xeidIdx->T > 0) nodes = xeidIdx->T;
                xeidIdx = NULL;
            }
        }
        else
            nodes = std::get<3>(pattern_);


#if SAFEGUARD
        zcollected.clear();
#endif
    }

    void turnbackLU::computeNS(const mat& A, shared_ptr<mat> Zio, const std::tuple<int,int,int,int> pattern_, const int nrEcs, const bool eqProj)
    {
        if (opt.verbose >= MAT) cout << "turnbackLU::computeNS: A\n" << (matd)A << endl;
        if (opt.colUpdates) { cout << "column updates deprecated; see older commits!" << endl; throw; }
        reset(A, pattern_);
#if SAFEGUARD
        if (opt.facM == 0)
        {
            ws->lu0->computeNS(A, Zio, true);
            if (opt.verbose >= CONV) cout << "A dim " << A.rows() << " x " << A.cols() << ", rank " << ws->lu0->rank() << endl;
            if (opt.verbose >= CONV) cout << "lu->Z() dim " << ws->lu0->Z().rows() << " x " << ws->lu0->Z().cols() << endl;
            if (opt.verbose >= CONV && ws->lu0->Z().cols() > 0) cout << "turnbackLU::computeNS: AZstandard " << (A * ws->lu0->Z()).norm() << endl;
            if (opt.verbose >= CONV) cout << "Z\n" << (matd)ws->lu0->Z() << endl;
            if (opt.verbose >= CONV) if (ws->lu0->Z().cols() > 0) cout << "NS error standard LU NS: " << (A * ws->lu0->Z()).norm() << endl;
            ws->lu0->_Q = NULL; ws->lu0->_U = NULL; ws->lu0->_P = NULL;
            if (opt.verbose >= VAR) cout << "turnbackLU:: U\n" << (matd)ws->lu0->U().diagonal().transpose() << endl;
            Eigen::ColPivHouseholderQR<matd> qrd((matd)A);
            if (opt.verbose >= CONV) cout << "qrd rank " << qrd.rank() << endl;
            nserrThres = max(1e-12, 4*(A * ws->lu0->Z()).norm());
            computeLUNS(tid);
            ws->thz.reset();
        }
        else
        {
            ws->qr->compute(A);
            ws->qr->setPivotThreshold(1e-12);
            matd Q = (matd)ws->qr->matrixQ();
            matd RT = (matd)ws->qr->matrixR();
            matd R = RT.leftCols(ws->qr->rank());
            matd T = RT.rightCols(A.cols() - ws->qr->rank());
            matd ZZ = matd::Zero(A.cols(), A.cols() - ws->qr->rank());
            ZZ.topRows(ws->qr->rank()) = -1 * R.inverse() * T;
            ZZ.bottomRows(A.cols() - ws->qr->rank()).setIdentity();
            ZZ = ws->qr->colsPermutation() * ZZ;
            cout << "AZqr " << (A * ZZ).norm() << endl;
        }
#endif
#if TIMEMEASUREMENTS
        turnback::timer tloop;
        turnback::timer t0;
        turnback::timer t1;
#endif
#if TIMEMEASUREMENTS
        t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::initial fac"));
        if (opt.verbose >= CONV) t1.stop("initial fac");
#endif

#if TIMEMEASUREMENTS
        t1 = turnback::timer();
#endif
        // find the order, b, pivotcols
        if (!bInitialized || !eqProj) getOB(A, nrEcs);
        if (opt.verbose >= BASIC) printOB(A);
#if TIMEMEASUREMENTS
        t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::find b"));
        if (opt.verbose >= CONV) t1.stop("find b");
#endif

        ws->chosenCols.head(cols).setZero();
        dimNS = 0;
        if (rankZ > 0)
        {
            // initial
#if TIMEMEASUREMENTS
            t1 = turnback::timer();
#endif
            bool fatalError = false;
            bool finished = false;
#if MULTITHREADING
#pragma omp parallel
#endif // MULTITHREADING
            {
                timer timerthr;
                // Getting thread number
#if MULTITHREADING
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif // MULTITHREADING
                if (tid < nrThreadsUsed)
                {
                    while (!finished)
                    {
                        // choose a subset
                        int curSubset = -1;
#if MULTITHREADING
#pragma omp critical
#endif // MULTITHREADING
                        {
                            int sb;
                            if (opt.pivotM == 3)
                            {
                                sb = nrSubsets-1;
                                for (; sb >= 0; sb--)
                                {
                                    if (ws->subsetActivity[sb] == 0)
                                    {
                                        curSubset = sb;
                                        ws->subsetActivity[sb] = 1;
                                        if (opt.verbose >= BASIC) cout << "thread-" << tid << " chose subset " << sb << endl;
                                        finished = true;
                                        break;
                                    }
                                }
                            }
                            else
                            {
                                sb = 0;
                                for (; sb < nrSubsets; sb++)
                                {
                                    if (ws->subsetActivity[sb] == 0)
                                    {
                                        curSubset = sb;
                                        ws->subsetActivity[sb] = 1;
                                        if (opt.verbose >= BASIC) cout << "thread-" << tid << " chose subset " << sb << endl;
                                        finished = true;
                                        break;
                                    }
                                }
                            }
                            finished = true;
                            for (sb = 0; sb < nrSubsets; sb++)
                            {
                                if (ws->subsetActivity[sb] == 0)
                                {
                                    finished = false;
                                    break;
                                }
                            }
                        }

                        unique_ptr<turnbackIndices> tbx; tbx = make_unique<turnbackIndices>();
                        initTBIndices(tbx, tid);
#if TIMEMEASUREMENTS
                        t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::initial fac j0"));
                        if (opt.verbose >= CONV) t1.stop("initial fac j0");
                        tloop = turnback::timer();
#endif
                        int testAzCtr = 0;
                        if (curSubset > -1)
                        {
                        for (int jj = 0; jj < ws->jsPerSubset[curSubset].size(); jj++)
                        {
                            if ((opt.pivotM == 1 || opt.pivotM == 3) && jj > 0) continue;
#if TIMEMEASUREMENTS
                            t1 = turnback::timer();
#endif
                            int j = ws->jsPerSubset[curSubset][jj];
                            tbx->j = j;
                            tbx->piv = ws->pivotCols[ws->pivotcolssort[j]];
                            tbx->bcur = ws->b(ws->bsort[j]);
                            if (order==0) tbx->bnext = cols;
                            if (order==1) tbx->bnext = -1;
                            if (j+1 < rankZ) tbx->bnext = ws->b(ws->bsort[j+1]); // holds the actual position of the first column b(ws->bsort[j]) in the lu decomposition, usually jprep + 1
                            if (opt.verbose >= CONV && tid == opt.tid) cout << "j " << j << " ws->pivotcolssort[j] " << ws->pivotcolssort[j] << " piv " << tbx->piv << endl;
                            if (ws->subsetNr[tbx->piv] != tbx->subsetNr) tbx->anew = true;
#if TIMEMEASUREMENTS
                            t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::getMAR"));
                            if (opt.verbose >= CONV) t1.stop("getMAR");
                            t1 = turnback::timer();
#endif

                            if (opt.verbose >= CONV && tid == opt.tid) printIter(tid, j, tbx);

                            if (opt.verbose >= CONV && tid == opt.tid) cout << "anew " << tbx->anew << " testAzCtr " << testAzCtr << endl;
                            if (tbx->anew)
                            {
                                if (testAzCtr == 0)
                                {
                                    tbx->subsetNr = curSubset; //ws->subsetNr[tbx->piv];
                                    tbx->leftI = ws->subsetRg.col(curSubset)(0);
                                    tbx->rightI = min(cols-1, ws->subsetRg.col(curSubset)(1));
                                    if (!xeidIdx){
                                        // if (opt.colUpdates) // NI
                                        compOverlap(tid, tbx, A);
                                        tbx->rightI = min(cols-1, max(tbx->rightI, tbx->leftI + tbx->colRg + ws->subsetCard(curSubset)));
                                    }
                                    // ws->subsetRg.row(1).segment(tbx->subsetNr, cols-tbx->subsetNr).array() = tbx->rightI;
                                    if (tid == opt.tid)
                                    {
                                        if (opt.verbose >= BASIC) cout << "thread " << tid << " subsetNr " << tbx->subsetNr << " curSubset " << curSubset << endl;
                                        if (tbx->subsetNr != curSubset) throw;
                                    }
                                    tbx->leftsubsetnr = curSubset-1;
                                    tbx->rightsubsetnr = curSubset;
                                }
                                if (opt.facM == 0)
                                {
                                    initialFac_XEID(tid, ws->lu_omp[tid], A, tbx);
                                }
                                else
                                {
                                    initialFac_XEID(tid, ws->qr_omp[tid], A, tbx);
                                }
                                tbx->anew = false;
                            }
                            else if (opt.colUpdates)
                            {
                                remCol(tid, ws->lu_omp[tid], tbx->piv, A); // remove the pivot column in case it is in the decomposition
                                int jrepsave = tbx->jrep;
                                if (order == 0)
                                {
                                    // if there are remaining columns on the left from the last b we need to replace them by zero
                                    // note that there can be columns left of tbx->bprev due to getIdx
                                    tbx->jrep = tbx->bprev;
                                    // tbx->jrep = 0; // tbx->bprev;
                                    remCols(tid, ws->lu_omp[tid], tbx, "right", A);
                                }
                                else if (order == 1)
                                {
                                    // if there are remaining columns on the right from the last b we need to replace them by zero
                                    // note that there can be columns right of tbx->bprev due to getIdx
                                    tbx->jrep = tbx->bprev;
                                    // tbx->jrep = cols-1; // tbx->bprev;
                                    remCols(tid, ws->lu_omp[tid], tbx, "left", A);
                                }
                                tbx->jrep = jrepsave;
#if TIMEMEASUREMENTS
                                t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::remcols on the left"));
                                if (opt.verbose >= CONV) t1.stop("remcols on the left");
                                t1 = turnback::timer();
#endif
                                // add columns until rank condition is met (range = rank+1)
                                stage1(tid, ws->lu_omp[tid], A, tbx);
#if TIMEMEASUREMENTS
                                t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::stage1"));
                                if (opt.verbose >= CONV) t1.stop("stage1");
                                t1 = turnback::timer();
#endif
                            }
#if SAFEGUARD
                            // check range of A(tbx->bcur:tbx->jrep)
                            matd Ablock = matd::Zero(rows, cols);
                            int ccctr = 0;
                            for (int cc=0; cc < cols; cc++)
                            {
                                if (_A.col(cc).norm() > 0) Ablock.col(ccctr) = _A.col(cc);
                                ccctr++;
                            }
                            Eigen::ColPivHouseholderQR<matd> qrd(Ablock);
                            if (qrd.rank() != lu->rank())
                            {
                                if (opt.verbose >= VAR)
                                {
                                    cout << "A\n" << (matd)A << endl;
                                    cout << "_A\n" << (matd)_A << endl;
                                    cout << "Ablock\n" << Ablock.leftCols(ccctr) << endl;
                                    cout << "QR\n" << (matd) qrd.matrixR() << endl;
                                }
                                cout << "turnback::computeNS: iteration " << j << " qr rank of A " << qrd.rank() << " compared with lu rank " << lu->rank() << "; this means that most likely the updated LU decomposition does not represent the actual matrix A" << endl;
                                // throw;
                            }
                            else if (opt.verbose >= VAR) cout << "QR and LU of A(:,:) have same rank " << qrd.rank() << " - " << lu->rank() << endl;
#endif
#if TIMEMEASUREMENTS
                            turnback::timer tm1;
#endif
                            int status = 1;
                            if (testAzCtr < 5)
                            {
                                if (opt.facM == 0) status = computez(tid, ws->lu_omp[tid], j, tbx, A, testAzCtr);
                                else status = computez(tid, ws->qr_omp[tid], j, tbx, A, testAzCtr);
                            }
                            else
                            {
                                if (opt.facM == 0) status = computez(tid, ws->lu_omp[tid], j, tbx, A);
                                else status = computez(tid, ws->qr_omp[tid], j, tbx, A);
                            }
                            if (tbx->fatalError)
                            {
                                // exit and return the standard LU based Z
                                cout << "[thread-" << tid << "] turnbackLU::computeNS: FATAL ERROR in turnbackLU, most likely based on instability of LU, return standard (dense) NS" << endl;
                                fatalError = true;
                                break;
                            }
                            if (status == 0)
                            {
                                testAzCtr++;
                                tbx->anew = true;
                                jj--;
                                if (xeidIdx) ws->subsetRg.col(j)(1) += 0;
                                tbx->colRg = abs(tbx->jrep - tbx->bcur);
                                if (opt.verbose >= CONV) cout << "[thread-" << tid << "] turnbackLU: numerical issue in computez, repeat this iteration " << tbx->j << endl;
                                // continue;
                            }
                            else
                            {
                                if (dimNS == rankZ)
                                {
                                    cout << "NS identified with dimNS " << dimNS << " and nsdim " << rankZ << "!" << endl;
                                    finished = true;
                                    break;
                                }

                                testAzCtr = 0;
                                // success 1
                                // failure -1
                            }
#if SAFEGUARD
                            else
                            {
                                for (int zc=0; zc < zcollected.size(); zc++)
                                {
                                    if ((z.head(cols) - zcollected[zc]).norm() < 1e-5 || (z.head(cols) + zcollected[zc]).norm() < 1e-5)
                                    {
                                        cout << "turnbackLU::computez: ERROR, z already exists at iter " << zc << ", z\n" << z.head(cols).transpose() << endl;
                                        cout << "turnbackLU::computez: ERROR, z already exists at iter " << zc << ", zc\n" << zcollected[zc].transpose() << endl;
                                        cout << "turnbackLU::computez: chosen pivot columns \n" << ws->chosenCols_omp[tid].head(cols).transpose() << endl;
                                        throw;
                                    }
                                }
                                zcollected.push_back(z.head(cols));
                            }
#endif
#if TIMEMEASUREMENTS
                            t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::computez"));
                            if (opt.verbose >= CONV) t1.stop("computez");
#endif
                            tbx->bprev = tbx->bcur;
                        }
                        }
#if MULTITHREADING
#pragma omp critical
#endif // MULTITHREADING
                        {
                        if (opt.verbose >= BASIC) cout << "thread " << tid << " finished subset " << curSubset << endl;
                        }
                    }
                }
                timerthr.stop();
                if (opt.verbose >= BASIC) cout << "[thread-" << tid << "] finished in " << timerthr.time << endl;
            }
            if (fatalError)
            {
                cout << "turnback: FATAL ERROR: return standard LU-NS" << endl;
                ws->lu0->reset();
                if (Zio) ws->lu0->computeNS(A, Zio, true);
                else ws->lu0->computeNS(A, ws->_Z);
                rank0 = ws->lu0->rank();
                return;
            }
#if TIMEMEASUREMENTS
            tloop.stop(); times.push_back(turnback::time(tloop.time, "turnbackLU::jloop"));
            t1 = turnback::timer();
#endif
            if (Zio)
            {
                Zio->conservativeResize(cols, rankZ);
                ws->thz.setFromTriplets(*Zio);
            }
            else
            {
                ws->_Z = make_unique<mat>(cols, rankZ);
                ws->thz.setFromTriplets(*ws->_Z);
            }
#if TIMEMEASUREMENTS
            t1.stop(); times.push_back(turnback::time(t1.time, "turnbackLU::constructZ"));
            if (opt.verbose >= CONV) t1.stop("constructZ");
#endif 

#if SAFEGUARD
            if (opt.verbose >= NONE)
            {
                double AZerror = 0;
                if (Zio) AZerror = (A * *Zio).norm();
                else AZerror = (A * *ws->_Z).norm();
                if (AZerror > 1e-12)
                {
                    cout << "TurnbackLU::computeNS: NS corrupted with projection error " << AZerror << endl;
                    if (opt.verbose >= MAT)
                    {
                        cout << "A\n" << (matd)A << endl;
                        if (Zio) cout << "Z\n" << (matd)*Zio << endl;
                    }
                    // std::cin.get();
                    // throw;
                }
                else if (opt.verbose >= NONE) cout << "TurnbackLU: NS clean with error " << AZerror << endl;
                Eigen::ColPivHouseholderQR<matd> qrd((matd)*Zio);
                Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > qr;
                qr.compute(*Zio);
                if (qrd.rank() < min(Zio->rows(), Zio->cols()))
                {
                    cout << "Z dim " << Zio->rows() << " x " << Zio->cols() << endl;
                    cout << "turnbackLU::computeNS: Z is not full rank; detected rank " << qrd.rank() << ", assumed rank: " << min(Zio->rows(), Zio->cols()) << endl;
                    if (abs(qrd.rank() - min(Zio->rows(), Zio->cols())) > 5) 
                    {
                        if (opt.verbose >= NONE) cout << "Z\n" << (matd)*Zio << endl;
                        cout << "turnbackLU::computeNS: extreme rank difference qrd " << qrd.rank() << " qr " << qr.rank() << " Z dim " << Zio->rows() << " x " << Zio->cols() << endl;
                        cout << "chosenCols " << ws->chosenCols.head(cols).transpose() << endl;
                        cout << "R diag\n" << qrd.matrixR().leftCols(qrd.rank()).diagonal().transpose() << endl;
                        cout << "R diag\n" << qrd.matrixR().leftCols(Zio->cols()).diagonal().transpose() << endl;
                        std::cin.get();
                    }
                }
                else if (opt.verbose >= NONE) { cout << "turnbackLU::computeNS: Z is full rank; detected rank " << qrd.rank() << ", assumed rank: " << min(Zio->rows(), Zio->cols()) << endl; }
            }
#endif
        }
        else
        {
            if (Zio) Zio->resize(cols,0);
            else ws->_Z = make_unique<mat>(cols, 0);
        }
        if (Zio)
        {
            if (opt.verbose >= MAT) cout << "turnbackLU::computeNS: computed Z\n" << (matd)(*Zio) << endl;
            if (opt.verbose >= MAT) cout << "turnbackLU::computeNS: computed ZTZ\n" << (matd)(Zio->transpose() * *Zio) << endl;
            if (opt.verbose >= CONV) cout << "turnbackLU: for the matrix of " << rows << " x " << cols << " with a null-space of column space " << rankZ << ", " << nrFac + nrFac_bin << " factorizations were conducted, " << nrRemCols << " columns were removed and " << nrAddCols << " columns were added" << endl;
        }
        else
        {
            if (opt.verbose >= MAT) cout << "turnbackLU::computeNS: computed Z\n" << (matd)(Z()) << endl;
            if (opt.verbose >= MAT) cout << "turnbackLU::computeNS: computed ZTZ\n" << (matd)(ws->_Z->transpose() * *ws->_Z) << endl;
            if (opt.verbose >= CONV) cout << "turnbackLU: for the matrix of " << rows << " x " << cols << " with a null-space of column space " << rankZ << ", " << nrFac + nrFac_bin << " factorizations were conducted, " << nrRemCols << " columns were removed and " << nrAddCols << " columns were added" << endl;
        }
#if TIMEMEASUREMENTS
        tloop.stop(); times.push_back(turnback::time(tloop.time, "turnbackLU::computeNS: loop time"));
#endif
    }

    void turnbackLU::getOB(const mat& A, const int nrEcs)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        for (int tid = 0; tid < ws->JPerThread.size(); tid++)
        {
            ws->JPerThread[tid].clear();
        }
        order = 0;
        unique_ptr<turnbackIndices> tbx0; tbx0 = make_unique<turnbackIndices>();
        if (xeidIdx && nrEcs == 0)
        {
            // cheap assembly for XEID
            findBX(A);
            ws->_Q = make_unique<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> >(cols);
            ws->_Q->setIdentity();
        }
        else
        {
            initialFac(0, ws->lu0, A, tbx0);
            rank0 = ws->lu0->rank();
            rankZ = cols - rank0;
            findB();
            fixb();
        }
        bInitialized = true;
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::getOB"));
        if (opt.verbose >= CONV) tm1.stop("getOB");
#endif
    }

    void turnbackLU::findBX(const mat& A)
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif

        nrSubsets = xeidIdx->T;
        rank0 = min(xeidIdx->T*xeidIdx->nx, xeidIdx->T*(xeidIdx->nu+xeidIdx->nx));
        rankZ = cols - rank0;
        int rn = rankZ;
        int rnThread = max(1, (int)round(double(rankZ) / (double)nrThreads));
        int a = rn - (3*xeidIdx->nx+2*xeidIdx->nf);
        int b = 3*xeidIdx->nx+2*xeidIdx->nf; // the number of threads for the last cheaper bit
        int cTh = (int)(1. * (double)a + (double)b) / (double)nrThreads; // the number of threads for the last cheaper bit // choose value < 1 for more columns in last threads
        // cout << "rn " << rn << " rnThread " << rnThread << endl;
        // cout << "cols " << cols << " T " << T << " rank0 " << rank0 << " rn " << rn << endl;
        if (opt.verbose >= BASIC) cout << "=== turnbackLU::FINDBX with xeidIdx->nx " << xeidIdx->nx << " nt " << xeidIdx->nt << " xeidIdx->nf " << xeidIdx->nf << " columns per thread " << cTh << " ===" << endl;

        for (int i = 0 ; i < rankZ; i++)
        {
            ws->bsort[i] = i;
            ws->pivotcolssort[i] = i;
        }
        ws->pivotCols_.head(cols).setZero();
        // we need to find the right pivot columns in case of underactuation
        // conduct rank revealing decomposition of Mtua and D3ua
        int nrPivColsSubset = xeidIdx->nf + xeidIdx->nx/2-xeidIdx->nua; // the number of pivot columns per subset
        int NegPivColsSubset = xeidIdx->nx/2 - xeidIdx->nua; // the number of underrank of the subset due to underactuation
        int multiplier = 2 * ceil((double)abs(xeidIdx->nua) / ((double)(xeidIdx->nx/2 - xeidIdx->nua)));
        if (xeidIdx->nua == 0) multiplier = 0;
        if (opt.verbose >= BASIC)
        {
            cout << "T " << xeidIdx->T << endl;
            cout << "nua " << xeidIdx->nua << endl;
            cout << "nrPivColsSubset " << nrPivColsSubset << endl;
            cout << "NegPivColsSubset " << NegPivColsSubset << endl;
            cout << "nua - nrPivColsSubset " << nrPivColsSubset - xeidIdx->nua << endl;
            cout << "(nua - nrPivColsSubset) / (double)nrPivColsSubset) " << (double)nrPivColsSubset / (double)abs(xeidIdx->nua - nrPivColsSubset) << endl;
            cout << "multiplier " << multiplier << endl;
        }

        // assemble pivot columns
        int pvc = 0;
        int bstart = 0;
        int boff = 0;
        int tel = 0;
        if (opt.pivotM == 0 || opt.pivotM == 1 || opt.pivotM == 3)
        {
            for (int t = 0; t < xeidIdx->T; t++)
            {
                boff = 0;
                ws->subsetRg.col(t)(0) = bstart;
                // forces
                tel += xeidIdx->nt;
                for (int pvs = 0; pvs < xeidIdx->nf; pvs++)
                {
                    ws->pivotCols[pvc] = tel + pvs;
                    ws->pivotCols_[tel + pvs] = 2; // these pivots carry no rank condition, can't be used in subsets
                    ws->subsetNr[ws->pivotCols[pvc]] = t;
                    boff++;
                    pvc++;
                }
                tel += xeidIdx->nf;
                // mass matrix
                tel += xeidIdx->nx/2+xeidIdx->nua;
                // mass matrix
                for (int pvs = 0; pvs < xeidIdx->nx/2-xeidIdx->nua; pvs++)
                {
                    ws->pivotCols[pvc] = tel + pvs;
                    ws->pivotCols_[ws->pivotCols[pvc]] = 1;
                    ws->subsetNr[ws->pivotCols[pvc]] = t;
                    boff++;
                    pvc++;
                }
                tel += xeidIdx->nu-xeidIdx->nf;
                boff = 
                    (2+multiplier)*xeidIdx->nx + (3+multiplier)*(xeidIdx->nt + xeidIdx->nf) - 1;
                ws->subsetRg.col(t)(1) = bstart + boff;
                bstart += xeidIdx->nx + xeidIdx->nu;
            }
        }
        else if (opt.pivotM == 2)
        {
            unique_ptr<turnbackIndices> tbx; tbx = make_unique<turnbackIndices>();
            for (int t = 0; t < xeidIdx->T; t++)
            {
                boff = 0;
                ws->subsetRg.col(t)(0) = bstart;
                bstart += xeidIdx->nx + xeidIdx->nu;
                tbx->bcur = t * (xeidIdx->nx+xeidIdx->nu);
                tbx->colRg = xeidIdx->nx+xeidIdx->nu;
                initialFac_rg(0, ws->lu_omp[0],  A, tbx);
                if (opt.verbose >= BASIC) cout << "pivots in range " << t*(xeidIdx->nx+xeidIdx->nu) << " to " << (t+1)*(xeidIdx->nx+xeidIdx->nu) << ": ";
                for (int c = xeidIdx->nx; c < cols; c++)
                {
                    if (t*(xeidIdx->nx+xeidIdx->nu) <= ws->lu_omp[0]->qq()[c]-1 && ws->lu_omp[0]->qq()[c]-1 < (t+1)*(xeidIdx->nx+xeidIdx->nu))
                    {
                        cout << ws->lu_omp[0]->qq()[c]-1 << " ";
                        ws->pivotCols[pvc] = ws->lu_omp[0]->qq()[c]-1;
                        ws->pivotCols_[ws->pivotCols[pvc]] = 1;
                        ws->subsetNr[ws->pivotCols[pvc]] = t;
                        boff++;
                        pvc++;
                    }
                }
                sort(ws->pivotcolssort.begin(), ws->pivotcolssort.begin() + rankZ,
                        [&](const int& i1, const int& i2) {
                        return (ws->pivotCols.data()[i1] < ws->pivotCols.data()[i2]);
                        }
                    );
                cout << endl;
                boff = 
                    (2+multiplier)*xeidIdx->nx + (3+multiplier)*(xeidIdx->nt + xeidIdx->nf) - 1;
                ws->subsetRg.col(t)(1) = bstart + boff;
            }
        }

        pvc = 0;
        boff = 0;
        int seg = 0;

        for (int t=0; t < xeidIdx->T; t++)
        {
            for (int cc = 0; cc < xeidIdx->nu; cc++)
            {
                ws->b[pvc] = boff;
                ws->JPerThread[seg].push_back(pvc);
                ws->jsPerSubset[t].push_back(pvc);
                pvc++;
            }
            boff += xeidIdx->nx + xeidIdx->nu;
            // in order to not cut subsets
            if (pvc > 0 && pvc - seg * cTh > cTh) seg = min(nrThreads-1, seg+1);
        }
        if (ws->JPerThread[seg].size() > 0) nrThreadsUsed = seg+1;
        else nrThreadsUsed = seg;
        ws->b[pvc] = boff+1;

        if (opt.verbose >= VAR)
        {
            cout << "turnbackLU::findBX: order " << order << endl;
            cout << "turnbackLU::findBX: tlu b " << ws->b.head(rankZ).transpose() << endl;
            cout << "turnbackLU::findBX: tlu b sorted: ";
            for (int i = 0 ; i < rankZ; i++)
            {
                cout << ws->b[ws->bsort[i]] << " ";
            }
            cout << endl;
            cout << "turnbackLU::findBX: ws->pivotCols " << ws->pivotCols.head(rankZ).transpose() << endl;
            cout << "turnbackLU::findBX: ws->pivotCols sorted: ";
            for (int i = 0 ; i < rankZ; i++)
            {
                cout << ws->pivotCols[ws->pivotcolssort[i]] << " ";
            }
            cout << endl;
        }
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::findb"));
        if (opt.verbose >= CONV) tm1.stop("findb");
#endif
        // for extra variables
        if (cols - xeidIdx->T*(xeidIdx->nx+xeidIdx->nu) > 0)
        {
            if (opt.verbose >= BASIC) cout << "turnback::findBX: there are " << cols - xeidIdx->T*(xeidIdx->nx+xeidIdx->nu) << " extra variables, pad with one vectors" << endl;
            for (int v = 0; v < cols - xeidIdx->T*(xeidIdx->nx+xeidIdx->nu); v++)
            ws->thz.setTriplet(cols-1-v, rankZ-1-v, 1);
        }
    }

    void turnbackLU::findB()
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        for (int i = 0 ; i < rankZ; i++) {
            ws->bsort[i] = i;
            ws->pivotcolssort[i] = i;
        }
        if (order == 0)
        {
            // there might be ones in U2 that are better pivots
            // go through U2 and look for 1's that are unique in a row (for order 1, this means there is no other value in the same row left of the entry)
            // b indicates first nnz in each column of Z
            ws->b.head(rankZ).setConstant(cols + 10000);
            for (int j = 0; j < rankZ; j++)
            {
                // handle the I part of [U2 \\ I]
                // to get privot columns
                // range columns
                if (ws->lu0->Q().indices()[j+rank0] < ws->b(j)) ws->b(j) = ws->lu0->Q().indices()[j+rank0];
                // pivot columns
                ws->pivotCols(j) = ws->lu0->Q().indices()[j+rank0];

                // set b accordinly to the identified pivot column
                for (int jj = 0; jj < rank0; jj++)
                {
                    for (mat::InnerIterator jt(ws->lu0->U(),j+rank0); jt; ++jt)
                    {
                        // go through the rows of U2.col(j)
                        if (jj == jt.row()) if (ws->lu0->Q().indices()[jj] < ws->b(j)) ws->b(j) = ws->lu0->Q().indices()[jj];
                    }
                }
            }
            sort(ws->bsort.begin(), ws->bsort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->b.data()[i1] < ws->b.data()[i2]);
                    }
                );
            sort(ws->pivotcolssort.begin(), ws->pivotcolssort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->pivotCols.data()[i1] < ws->pivotCols.data()[i2]);
                    }
                );
        }
        else if (order == 1)
        {
            // FIXME: needs implementation
            // b indicates last nnz in each column of Z
            ws->b.head(rankZ).setZero();
            for (int j = 0; j < rankZ; j++)
            {
                // handle the I part of [U2 \\ I]
                for (int jj = rank0; jj < cols; jj++)
                {
                    // cout << "Q row " << it.row() << " col " << it.col() << " val " << it.value() << endl;
                    if (jj - rank0 == j) if (ws->lu0->Q().indices()[jj] > ws->b(j)) ws->b(j) = ws->lu0->Q().indices()[jj];
                    ws->pivotCols(jj-rank0) = ws->lu0->Q().indices()[jj];
                }
                for (int jj = 0; jj < rank0; jj++)
                {
                    for (mat::InnerIterator jt(ws->lu0->U(),j+rank0); jt; ++jt)
                    {
                        // go through the rows of U2.col(j)
                        if (jj == jt.row()) if (ws->lu0->Q().indices()[jj] > ws->b(j)) ws->b(j) = ws->lu0->Q().indices()[jj];
                    }
                }
            }
            sort(ws->bsort.begin(), ws->bsort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->b.data()[i1] > ws->b.data()[i2]);
                    }
                );
            sort(ws->pivotcolssort.begin(), ws->pivotcolssort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->pivotCols.data()[i1] > ws->pivotCols.data()[i2]);
                    }
                );
        }

        // assign ws->pivotCols_
        ws->pivotCols_.head(cols).setZero();
        for (int i=0; i < rankZ; i++)
            ws->pivotCols_[ws->pivotCols[i]] = 1; 

        if (opt.verbose >= VAR)
        {
            cout << "turnbackLU::findB: order " << order << endl;
            cout << "turnbackLU::findB: tlu b " << ws->b.head(rankZ).transpose() << endl;
            cout << "turnbackLU::findB: tlu b sorted: ";
            for (int i = 0 ; i < rankZ ; i++)
            {
                cout << ws->b[ws->bsort[i]] << " ";
            }
            cout << endl;
            cout << "turnbackLU::findB: ws->pivotCols " << ws->pivotCols.head(rankZ).transpose() << endl;
            cout << "turnbackLU::findB: ws->pivotCols sorted: ";
            for (int i = 0 ; i < rankZ ; i++)
            {
                cout << ws->pivotCols[ws->pivotcolssort[i]] << " ";
            }
            cout << endl;
            cout << "turnbackLU::findB: Q N " << Q().indices().tail(cols-rank()).transpose() << endl;
        }
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::findb"));
        if (opt.verbose >= CONV) tm1.stop("findb");
#endif
    }

    int turnbackLU::count2b()
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        int c2b = 0;
        for (int j = 1; j < rankZ; j++)
        {
            if (ws->b[ws->bsort[j]] == ws->b[ws->bsort[j-1]])
            {
                c2b++;
                if (opt.verbose >= VAR) cout << "turnbackLU::count2b: The column " << ws->b[ws->bsort[j]] << " appears both at " << j << " and " << j-1 << endl;
            }
        }
        return c2b;
    }

    void turnbackLU::fixb()
    {
#if TIMEMEASUREMENTS
        turnback::timer tm1;
#endif
        if (opt.verbose >= CONV) cout << "nrThreads " << nrThreads << " cols " << cols << " rank0 " << rank0 << " rankZ " << rankZ;
        int rnThread = max(1, (int)round(double(rankZ) / (double)nrThreads));
        if (opt.verbose >= CONV) cout << " rnThread " << rnThread << endl;
        int seg = 0;
        int pvc = 0;
        int boff = 0;
        if (order == 0)
        {
            sort(ws->bsort.begin(), ws->bsort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->b.data()[i1] < ws->b.data()[i2]);
                    }
                );
        }
        else if (order == 1)
        {
            boff = cols-1;
            sort(ws->bsort.begin(), ws->bsort.begin() + rankZ,
                    [&](const int& i1, const int& i2) {
                    return (ws->b.data()[i1] > ws->b.data()[i2]);
                    }
                );
        }
        // fixb
        int i = 0;
        int offset = 1;
        nrSubsets = 0;
        int bstart = 0;
        ws->subsetRg.col(nrSubsets)(0) = ws->b[ws->bsort[0]];
        // define the gap when the number of nodes is known (for example FEM or MPC (T) nodes)
        int gap = 1;
        int ctrGap = 0;
        int ctrSubsets = 0;
        while (true)
        {
            for (int i = 0; i < rankZ-1; i++)
            {
                if (ws->pivotCols[ws->pivotcolssort[i+1]] - ws->pivotCols[ws->pivotcolssort[i]] > gap)
                {
                    ctrSubsets++;
                }
            }
            // cout << "number subset " << ctrSubsets << " T+1 " << T+1 << " int(T+1/2) " << int(T+1/2) << endl;
            if (ctrSubsets < int(nodes+1/2))
            {
                gap--;
                break;
            }
            else if (ctrSubsets < nodes+1) break;
            ctrSubsets = 0;
            gap++;
        }
        if (opt.verbose >= CONV) cout << "turnbackLU::fixb: the gap is " << gap << endl;

        while (i < rankZ)
        {
            ws->subsetNr[ws->pivotCols[ws->pivotcolssort[i]]] = nrSubsets;
            ws->subsetCard[nrSubsets]++;
            ws->jsPerSubset[nrSubsets].push_back(i);
            if (i < rankZ-1)
            {
                if (ws->pivotCols[ws->pivotcolssort[i+1]] - ws->pivotCols[ws->pivotcolssort[i]] <= gap)
                {
                    ws->b[ws->bsort[i+1]] = ws->b[ws->bsort[i]];
                    offset++;
                }
                else
                {
                    offset = 1;
                    ws->subsetsPerThread[seg].push_back(nrSubsets);
                    if (i > 0 && (int)((double)i / double(rnThread)) > seg) seg = min(nrThreads-1, seg+1);
                    nrSubsets++;
                    ws->subsetRg.col(nrSubsets)(0) = ws->pivotCols[ws->pivotcolssort[i]];
                }
            }
            ws->JPerThread[seg].push_back(i);
            i++;
        }
        // set to zero
        ws->subsetRg.row(0).head(nrSubsets).array() -= ws->b[ws->bsort[0]];
        ws->b.head(rankZ).array() -= ws->b[ws->bsort[0]];
        nrSubsets++;

        if (ws->JPerThread[seg].size() > 0) nrThreadsUsed = seg+1;
        else nrThreadsUsed = seg;
#if TIMEMEASUREMENTS
        tm1.stop(); times.push_back(turnback::time(tm1.time, "turnbackLU::count2b"));
        if (opt.verbose >= CONV) tm1.stop("count2b");
#endif
    }

    void turnbackLU::initTBIndices(unique_ptr<turnbackIndices>& tbx, int tid)
    {
        if (tid+1 < nrThreadsUsed) tbx->jrep = ws->JPerThread[tid+1][0]; 
        else tbx->jrep = cols;
        if (order == 1) tbx->jrep = ws->JPerThread[tid][0]-1; // holds the current / last index in the lu decomposition where a column was replaced
        if (order == 0) tbx->jrep = 0;
        else tbx->jrep = cols;
        if (tid > 0) tbx->bprev = ws->b(ws->bsort[ws->JPerThread[tid-1][ws->JPerThread[tid-1].size()-1]]); // holds the actual position of the first column b(ws->bsort[j]) in the lu decomposition, usually jprep + 1
        else
        {
            if (order == 0) tbx->bprev = 0;
            else if (order == 1) tbx->bprev = cols-1;
        }
    }

    void turnbackLU::printOB(const mat& A)
    {
        cout << "turnbackLU::getOB: number of Threads used: " << nrThreadsUsed << endl;
        cout << "turnbackLU::getOB: matrix dimensions " << rows << " x " << cols << endl;
        // compare with qr
        ws->qr = make_shared<Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int> > >();
        ws->qr->compute(A);
        cout << "turnbackLU::getOB: the matrix has rank " << rank0 << "; its nullspace has rank " << rankZ << " (qr rank: " << ws->qr->rank() << ")" << endl;
        cout << "turnbackLU::getOB: order " << order << endl;
        cout << "turnbackLU::getOB: tlu b " << ws->b.head(rankZ).transpose() << endl;
        cout << "turnbackLU::getOB: b sorted      : ";
        for (int i = 0 ; i < rankZ ; i++) {
            cout << ws->b[ws->bsort[i]] << " ";
        }
        cout << endl;
        cout << "turnbackLU::getOB: pivCols sorted: ";
        for (int i = 0 ; i < rankZ ; i++) {
            cout << ws->pivotCols[ws->pivotcolssort[i]] << " ";
        }
        cout << endl;
        cout << "turnbackLU::getOB: pivsort: ";
        for (int i = 0 ; i < rankZ ; i++) {
            cout << ws->pivotcolssort[i] << " ";
        }
        cout << endl;
        cout << "turnbackLU::getOB: subsetNr: ";
        cout << ws->subsetNr.head(cols).transpose() << endl;
        cout << "turnbackLU::getOB: subsetRg:\n";
        cout << ws->subsetRg.leftCols(cols) << endl;
        cout << "turnbackLU::getOB: subsetCard\n";
        cout << ws->subsetCard.head(cols).transpose() << endl;
        cout << "j's per subset\n";
        for (int i = 0; i < nrSubsets; i++)
        {
            cout << "j's of subset " << i << ": ";
            for (int j : ws->jsPerSubset[i]) cout << j << " ";
            cout << endl;
        }
        for (int nth = 0; nth < nrThreadsUsed; nth++)
        {
            cout << "turnbackLU: " << ws->JPerThread[nth].size() << " j's of thread " << nth << ": ";
            for (int j : ws->JPerThread[nth]) cout << j << ", ";
            cout << endl;
        }
        cout << "subsetsPerThread\n";
        for (int i = 0; i < ws->subsetsPerThread.size(); i++)
        {
            cout << "Thread-" << i << ": ";
            for (int j : ws->subsetsPerThread[i]) cout << j << " ";
            cout << endl;
        }
    }

    void turnbackLU::printIter(int tid, int j, const unique_ptr<turnbackIndices>& tbx)
    {
        cout << "[thread-" << tid << "] ====== iteration " << j << " with";
        cout << " turnbackLU order " << order << " cols " << cols << " ws->bsort[j] " << ws->bsort[j] << " nrFac " << nrFac << " nrFac_bin " << nrFac_bin << " nrAddCols " << nrAddCols << " nrRemCols " << nrRemCols << endl;
        tbx->print();
    }

    void turnbackLU::printInfo(const mat& A)
    {
        mat ATA = A.transpose() * A;
        mat ZTZ = Z().transpose() * Z();
        cout << "turnbackLU: for the matrix with " << rows << " x " << cols << " (" << rows*cols << ") of rank " << rank0 << " and its NS with " << rows << " x " << rankZ << " (" << rows * rankZ << ") , turnbackLU handled overall " << nrAddCols + nrRemCols << " columns while added " <<  nrAddCols << " columns, removed " << nrRemCols << " columns with " << nrFac + nrFac_bin << " factorizations" << ", A has " << (double)A.nonZeros() << " nnz's and its density is  " << (double)A.nonZeros() / double(A.cols() * A.rows()) << "; the NS has " << (double)Z().nonZeros() << " nnz's and its density is " << (double)Z().nonZeros() / double(Z().cols() * Z().rows()) << ", for ATA we have " << (double)ATA.nonZeros() << " nnz's and its density is " << (double)ATA.nonZeros() / double(ATA.cols() * ATA.rows()) << ", for ZTZ we have " << (double)ZTZ.nonZeros() << " nnz's and its density is " << (double)ZTZ.nonZeros() / double(ZTZ.cols() * ZTZ.rows()) << endl; // FIXME: are these the real nnz?
    }

#if SAFEGUARD
    void turnbackLU::computeLUNS(const mat& A)
    {
        shared_ptr<turnbackIndices> tbx; tbx = make_shared<turnbackIndices>();
        initialFac(0, ws->lu0, A, tbx);
        rank0 = lu->rank();
        mat U1 = lu->U().block(0,0,lu->rank(),lu->rank());
        mat U2 = lu->U().block(0,lu->rank(),lu->rank(),rankZ);
        mat I(rankZ, rankZ); I.setIdentity();
        mat Zhard = mat(cols, rankZ);
        ws->thz.reset();
        U1.triangularView<Eigen::Upper>().solveInPlace(U2);
        ws->thz.getTriplets(-U2);
        ws->thz.getTriplets(I, lu->rank(), 0);
        Zhard.setZero();
        ws->thz.setFromTriplets(Zhard);
        lu->Q().applyThisOnTheLeft(Zhard);
        if (Zhard.cols() > 0) cout << "NS test " << (A * Zhard).norm() << endl;
        ws->thz.reset();
    }
#endif

    void turnbackLU::printTime()
    {
#if TIMEMEASUREMENTS
        double allTime = 0;
        vector<string> tagList;
        vec sumTime;
        for (time t : times)
        {
            bool found = false;
            for (string s : tagList)
            {
                if (s == t.tag)
                {
                    found = true;
                    break;
                }
            }
            if (!found) tagList.push_back(t.tag);
        }
        sumTime.resize(tagList.size()); sumTime.setZero();

        if (opt.verboseTime > NONE) cout << "- - - NIpmHLSP, level wise TIMEMEASUREMENTS\n";
        if (opt.verboseTime > NONE) cout << "--- level " << l << ":\n";
        for (time t : times)
        {
            if (opt.verboseTime > NONE) cout << t.tag << " in " << t.t << " [s]" << endl;
            // find tag in taglist
            int ctr = 0;
            for (string s : tagList)
            {
                if (t.tag == s)
                    sumTime[ctr] += t.t;
                ctr++;
            }
        }
        if (allTime == 0) allTime = sumTime.sum();
        int ctr = 0;
        cout << "= = = NIpmHLSP::hlsp: TIMEMEASUREMENTS of HLSP" << endl;
        for (string s : tagList)
        {
            cout << "---> " << sumTime[ctr] / allTime * 100 << " \% or " << sumTime[ctr] << " [s] " << " for " << s << ", overall       solver time " << allTime << " [s]" << endl;
            ctr++;
        }
#else
        cout << "TIMEMEASUREMENTS not enabled; enable in build: ccmake . : TIMEMEASUREMENTS : ON" << endl;
#endif
    }

}
