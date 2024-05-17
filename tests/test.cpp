// This file is part of TURNBACKLU.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>

#include <turnbackLU/typedefs.h>
#include <turnbackLU/turnbackLU.h>
#include <eigenlusol/eigenlusol.h>

typedef Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1> >::Index Index;

using namespace std;

int main()
{
    turnback::tripletHandler th(1e7);

    // create eigelusol instance
    cpplusol::options clopt;
    clopt.maxmn = 1e7;
    clopt.asize = 1e7;
    shared_ptr<cpplusol::eigenlusol> lu = make_shared<cpplusol::eigenlusol>(clopt);

    // create turnback instance
    turnback::options tbopt;
    tbopt.maxn = 1e4;
    tbopt.verbose = turnback::NONE;
    tbopt.pattern = true;
    turnback::turnbackLU tlu = turnback::turnbackLU(lu, tbopt);

    int nq = 22;
    int nv = 22;
    int nx = nq+nv;

    std::string filename = "turnback.txt";
    std::ofstream file(filename);
    for (int T = 1; T < 26; T++)
    {
        for (int FFoffset = 0; FFoffset <= nq; FFoffset++)
        {
            int nt = nv-FFoffset;
            int nf = 4*6;
            int nu = nt+nf;
            int nxu = nx + nu;
            int n = T * (nx + nu);
            int me = T * nx;
            std::tuple<int,int,int,int> pattern = {nx,nt,nf,T};
            // std::tuple<int,int,int,int> pattern = {0,0,0,T};

            cout << "\n=========== solve Turnback XEID with T " << T << " nx " << nx << " nu " << nu << " nt " << nt << " nf " << nf << endl; 

            vec x = vec::Ones(nx);
            vec u = vec::Zero(nu);

            mat ZTZ;
            mat Q = mat(nx, nx); Q.setIdentity();
            mat Qf = mat(nx, nx); Qf.setIdentity();
            mat R = mat(nu, nu); R.setIdentity();
            mat S = mat(nx, nu); 
            vec q = vec::Zero(nx);
            vec qf = vec::Zero(nx);
            vec r = vec::Zero(nu);
            mat Inx = mat(nx,nx); Inx.setIdentity();
            mat Inq = mat(nq,nq); Inq.setIdentity();
            mat Mnq = mat(nx,nx);
            matd Mnq_ = matd::Random(nq,nq);
            Mnq = Mnq_.sparseView();
            Mnq = Mnq.transpose() * Mnq;

            mat Inu = mat(nu,nu); Inu.setIdentity();

            // set random dynamics of form Ax + Bu
            matd Axed = matd::Zero(nq,nx);
            Axed.rightCols(nx/2) = (matd)Mnq;
            mat Axe = Axed.sparseView();

            matd Bued = MatrixXd::Zero(nx, nu); 
            Bued.block(nq+FFoffset,0,nt,nt).setIdentity();
            Bued.block(nq,nt,nq,nf).setRandom(); // contact force jacobian
            mat Bue = Bued.sparseView();

            // Ae
            mat Ae = mat(me, n);
            th.reset();
            for (int i = 0; i < T - 1; i++)
            {
                th.getTriplets(-Bue, i*nx, i*nxu);
                th.getTriplets(Inq, i*nx, i*nxu+nu);
                th.getTriplets(-Inq, (i+1)*nx, i*nxu+nu);
                th.getTriplets(-Inq, (i+1)*nx, i*nxu+nu+nq);
                matd D3d = matd::Random(nq,nq);
                mat D3 = D3d.sparseView();
                th.getTriplets(D3, i*nx+nq, i*nxu+nu+nq);
                if (i>0)
                {
                    th.getTriplets(D3, i*nx+nq, (i-1)*nxu+nu+nq);
                    matd D2d = matd::Random(nq,nq);
                    mat D2 = D2d.sparseView();
                    th.getTriplets(D2, i*nx+nq, (i-1)*nxu+nu);
                }
            }
            th.getTriplets(Inq, (T-1)*nx, ((T-1))*(nx+nu) + nu);
            th.getTriplets(-Bue, (T-1)*nx, ((T-1))*(nx+nu));
            th.getTriplets(Mnq, (T-1)*nx+nq, (T-1)*(nx+nu) + nu+nq);
            matd D3d = matd::Random(nq,nq);
            mat D3 = D3d.sparseView();
            if (T-1>0)
            {
                th.getTriplets(D3, (T-1)*nx+nq, (T-2)*nxu+nu+nq);
                matd D2d = matd::Random(nq,nq);
                mat D2 = D2d.sparseView();
                th.getTriplets(D2, (T-1)*nx+nq, (T-2)*nxu+nu);
            }
            th.setFromTriplets(Ae); 

            shared_ptr<mat> Z = make_shared<mat>(0,0);

            cout << "Mat size " << Ae.rows() << " X " << Ae.cols() << " nnz " << Ae.nonZeros() << endl;

            mat ATA = Ae.transpose() * Ae;
            Z = make_shared<mat>(0,0);

            tlu.bInitialized = false;
            turnback::timer t1;
            tlu.computeNS(Ae, Z, pattern, 0, true); // projected, i.e. findb based on lu
            t1.stop("turnbackLU projected");
            ZTZ = Z->transpose() * *Z;
            cout << "projection error tlu " << (Ae * *Z).norm() << " with nnz ZTZ " << ZTZ.nonZeros() << " and density " << (double)ZTZ.nonZeros() / double(ZTZ.cols() * ZTZ.rows()) << endl;
            cout << "Data: " << nx * nu * T << " & " << (double)ATA.nonZeros() << " & " << (double)ATA.nonZeros() / double(ATA.cols() * ATA.rows()) << " & " << tlu.nrAddCols << " & " <<  tlu.nrRemCols << " & " << ZTZ.nonZeros() << " & " << (double)ZTZ.nonZeros() / double(ZTZ.cols() * ZTZ.rows()) << " & " << endl;
            tlu.printInfo(Ae);
            tlu.printTime();
            if ((Ae * *Z).norm() > 1e-3) throw;

            file << T << "; " << FFoffset << "; " << nx << "; " << nu << "; " << nt << "; " << nf << "; " << t1.time << "; " << tlu.nrFac + tlu.nrFac_bin << "; " << ZTZ.nonZeros() << "; " << (double)ZTZ.nonZeros() / double(ZTZ.cols() * ZTZ.rows()) << endl; 
        }
    }
    file << endl;
    file.close();
}
