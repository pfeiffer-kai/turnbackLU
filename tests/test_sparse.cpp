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
    tbopt.pattern = false;
    turnback::turnbackLU tlu = turnback::turnbackLU(lu, tbopt);

    std::tuple<int,int,int,int> pattern = {0,0,0,1};
    // Ae
    mat Ae = mat(12, 18);
    for (int i = 0; i<9; i++)
    {
        Ae.coeffRef(i, 2*i) = 1;
        Ae.coeffRef(i, 2*i+1) = 1;
        Ae.coeffRef(i+3, 2*i) = -1;
        Ae.coeffRef(i+3, 2*i+1) = -1;
    }
    Ae.makeCompressed();

    shared_ptr<mat> Z = make_shared<mat>(0,0);

    cout << "Mat size " << Ae.rows() << " X " << Ae.cols() << " nnz " << Ae.nonZeros() << endl;

    mat ATA = Ae.transpose() * Ae;
    Z = make_shared<mat>(0,0);

    tlu.bInitialized = false;
    turnback::timer t1;
    tlu.computeNS(Ae, Z, pattern, 0, true); // projected, i.e. findb based on lu
    t1.stop("turnbackLU projected");
    mat ZTZ = Z->transpose() * *Z;
    cout << "projection error tlu " << (Ae * *Z).norm() << " with nnz ZTZ " << ZTZ.nonZeros() << " and density " << (double)ZTZ.nonZeros() / double(ZTZ.cols() * ZTZ.rows()) << endl;
    tlu.printInfo(Ae);
    tlu.printTime();
    if ((Ae * *Z).norm() > 1e-3) throw;

}
