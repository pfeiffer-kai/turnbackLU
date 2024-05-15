## C++ implementation of turnback algorithm

This is a C++ implementation of the turnback algorithm based on the updateable LU factorization package LUSOL [1].

It efficiently reuses matrix subsets [2].

If the matrix represents dynamics integrated by the explicit Euler method (XEID), further simplifications are made [3].

The algorithm is parallelized using OpenMP.

## Install 
   * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
   * [eigen_clusol](https://github.com/pfeiffer-kai/eigen_clusol)
   * [clusol: modified version of lusol (https://github.com/nwh/lusol)](https://github.com/pfeiffer-kai/clusol)
   * [OpenMP](https://www.openmp.org/)
   
## Usage

```bash
mkdir build
cd build
make -j2 && sudo make install
cmake -DCOMPILE_TESTS=ON -DMULTITHREADING=ON ..
make -j2 && sudo make install
cd tests
export OMP_NUM_THREADS=2
./test
```

## Publications

<a id="1">[1]</a> https://web.stanford.edu/group/SOL/software/lusol/

<a id="1">[2]</a> Kai Pfeiffer, Abderrahmane Kheddar, "Efficient Lexicographic Optimization for Prioritized Robot Control and Planning", 2024, https://arxiv.org/abs/2403.09160

<a id="1">[3]</a> Kai Pfeiffer, Abderrahmane Kheddar, "Sequential hierarchical least-squares programming for prioritized non-linear optimal control", 2024, https://www.tandfonline.com/doi/abs/10.1080/10556788.2024.2307467

## Authors

- Kai Pfeiffer (<kaipfeifferrobotics@gmail.com>) 
