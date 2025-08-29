# caGMRES: Communication-Avoiding GMRES (MPI)

An MPI implementation of CA-GMRES with s-step Krylov generation, batched inner products, modified Gram–Schmidt, delayed re-orthogonalization, optional diagonal preconditioning, and a built-in communication profiler. Full algorithmic background and results are documented in the accompanying thesis. 

## Repository layout

```
.
├─ gmres                     # Executable (built by make)
├─ matrices/
│  ├─ dense6x6.mtx
│  ├─ dense6x6_rhs.mtx
│  ├─ strongScaling/…        # e.g., s3dkt3m2.mtx and *_rhs.mtx
│  └─ weakScaling/…          # e.g., P1.mtx and *_rhs.mtx
├─ ca_gmres.c
├─ ca_gmres.h
├─ ·······
├─strongScaling.sh
└─ weakScaling.s
```

Matrix paths and scripts are used by the test workflow. 

## Build(Running on TCHPC Callan)

```
make clean

module load mpi/2021.12

make
```


## Quick start

Sanity test on a tiny dense 6×6 system:

```
# Classical GMRES (s=1; default if s omitted)
mpirun -np 2 ./gmres matrices/dense6x6.mtx matrices/dense6x6_rhs.mtx 10 1 3

# CA-GMRES with s=2 or s=3
mpirun -np 2 ./gmres matrices/dense6x6.mtx matrices/dense6x6_rhs.mtx 10 1 3 2
mpirun -np 2 ./gmres matrices/dense6x6.mtx matrices/dense6x6_rhs.mtx 10 1 3 3
```

These runs also exercise the communication profiler.  

## Usage

```
mpirun -np <procs> ./gmres <matrix.mtx> <rhs.mtx> <m> <precond> <restarts> <s> <tol>
```

- `<m>`: restart length.
- `<precond>`: 0 or 1.
- `<restarts>`: max restarts.
- `<s>`: s-step block size. `1` = classical GMRES; `>1` = CA-GMRES.**8** recommended on most systems
- `<tol>`: convergence tolerance.
   If `<s>` is omitted the solver defaults to classical GMRES. 

Example on a large sparse matrix used in scaling tests:

```
# s3dkt3m2 strong-scaling case
mpirun -np 20 ./gmres matrices/strongScaling/s3dkt3m2.mtx \
  matrices/strongScaling/s3dkt3m2_rhs.mtx 30 1 3 8 1e-6
```



## Reproducing experiments

### Strong scaling

Runs classical GMRES (s=1) and CA-GMRES (s=8) across multiple process counts. The script repeats each configuration five times and reports averages, comm time, comm%, and inner-product counts.

```
chmod +x strongScaling.sh
./strongScaling.sh
```
### Strong scaling

Runs classical GMRES (s=1) and CA-GMRES (s=8) with proportionally scaled problem sizes. The script maintains constant work per processor, repeats each configuration ten times and reports averages, comm time, comm%, inner-product counts, and weak scaling efficiency.

```
chmod +x weakScaling
./weakScaling.sh
```
Outputs include per-np runs and a consolidated results table.  


Example invocations emitted by the script include:

```
# 1 proc
mpirun -np 1  ./gmres matrices/weakScaling/P1.mtx  matrices/weakScaling/P1_rhs.mtx  500 0 100 1
mpirun -np 1  ./gmres matrices/weakScaling/P1.mtx  matrices/weakScaling/P1_rhs.mtx  500 0 100 8

# 16 procs
mpirun -np 16 ./gmres matrices/weakScaling/P16.mtx  matrices/weakScaling/P16_rhs.mtx 500 0 100 1
mpirun -np 16 ./gmres matrices/weakScaling/P16.mtx  matrices/weakScaling/P16_rhs.mtx 500 0 100 8
```

The script prints per-matrix summaries and a final table.   

## Output and profiler

Typical solver output includes iteration info plus a communication profile with totals, inner-product counts, and averages. Example excerpts for dense6x6 and s3dkt3m2 appear in the test log and in the script summaries.  

## Results highlights

From the thesis evaluation: CA-GMRES reduces global synchronizations by 92.6% and achieved 12.0×–38.2× speedups on 1–20 processes while retaining stability comparable to classical GMRES. The algorithm becomes bandwidth-bound rather than latency-bound due to batched inner products. 

