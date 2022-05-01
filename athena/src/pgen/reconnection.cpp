//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orszag_tang.cpp
//! \brief Problem generator for Orszag-Tang vortex problem.
//!
//! REFERENCE: For example, see: G. Toth,  "The div(B)=0 constraint in shock capturing
//!   MHD codes", JCP, 161, 605 (2000)
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

void MyBoundaryInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void MyBoundaryOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void LocalizedDiffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
     const AthenaArray<Real> &bmag, const int is, const int ie, const int js,
                   const int je, const int ks, const int ke) ;

namespace {
Real d0, beta0, eta0, w_eta;
} // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  d0     = pin->GetReal("problem", "d0");
  beta0  = pin->GetReal("problem", "beta0");
  eta0   = pin->GetReal("problem", "eta0");
  w_eta  = pin->GetReal("problem", "w_eta");

  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, MyBoundaryInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, MyBoundaryOuterX2);

  EnrollFieldDiffusivity(LocalizedDiffusivity); // User-defined diffusivity
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Orszag-Tang test.  The initial conditions are
//  constructed assuming the domain extends over [-0.5x0.5, -0.5x0.5], so that exact
//  symmetry can be enforced across x=0 and y=0.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real gm = peos->GetGamma();

  Real p0 = 1.0/gm;
  Real B0 = sqrt(2.0*p0/beta0);

  // Initialize density, momentum, face-centered fields
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = d0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }

  // initialize interface B
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
        pfield->b.x1f(k,j,i) = 0.0;
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = -B0 * tanh(2.0*pcoord->x1v(i));
      }
    }
  }
  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = B0/cosh(2.0*pcoord->x1v(i));
      }
    }
  }

  // initialize total energy
  if (NON_BAROTROPIC_EOS) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) =
              p0/gm1 +
              0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                   SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                   SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))) + (0.5)*
              (SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
               + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  return;
}



void LocalizedDiffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, 
      const AthenaArray<Real> &w,
     const AthenaArray<Real> &bmag, const int is, const int ie, const int js,
                   const int je, const int ks, const int ke)
{
  Coordinates *pcoord = pmb->pcoord;
  Real h = pmb->pmy_mesh->mesh_size.x2max;
  Real w_eta2_i = 1.0/SQR(w_eta);
  
  for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
#pragma omp simd
        for(int i=is; i<=ie; i++){
          Real &etaB = pfdif->etaB(FieldDiffusion::DiffProcess::ohmic,k,j,i);
          Real x = pcoord->x1v(i);
          Real r2 = x*x + SQR(y-h);
          Real tmp = r2*w_eta2_i;
          etaB = eta0 * exp(-tmp);
        }
      }
  }

  return;
}


void MyBoundaryInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,jl-j,i) = -prim(IVY,k,jl+j-1,i);  // reflect 2-velocity
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,jl-j,i) = prim(n,k,jl+j-1,i);
          }
        }
      }
    }
  }
  
  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(jl-j),i) =  -b.x1f(k,(jl+j-1),i);
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(jl-j),i) = b.x2f(k,(jl+j  ),i);
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(jl-j),i) =  b.x3f(k,(jl+j-1),i);
        }
      }
    }
  }
  return;
  }


void MyBoundaryOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,ju+j,i) = -prim(IVY,k,ju-j+1,i);  // reflect 2-velocity
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,ju+j,i) = prim(n,k,ju-j+1,i);
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(ju+j  ),i) =  -b.x1f(k,(ju-j+1),i);
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(ju+j+1),i) = b.x2f(k,(ju-j+1),i);
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(ju+j  ),i) =  b.x3f(k,(ju-j+1),i);
        }
      }
    }
  }

  return;
}
