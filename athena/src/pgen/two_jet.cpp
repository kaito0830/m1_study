//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file jet.cpp
//! \brief Sets up a nonrelativistic jet introduced through L-x1 boundary (left edge)
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// BCs on L-x1 (left edge) of grid with jet inflow conditions
void JetInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void JetOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

namespace {
// Make radius of jet and jet variables global so they can be accessed by BC functions
// Real r_amb,
Real d_amb, p_amb, vx_amb, vy_amb, vz_amb, bx_amb, by_amb, bz_amb;
Real r_jet_L, d_jet_L, p_jet_L, vx_jet_L, vy_jet_L, vz_jet_L, bx_jet_L, by_jet_L, bz_jet_L;
Real r_jet_R, d_jet_R, p_jet_R, vx_jet_R, vy_jet_R, vz_jet_R, bx_jet_R, by_jet_R, bz_jet_R;
Real gm1, x2_0, x3_0;
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize global variables
  d_amb  = pin->GetReal("problem", "d");
  p_amb  = pin->GetReal("problem", "p");
  vx_amb = pin->GetReal("problem", "vx");
  vy_amb = pin->GetReal("problem", "vy");
  vz_amb = pin->GetReal("problem", "vz");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx_amb = pin->GetReal("problem", "bx");
    by_amb = pin->GetReal("problem", "by");
    bz_amb = pin->GetReal("problem", "bz");
  }
  d_jet_L  = pin->GetReal("problem", "djet_L");
  p_jet_L  = pin->GetReal("problem", "pjet_L");
  vx_jet_L = pin->GetReal("problem", "vxjet_L");
  vy_jet_L = pin->GetReal("problem", "vyjet_L");
  vz_jet_L = pin->GetReal("problem", "vzjet_L");

  d_jet_R  = pin->GetReal("problem", "djet_R");
  p_jet_R  = pin->GetReal("problem", "pjet_R");
  vx_jet_R = pin->GetReal("problem", "vxjet_R");
  vy_jet_R = pin->GetReal("problem", "vyjet_R");
  vz_jet_R = pin->GetReal("problem", "vzjet_R");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx_jet_L = pin->GetReal("problem", "bxjet_L");
    by_jet_L = pin->GetReal("problem", "byjet_L");
    bz_jet_L = pin->GetReal("problem", "bzjet_L");

    bx_jet_R = pin->GetReal("problem", "bxjet_R");
    by_jet_R = pin->GetReal("problem", "byjet_R");
    bz_jet_R = pin->GetReal("problem", "bzjet_R");
  }
  r_jet_L = pin->GetReal("problem", "rjet_L");
  r_jet_R = pin->GetReal("problem", "rjet_R");

  x2_0 = 0.5*(mesh_size.x2max + mesh_size.x2min);
  x3_0 = 0.5*(mesh_size.x3max + mesh_size.x3min);

  // enroll boundary value function pointers
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, JetInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, JetOuterX1);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Jet problem
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gm1 = peos->GetGamma() - 1.0;

  // initialize conserved variables
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN,k,j,i) = d_amb;
        phydro->u(IM1,k,j,i) = d_amb*vx_amb;
        phydro->u(IM2,k,j,i) = d_amb*vy_amb;
        phydro->u(IM3,k,j,i) = d_amb*vz_amb;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = p_amb/gm1
                                 + 0.5*d_amb*(SQR(vx_amb)+SQR(vy_amb)+SQR(vz_amb));
        }
      }
    }
  }

  // initialize interface B
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = bx_amb;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x2f(k,j,i) = by_amb;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x3f(k,j,i) = bz_amb;
        }
      }
    }
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) += 0.5*(SQR(bx_amb) + SQR(by_amb) + SQR(bz_amb));
          }
        }
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void JetInnerX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

void JetInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // set primitive variables in inlet ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
        if (rad <= r_jet_L) {
          prim(IDN,k,j,il-i) = d_jet_L;
          prim(IVX,k,j,il-i) = vx_jet_L;
          prim(IVY,k,j,il-i) = vy_jet_L;
          prim(IVZ,k,j,il-i) = vz_jet_L;
          prim(IPR,k,j,il-i) = p_jet_L;
        } else {
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
          prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
          prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
          prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
          prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
        }
      }
    }
  }

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_L) {
            b.x1f(k,j,il-i) = bx_jet_L;
          } else {
            b.x1f(k,j,il-i) = b.x1f(k,j,il);
          }
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_L) {
            b.x2f(k,j,il-i) = by_jet_L;
          } else {
            b.x2f(k,j,il-i) = b.x2f(k,j,il);
          }
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_L) {
            b.x3f(k,j,il-i) = bz_jet_L;
          } else {
            b.x3f(k,j,il-i) = b.x3f(k,j,il);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JetOuterX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

void JetOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // set primitive variables in inlet ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
        if (rad <= r_jet_R) {
          prim(IDN,k,j,iu+i) = d_jet_R;
          prim(IVX,k,j,iu+i) = vx_jet_R;
          prim(IVY,k,j,iu+i) = vy_jet_R;
          prim(IVZ,k,j,iu+i) = vz_jet_R;
          prim(IPR,k,j,iu+i) = p_jet_R;
        } else {
          prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
          prim(IVX,k,j,iu+i) = prim(IVX,k,j,iu);
          prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
          prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
          prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
        }
      }
    }
  }

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_R) {
            b.x1f(k,j,iu+i) = bx_jet_R;
          } else {
            b.x1f(k,j,iu+i) = b.x1f(k,j,iu);
          }
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_R) {
            b.x2f(k,j,iu+i) = by_jet_R;
          } else {
            b.x2f(k,j,iu+i) = b.x2f(k,j,iu);
          }
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real rad = std::sqrt(SQR(pco->x2v(j)-x2_0) + SQR(pco->x3v(k)-x3_0));
          if (rad <= r_jet_R) {
            b.x3f(k,j,iu+i) = bz_jet_R;
          } else {
            b.x3f(k,j,iu+i) = b.x3f(k,j,iu);
          }
        }
      }
    }
  }
}