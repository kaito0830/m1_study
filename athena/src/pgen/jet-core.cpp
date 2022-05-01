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
#include "../orbital_advection/orbital_advection.hpp"
#include "../utils/utils.hpp" 

// BCs on L-x1 (left edge) of grid with jet inflow conditions
void JetInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

namespace {
// Make radius of jet and jet variables global so they can be accessed by BC functions
// Real r_amb,
Real d_amb, p_amb, vx_amb, vy_amb, vz_amb, bx_amb, by_amb, bz_amb;
Real r_jet, d_jet, p_jet, vx_jet, vy_jet, vz_jet, bx_jet, by_jet, bz_jet;
Real r_core, d_core, p_core;
Real gm1, x2_0, x3_0, x1_core0, x2_core0, x3_core0;
} // namespace


// temperature press / rho
// cooling function
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  AllocateUserOutputVariables(1); // ==> user_out_var(nuov, k, j, i)
  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        // temperature = press / rho
        user_out_var(0,k,j,i) = phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i);
      }
    }
  }
}

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
  d_jet  = pin->GetReal("problem", "djet");
  p_jet  = pin->GetReal("problem", "pjet");
  vx_jet = pin->GetReal("problem", "vxjet");
  vy_jet = pin->GetReal("problem", "vyjet");
  vz_jet = pin->GetReal("problem", "vzjet");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx_jet = pin->GetReal("problem", "bxjet");
    by_jet = pin->GetReal("problem", "byjet");
    bz_jet = pin->GetReal("problem", "bzjet");
  }
  r_jet = pin->GetReal("problem", "rjet");
  x2_0 = 0.5*(mesh_size.x2max + mesh_size.x2min);
  x3_0 = 0.5*(mesh_size.x3max + mesh_size.x3min);

  r_core = pin->GetOrAddReal("problem", "rcore", 10.0);
  d_core = pin->GetOrAddReal("problem", "dcore", 5.0);
  p_core = pin->GetOrAddReal("problem", "pcore", 6e-3);
  //bx_core = pin->GetOrAddReal("problem", "bxcore", 0.0);
  //by_core = pin->GetOrAddReal("problem", "bycore", 0.2);
  //bz_core = pin->GetOrAddReal("problem","bzcore", 0.0);

  x1_core0 = 0.5*(mesh_size.x1max + mesh_size.x1min);
  x2_core0 = 0.5*(mesh_size.x2max + mesh_size.x2min);
  x3_core0 = 0.5*(mesh_size.x3max + mesh_size.x3min);

  // enroll boundary value function pointers
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, JetInnerX1);
  EnrollUserExplicitSourceFunction(Cooling); // cooling function
  return;
}

//cooling function
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar)
{
  Real g = pmb->peos->GetGamma();
  Real temp_goal = 10.0;
  Real tau = 0.01;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real temp = prim(IPR,k,j,i) / prim(IDN,k,j,i);
        if (temp > temp_goal) {
          cons(IEN,k,j,i) -= dt / tau * prim(IDN,k,j,i) * (temp - temp_goal) / (g - 1.0);
        }
      }
    }
  }
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
        
        Real rad3 = std::sqrt(SQR(pcoord->x1v(i)-x1_core0) + SQR(pcoord->x2v(j)-x2_core0) + SQR(pcoord->x3v(k)-x3_core0));
        if (rad3 < r_core ) {
            phydro->u(IDN,k,j,i) = d_core;
            phydro->u(IM1,k,j,i) = 0.0;
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
            if (NON_BAROTROPIC_EOS) {
                phydro->u(IEN,k,j,i) = p_core/gm1;
            }
        } else {
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
          Real rad2 = sqrt(SQR(pcoord->x1v(i)-x1_core0) + SQR(pcoord->x3v(k)-x3_core0));
          if (rad2 < r_core) {
            pfield->b.x2f(k,j,i) = by_amb;
          } else {
            pfield->b.x2f(k,j,i) = 0.0;
          }
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
        if (rad <= r_jet) {
          prim(IDN,k,j,il-i) = d_jet;
          prim(IVX,k,j,il-i) = vx_jet;
          prim(IVY,k,j,il-i) = vy_jet;
          prim(IVZ,k,j,il-i) = vz_jet;
          prim(IPR,k,j,il-i) = p_jet;
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
          if (rad <= r_jet) {
            b.x1f(k,j,il-i) = bx_jet;
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
          if (rad <= r_jet) {
            b.x2f(k,j,il-i) = by_jet;
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
          if (rad <= r_jet) {
            b.x3f(k,j,il-i) = bz_jet;
          } else {
            b.x3f(k,j,il-i) = b.x3f(k,j,il);
          }
        }
      }
    }
  }
}

