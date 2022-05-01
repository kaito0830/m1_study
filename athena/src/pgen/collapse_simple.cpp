//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file collapse.cpp
//! \brief Problem generator for collapse of a Bonnor-Ebert like sphere with AMR or SMR

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif


// dimension-less constants
constexpr Real four_pi_G = 1.0;

// Mask the density outside the initial sphere
void SourceMask(AthenaArray<Real> &src, int is, int ie, int js, int je,
                int ks, int ke, const MGCoordinates &coord) {
  const Real rc2 = rc*rc;
  for (int k=ks; k<=ke; ++k) {
    Real z = coord.x3v(k);
    for (int j=js; j<=je; ++j) {
      Real y = coord.x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = coord.x1v(i);
        Real r2 = x*x + y*y + z*z;
        if (r2 > rc2)
          src(k, j, i) = 0.0;
      }
    }
  }
  return;
}




//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  SetFourPiG(four_pi_G); // 4piG = 1.0
  EnrollUserMGGravitySourceMaskFunction(SourceMask);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real igm1 = 1.0 / (peos->GetGamma() - 1.0);

  for (int k=ks; k<=ke; ++k) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        // èâä˙èåèÇçÏÇÈ

      }
    }
  }
}

