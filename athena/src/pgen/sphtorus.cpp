/*============================================================================*/
/*! \file sphtorus.cpp
 *  \brief Problem generator for the torus problem (Stone et al. 1999)
 *
 * PURPOSE: Problem generator for the torus problem (Stone et al. 1999)
/*============================================================================*/

// C/C++ headers
#include <iostream>   // endl
#include <cmath>      // sqrt
#include <cstdlib>    // srand

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"

using namespace std;

#ifdef ISOTHERMAL
#error "Isothermal EOS cannot be used."
#endif

#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator does not support magnetic fields"
#endif

/*----------------------------------------------------------------------------*/
/* function prototypes and global variables*/
 //sets BCs on inner-x1 (left edge) of grid.
void stbv_iib(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
//sets BCs on outer-x1 (right edge) of grid.
void stbv_oib(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

Real magr(MeshBlock *pmb,   int i,   int j,   int k);
Real magt(MeshBlock *pmb,   int i,   int j,   int k);
Real magp(MeshBlock *pmb,   int i,   int j,   int k);
Real en, cprime, w0, rg, dist, acons, d0, denv, amp, beta;
static Real gm,gmgas;

inline Real CUBE(Real x){
  return ((x)*(x)*(x));
}

inline Real MAX(Real x, Real y){
  return (x>y?x:y);
}

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // read parameters
  gm = pin->GetReal("problem","GM");
  gmgas = pin->GetReal("hydro","gamma");
  dist = pin->GetReal("problem","dist");
  rg = pin->GetReal("problem","rg");
  d0= pin->GetReal("problem","d0");
  amp= pin->GetReal("problem","amp");
  denv = pin->GetReal("problem","denv");
  if (MAGNETIC_FIELDS_ENABLED)
    beta = pin->GetReal("problem","beta");
  w0 = 1.0;
  cprime = 0.5/dist;
  en = 1.0/(gmgas-1.0);
  acons=0.5*(dist-1.0)/dist/(en+1.0);

  // assign boundary conditions
  if(mesh_bcs[inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(inner_x1, stbv_iib);
  }
  if(mesh_bcs[outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(outer_x1, stbv_oib);
  }
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the torus problem (Stone et al. 1999)
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int ii,jj,ftorus;
  Real rv, pp, eq29, dens, wt, pr;

  std::srand(gid);
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // Background
        phydro->u(IDN,k,j,i) = denv;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        pr=denv/pcoord->x1v(i);

        // Torus - Papaloizou & Pringle 1984 eq (2.9), Stone et al. 1999
        rv=pcoord->x1v(i);
        wt = rv*sin(pcoord->x2v(j));
        eq29 = gm/(w0*(en + 1.))*(w0/rv-0.5*SQR(w0/wt) - cprime);
        if (eq29 > 0.0) {
          dens = pow(eq29/acons,en);
          pp=dens*eq29;
          if (pp > denv/rv) {
            phydro->u(IDN,k,j,i) = dens;
            phydro->u(IM3,k,j,i) = dens*sqrt(gm*w0)/wt;
            pr=MAX(acons*pow(dens,gmgas),pr)*(1+amp*((double)rand()/(double)RAND_MAX-0.5));
          }
        }
        phydro->u(IEN,k,j,i)=pr*en+0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
      }
    }
  }
}


/*  Boundary Condtions */
void stbv_iib(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
              Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);
        prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
        prim(IVY,k,j,is-i) = prim(IVY,k,j,is); //corotating ghost region
        prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
        prim(IPR,k,j,is-i) = prim(IPR,k,j,is-i+1);//-gm/SQR(pco->x1f(is-i+1))*prim(IDN,k,j,is-i+1)*pco->dx1v(is-i);
        if(prim(IVX,k,j,is-i) > 0.0)
          prim(IVX,k,j,is-i) = 0.0;
      }
    }
  }
  return;
}


void stbv_oib(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
              Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);
        prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
        prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
        prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
        prim(IPR,k,j,ie+i) = prim(IPR,k,j,ie);
        if(prim(IVX,k,j,ie+i) < 0.0)
          prim(IVX,k,j,ie+i) = 0.0;
      }
    }
  }
  return;
}
