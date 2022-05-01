/*============================================================================*/
/*! \file mhdtorus.cpp
 *  \brief Problem generator for the torus problem (Stone and Pringle 2001)
 *
 * PURPOSE: Problem generator for the torus problem (Stone and Pringle 2001)
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

Real A3(Real x1, Real x2,Real x3)
{
  Real eq29,w;
  Real a=0.0;
  Real dens;
  w=x1*sin(x2);
  eq29 = gm/(w0*(en + 1.))*(w0/x1-0.5*SQR(w0/w) - cprime);
  if (eq29 > 0.0) {
    dens  = pow(eq29/acons,en);
    if (dens > 2.0*denv)
      a = SQR(dens)/(beta);
  }
  return a;
}

#define ND 100

Real magr(MeshBlock *pmb,   int i,   int j,   int k)
{
  Real r,t,p,s,a,d,rd;
  Coordinates *pco = pmb->pcoord;
  int n;
  r = pco->x1f(i);
  t = pco->x2f(j);
  p = pco->x3f(k);
  s=pco->GetFace1Area(k, j, i);
  a=(A3(r,t+pco->dx2f(j),p+0.5*pco->dx3f(k))*sin(t+pco->dx2f(j))-A3(r,t,p+0.5*pco->dx3f(k))*sin(t))*r*pco->dx3f(k);
  return a/s;
}

Real magt(MeshBlock *pmb, int i, int j, int k)
{
  Coordinates *pco = pmb->pcoord;
  Real r,t,p,s,a,d,rd;
  int n;
  r = pco->x1f(i);
  t = pco->x2f(j);
  p = pco->x3f(k);
  s=pco->GetFace2Area(k, j, i);
  a=(A3(r+pco->dx1f(i),t,p+0.5*pco->dx3f(k))*(r+pco->dx1f(i))-A3(r,t,p+0.5*pco->dx3f(k))*r)*sin(t)*pco->dx3f(k);
  return -a/s;
}

Real magp(MeshBlock *pmb, int i, int j, int k)
{
  return 0.0;
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
  Real rv, pp, eq29, dens, wt;

  AthenaArray<Real> pr;
  std::srand(gid);       

  /* allocate memory for the gas pressure */
  pr.NewAthenaArray(block_size.nx3+2*(NGHOST),block_size.nx2+2*(NGHOST),block_size.nx1+2*(NGHOST));
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // Background
        phydro->u(IDN,k,j,i) = denv;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        pr(k,j,i)=denv/pcoord->x1v(i);

        // Torus
        rv=pcoord->x1v(i);
        wt = rv*sin(pcoord->x2v(j));
        eq29 = gm/(w0*(en + 1.))*(w0/rv-0.5*SQR(w0/wt) - cprime);
        if (eq29 > 0.0) {
          dens = pow(eq29/acons,en);
          pp=dens*eq29;
          if (pp > denv/rv) {
            phydro->u(IDN,k,j,i) = dens;
            phydro->u(IM3,k,j,i) = dens*sqrt(gm*w0)/wt;
            pr(k,j,i)=MAX(acons*pow(dens,gmgas),pr(k,j,i))*(1+amp*((double)rand()/(double)RAND_MAX-0.5));
          }
        }
      }
    }
  }


  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = magr(this,i,j,k);
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          if(abs(sin(pcoord->x2f(j))) < 1e-4)
            pfield->b.x2f(k,j,i) = 0.0;
          else
            pfield->b.x2f(k,j,i) = magt(this,i,j,k);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = 0.0;
        }
      }
    }
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IEN,k,j,i)=pr(k,j,i)*en+0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
        // //Adding the magnetic energy contributions onto the internal energy 
        if (MAGNETIC_FIELDS_ENABLED) {
          Real bx = ((pcoord->x1f(i+1)-pcoord->x1v(i))*pfield->b.x1f(k,j,i)
             +  (pcoord->x1v(i)-pcoord->x1f(i))*pfield->b.x1f(k,j,i+1))/pcoord->dx1f(i);
          Real by = ((pcoord->x2f(j+1)-pcoord->x2v(j))*pfield->b.x2f(k,j,i)
             +  (pcoord->x2v(j)-pcoord->x2f(j))*pfield->b.x2f(k,j+1,i))/pcoord->dx2f(j);
          Real bz = (pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))*0.5;
          phydro->u(IEN,k,j,i) += 0.5*(SQR(bx)+SQR(by)+SQR(bz));
        }
      }
    }
  }

  pr.DeleteAthenaArray();
}


/*  Boundary Condtions, outflowing, ix1, ox1, ix2, ox2  */
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
        prim(IPR,k,j,is-i)=prim(IPR,k,j,is-i+1);//-gm/SQR(pco->x1f(is-i+1))*prim(IDN,k,j,is-i+1)*pco->dx1v(is-i);
        if(prim(IVX,k,j,is-i) > 0.0)
          prim(IVX,k,j,is-i) = 0.0;
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = b.x1f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
      }
    }}
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
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
      }
    }}
  }
  return;
}
