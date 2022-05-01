//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_loop_poles.c
//  \brief Advection of a field loop THROUGH the poles in spherical_polar coordinates.
//
//  Originally developed by ZZ.  Sets up constant uniform-density flow in x-direction
//  through poles, and follows advection of loop.  Set xz>0 (xz<0) for loop through
//  upper (lower) pole.  Works in 2D and 3D.
//========================================================================================

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <stdio.h>
using namespace std;

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#define  INTERNAL_ENERGY_ENABLED 0
namespace{
Real amp;
// Parameters
Real rho_inf;
Real beta_Bondi;
Real bz_Bondi, R_Bondi;
//Real rho_inf, pr_inf, tem_inf, j_inf, dfloor_pole;
Real dfloor_pole, dfloor;
int flag_Bz, flag_Bz_type, flag_res, flag_cooling, flag_vis;
Real Rcyl_c, sigma_ring, Rcyl_cen,Rin_torus,Rout_torus,A,eta_in,den_res_lim;
// damping zone
Real f_den_damp, f_damp_min, f_damp_max, f_w_damp, f_Rin_torus,
  f_tem_center, den_center, pressure_floor, tem_center;
// SetBG switch
Real time_AddingB;
Real time_AddingVis, alpha, den_eta;
Real time_AddingNC,tem_cool,rho_cool,tem_lim,tem_lim_hot,theta_start,theta_end;
Real tem_norm,rho_norm,time_norm;
//variables for user_out_var(shuto)
int nav_uov = 66;
int flag_uov;
int nav_uov_add; // additional varialbles
Real dt_uov,tlim_uov;
AthenaArray<Real> J1,J2,J3,vor1,vor2,vor3,avrg,uov_tmp;
AthenaArray<Real> vzBR,vRBz,vpBz,vzBp,vRBp,vpBR,
  dvzBR_dz,dvRBz_dz,dvzBR_dR,dvRBz_dR,dvpBz_dz,dvzBp_dz,dvRBp_dR,dvpBR_dR;
AthenaArray<Real> tempt, vrad_prev;
AthenaArray<Real> x1area;
AthenaArray<Real> den_surf, pr_surf;
AthenaArray<Real> face_area_,edge_len_,edge_len_m1_;
EdgeField Je_;
int *granks = 0;
}

#ifdef MPI_PARALLEL
MPI_Comm gcomm;
#endif

void Curl1(AthenaArray<Real> &J1, MeshBlock *pmb);
void Curl2(AthenaArray<Real> &J2, MeshBlock *pmb);
void Curl3(AthenaArray<Real> &J3, MeshBlock *pmb);
void Vorticity1(AthenaArray<Real> &vor1, MeshBlock *pmb);
void Vorticity2(AthenaArray<Real> &vor2, MeshBlock *pmb);
void Vorticity3(AthenaArray<Real> &vor3, MeshBlock *pmb);
void Der_R(AthenaArray<Real> &df, AthenaArray<Real> &f, MeshBlock *pmb);
void Der_z(AthenaArray<Real> &df, AthenaArray<Real> &f, MeshBlock *pmb);

// User-defined boundary conditions
void BCInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BCOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DampingLayer_NoVdamp(Hydro *phydro, Coordinates *pcoord,
			  Real dt1, Real rmin, Real r_damp, Real w_rad, Real t_cross,
			  int il, int iu, int jl, int ju, int kl, int ku);

void Diffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
     const AthenaArray<Real> &bmag, const int is, const int ie, const int js,
                   const int je, const int ks, const int ke) ;

void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
     const AthenaArray<Real> &bc, int is, int ie, int js, int je, int ks, int ke) ;

// User-defined function 
inline Real CUBE(Real x){
  return ((x)*(x)*(x));
}
inline Real MAX(Real x, Real y){
  return (x>y?x:y);
}
inline Real MIN(Real x, Real y){
  return (x<y?x:y);
}
int SIGN1(Real x){
  return (x >= 0) - (x < 0);
}

void SetBGMagneticField(MeshBlock *pmb, int is, int ie, int js, int je, int ks, int ke);
Real A1_BG(const Real x1, const Real x2, const Real x3);
Real A2_BG(const Real x1, const Real x2, const Real x3);
Real A3_BG(const Real x1, const Real x2, const Real x3);


Real MagneticEnergyAroundStar(MeshBlock *pmb, int iout);
Real GetStellarUnsignedMagFlux(MeshBlock *pmb, int iout);
Real GetAccretionRate(MeshBlock *pmb, int iout);

Real A1_BG(const Real x1, const Real x2,const Real x3)
{
  return 0.0;
}

Real A2_BG(const Real x1, const Real x2,const Real x3) 
{
  return 0.0;
}

Real A3_BG(const Real x1, const Real x2,const Real x3)
{
  // Uniform vertical field
  Real R = x1*sin(x2);
  Real R_mod = x1*sin(x2) + 1e-6;
  Real a3_torus0 = 0.5 * bz_Bondi;
  Real a3_torus;

  if (flag_Bz_type == 0) {
    a3_torus = a3_torus0 * R; // uniform field
  }
  return a3_torus;
}


void SetBGMagneticField(MeshBlock *pmb, int is, int ie, int js, int je, int ks, int ke){
  Coordinates *pcoord=pmb->pcoord;
  Field *pfield = pmb->pfield;
  BoundaryValues *pbval = pmb->pbval;

  int f3=0;
  if (pmb->block_size.nx3 > 1) f3=1;

  AthenaArray<Real> a1,a2,a3;
  int nx1 = (ie-is)+1 + 2*(NGHOST);
  int nx2 = (je-js)+1 + 2*(NGHOST);
  int nx3 = (ke-ks)+1 + 2*(NGHOST);
  a1.NewAthenaArray(nx3+1,nx2+1,nx1);
  a2.NewAthenaArray(nx3+1,nx2,nx1+1);
  a3.NewAthenaArray(nx3,nx2+1,nx1+1);
  int level=pmb->loc.level;
  // Initialize components of the vector potential
  if (pmb->block_size.nx3 > 1) {
    for (int k=ks-NGHOST; k<=ke+1+NGHOST; k++) {
      for (int j=js-NGHOST; j<=je+1+NGHOST; j++) {
        for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
          Real x1c = 0.5*(pcoord->x1f(i)+pcoord->x1f(i+1));
            if ((pbval->nblevel[1][0][1]>level && j==js) || (pbval->nblevel[1][2][1]>level && j==je+1)
             || (pbval->nblevel[0][1][1]>level && k==ks) || (pbval->nblevel[2][1][1]>level && k==ke+1)
             || (pbval->nblevel[0][0][1]>level && j==js   && k==ks)
             || (pbval->nblevel[0][2][1]>level && j==je+1 && k==ks)
             || (pbval->nblevel[2][0][1]>level && j==js   && k==ke+1)
             || (pbval->nblevel[2][2][1]>level && j==je+1 && k==ke+1)) {
              Real x1l = pcoord->x1f(i)+0.25*pcoord->dx1f(i);
              Real x1r = pcoord->x1f(i)+0.75*pcoord->dx1f(i);
              a1(k,j,i) = 0.5*(A1_BG(x1l, pcoord->x2f(j), pcoord->x3f(k)) +
                               A1_BG(x1r, pcoord->x2f(j), pcoord->x3f(k)));
            } else {
              a1(k,j,i) = A1_BG(x1c, pcoord->x2f(j), pcoord->x3f(k));
            }
          }
        }
      }
      for (int k=ks-NGHOST; k<=ke+1+NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          Real x2c = 0.5*(pcoord->x2f(j)+pcoord->x2f(j+1));
          for (int i=is-NGHOST; i<=ie+1+NGHOST; i++) {
            if ((pbval->nblevel[1][1][0]>level && i==is) || (pbval->nblevel[1][1][2]>level && i==ie+1)
             || (pbval->nblevel[0][1][1]>level && k==ks) || (pbval->nblevel[2][1][1]>level && k==ke+1)
             || (pbval->nblevel[0][1][0]>level && i==is   && k==ks)
             || (pbval->nblevel[0][1][2]>level && i==ie+1 && k==ks)
             || (pbval->nblevel[2][1][0]>level && i==is   && k==ke+1)
             || (pbval->nblevel[2][1][2]>level && i==ie+1 && k==ke+1)) {
              Real x2l = pcoord->x2f(j)+0.25*pcoord->dx2f(j);
              Real x2r = pcoord->x2f(j)+0.75*pcoord->dx2f(j);
              a2(k,j,i) = 0.5*(A2_BG(pcoord->x1f(i), x2l, pcoord->x3f(k)) +
                               A2_BG(pcoord->x1f(i), x2r, pcoord->x3f(k)));
            } else {
              a2(k,j,i) = A2_BG(pcoord->x1f(i), x2c, pcoord->x3f(k));
            }
          }
        }
      }
      for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+1+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+1+NGHOST; i++) {
            if ((pbval->nblevel[1][1][0]>level && i==is) || (pbval->nblevel[1][1][2]>level && i==ie+1)
             || (pbval->nblevel[1][0][1]>level && j==js) || (pbval->nblevel[1][2][1]>level && j==je+1)
             || (pbval->nblevel[1][0][0]>level && i==is   && j==js)
             || (pbval->nblevel[1][0][2]>level && i==ie+1 && j==js)
             || (pbval->nblevel[1][2][0]>level && i==is   && j==je+1)
             || (pbval->nblevel[1][2][2]>level && i==ie+1 && j==je+1)) {
              Real x3l = pcoord->x3f(k)+0.25*pcoord->dx3f(k);
              Real x3r = pcoord->x3f(k)+0.75*pcoord->dx3f(k);
              a3(k,j,i) = 0.5*(A3_BG(pcoord->x1f(i), pcoord->x2f(j), x3l) +
                               A3_BG(pcoord->x1f(i), pcoord->x2f(j), x3r));
            } else {
              a3(k,j,i) = A3_BG(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
            }
          }
        }
      }
    }
    else { // 2D
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js-NGHOST; j<=je+1+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+1+NGHOST; i++) {
            if (i != ie+NGHOST+1)
              a1(k,j,i) = A1_BG(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
            if (j != je+NGHOST+1)
              a2(k,j,i) = A2_BG(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
            if (k != ke+1)
              a3(k,j,i) = A3_BG(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
          }
        }
      }
    }

    AthenaArray<Real> e1lenp, e1len, e2lenp, e2len, e3lenp, e3len, area;
    e1lenp.NewAthenaArray(nx1);
    e1len.NewAthenaArray(nx1);
    e2lenp.NewAthenaArray(nx1+1);
    e2len.NewAthenaArray(nx1+1);
    e3lenp.NewAthenaArray(nx1+1);
    e3len.NewAthenaArray(nx1+1);
    area.NewAthenaArray(nx1+1);
    // Initialize interface fields
    for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
      for (int j=js-NGHOST; j<=je+NGHOST; j++) {
        pcoord->Edge2Length(k+1, j, is-NGHOST, ie+1+NGHOST, e2lenp);
        pcoord->Edge2Length(k, j, is-NGHOST, ie+1+NGHOST, e2len);
        pcoord->Edge3Length(k, j+1, is-NGHOST, ie+1+NGHOST, e3lenp);
        pcoord->Edge3Length(k, j, is-NGHOST, ie+1+NGHOST, e3len);
        pcoord->Face1Area(k, j, is-NGHOST, ie+1+NGHOST, area);
        for (int i=is-NGHOST; i<=ie+1+NGHOST; i++) {
          // std::cout << i << "  " << j << "  " << k << std::endl;
          pfield->b.x1f(k,j,i) 
            += ((a3(k  ,j+1,i)*e3lenp(i) - a3(k,j,i)*e3len(i))
                   -(a2(k+1,j  ,i)*e2lenp(i) - a2(k,j,i)*e2len(i))) / area(i);
          pfield->b1.x1f(k,j,i) = pfield->b.x1f(k,j,i);
          // std::cout << pfield->b.x1f(k,j,i) << std::endl;
        }
      }
    }
    for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
      for (int j=js-NGHOST; j<=je+1+NGHOST; j++) {
        pcoord->Edge1Length(k+1, j, is-NGHOST, ie+NGHOST, e1lenp);
        pcoord->Edge1Length(k, j, is-NGHOST, ie+NGHOST, e1len);
        pcoord->Edge3Length(k, j, is-NGHOST, ie+1+NGHOST, e3len);
        pcoord->Face2Area(k, j, is-NGHOST, ie+NGHOST, area);
        for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
          if(area(i)<1e-5)
            pfield->b.x2f(k,j,i) = pfield->b1.x2f(k,j,i) = 0.0;
          else {
            pfield->b.x2f(k,j,i) 
             += ((a1(k+1,j,i  )*e1lenp(i) - a1(k,j,i)*e1len(i))
             -(a3(k  ,j,i+1)*e3len(i+1) - a3(k,j,i)*e3len(i))) / area(i);
            pfield->b1.x2f(k,j,i) = pfield->b.x2f(k,j,i);
          }
        }
      }
    }
    for (int k=ks-NGHOST*f3; k<=ke+1+NGHOST*f3; k++) {
      for (int j=js-NGHOST; j<=je+NGHOST; j++) {
        pcoord->Edge1Length(k, j+1, is-NGHOST, ie+NGHOST, e1lenp);
        pcoord->Edge1Length(k, j, is-NGHOST, ie+NGHOST, e1len);
        pcoord->Edge2Length(k, j, is-NGHOST, ie+1+NGHOST, e2len);
        pcoord->Face3Area(k, j, is-NGHOST, ie+NGHOST, area);
        for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
          pfield->b.x3f(k,j,i) 
            += ((a2(k,j  ,i+1)*e2len(i+1) - a2(k,j,i)*e2len(i))
             -(a1(k,j+1,i  )*e1lenp(i) - a1(k,j,i)*e1len(i))) / area(i);
          pfield->b1.x3f(k,j,i) = pfield->b.x3f(k,j,i);
        }
      }
    }
    e1lenp.DeleteAthenaArray();
    e1len.DeleteAthenaArray();
    e2lenp.DeleteAthenaArray();
    e2len.DeleteAthenaArray();
    e3lenp.DeleteAthenaArray();
    e3len.DeleteAthenaArray();
    area.DeleteAthenaArray();
    a1.DeleteAthenaArray();
    a2.DeleteAthenaArray();
    a3.DeleteAthenaArray();

}

Real MagneticEnergyAroundStar(MeshBlock *pmb, int iout)
{
  Real emstar=0.0;
  Field *pfield = pmb->pfield;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  Real r1 = 5.0;
  Real wr = 0.00001;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);
        Real bsq = SQR(pfield->bcc(IB1,k,j,i))
                 + SQR(pfield->bcc(IB2,k,j,i))
                 + SQR(pfield->bcc(IB3,k,j,i));
//     Real bsq = pow(pfield->bcc(IB1,k,j,i),2)
//              + pow(pfield->bcc(IB2,k,j,i),2)
//              + pow(pfield->bcc(IB3,k,j,i),2);
    // Integrate Em only within the radius of r1
        emstar += 0.5 * vol * bsq * 0.5*(1.0-tanh((pmb->pcoord->x1v(i)-r1)/wr));
      }
    }
  }

  return emstar;
}

Real TotalMassAroundStar(MeshBlock *pmb, int iout)
{
  Real mstar=0.0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  Real r1 = 5.0;
  Real wr = 0.00001;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);
        mstar += vol * pmb->phydro->w(IDN,k,j,i) 
          * 0.5*(1.0-tanh((pmb->pcoord->x1v(i)-r1)/wr));
      }
    }
  }

  return mstar;
}

Real GetStellarUnsignedMagFlux(MeshBlock *pmb, int iout)
{
  Real Phi_star=0.0;
  Field *pfield = pmb->pfield;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real r1 = 1.0 ; // sampling radius

  if(pmb->block_size.x1min > r1){
    return 0.0;
  }else if(pmb->block_size.x1max < r1){
    return 0.0;
  }else{

    int ii;
    for (ii=is; ii<=ie+1; ii++) {
      if(pmb->pcoord->x1f(ii) >= r1) break;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real area = pmb->pcoord->GetFace1Area(k,j,ii);
        Phi_star += area * fabs(pfield->b.x1f(k,j,ii)); // unsigned radial magneic field flux
      }
    }
    return Phi_star;
  }
  
}

Real GetAccretionRate(MeshBlock *pmb, int iout)
{
  int iout_start = 3; // CHECK IT!!
  Real mdot=0.0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real r1 = 1.0 + 4.0*(iout - iout_start); // sampling radius


  if(pmb->block_size.x1min > r1){
    return 0.0;
  }else if(pmb->block_size.x1max < r1){
    return 0.0;
  }else{

    int ii;
    for (ii=is; ii<=ie+1; ii++) {
      if(pmb->pcoord->x1f(ii) >= r1) break;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real area = pmb->pcoord->GetFace1Area(k,j,ii);
        mdot -= area * pmb->phydro->flux[X1DIR](IDN,k,j,ii);
      }
    }
    return mdot;
  }
  
}

Real GetSignedAccretionRate(MeshBlock *pmb, int iout)
{
  int iout_start = 11; // CHECK IT!!
  Real mdot=0.0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real r1 = 3.0 + 2.0*(iout - iout_start); // sampling radius


  if(pmb->block_size.x1min > r1){
    return 0.0;
  }else if(pmb->block_size.x1max < r1){
    return 0.0;
  }else{

    int ii;
    for (ii=is; ii<=ie+1; ii++) {
      if(pmb->pcoord->x1f(ii) >= r1) break;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real area = pmb->pcoord->GetFace1Area(k,j,ii);
        //mdot -= area * pmb->phydro->flux[X1DIR](IDN,k,j,ii);
        Real massflux = pmb->phydro->flux[X1DIR](IDN,k,j,ii);
        if (massflux < 0.0) {
          mdot -= area * massflux;
        } else {
          mdot -= 0.0;
        }
          //mdot -= area * pmb->phydro->flux[X1DIR](IDN,k,j,ii);
      }
    }
    return mdot;
  }
  
}


Real GetOutflowMassLossRate(MeshBlock *pmb, int iout)
{
  int iout_start = 14; // CHECK IT!!
  Real mdot=0.0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real r1 = 3.0 + 2.0*(iout - iout_start); // sampling radius

  Real vesc1 = sqrt(2.0/r1);

  if(pmb->block_size.x1min > r1){
    return 0.0;
  }else if(pmb->block_size.x1max < r1){
    return 0.0;
  }else{

    int ii;
    for (ii=is; ii<=ie+1; ii++) {
      if(pmb->pcoord->x1f(ii) >= r1) break;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real area = pmb->pcoord->GetFace1Area(k,j,ii);
        Real massflux = pmb->phydro->flux[X1DIR](IDN,k,j,ii);
        Real vrad = 0.5*(pmb->phydro->w(IVX,k,j,ii) + pmb->phydro->w(IVX,k,j,ii+1));
        
        if (massflux > 0.0 && vrad/vesc1 > 0.3) {
          mdot += area * massflux;
        } else {
          mdot += 0.0;
        }
      }
    }
    return mdot;
  }
  
}



void Curl1(AthenaArray<Real> &J1, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Field *pfield = pmb->pfield;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke; k++){
      Real dphi2 = pcoord->dx3v(k-1) + pcoord->dx3v(k);
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dsinBphi = pfield->bcc(IB3,k,j+1,i) * sin(pcoord->x2v(j+1))
            - pfield->bcc(IB3,k,j-1,i) * sin(pcoord->x2v(j-1));
          Real dBtheta = pfield->bcc(IB2,k+1,j,i) - pfield->bcc(IB2,k-1,j,i);
          J1(k,j,i) = r_inv * sint_inv * (dsinBphi/dtheta2 - dBtheta/dphi2)  ;
        }
      }
    }
  }
  else { // 2D
    for (int k=ks; k<=ke; k++){
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dsinBphi = pfield->bcc(IB3,k,j+1,i) * sin(pcoord->x2v(j+1))
            - pfield->bcc(IB3,k,j-1,i) * sin(pcoord->x2v(j-1));
          J1(k,j,i) = r_inv * sint_inv * dsinBphi/dtheta2  ;
        }
      }
    }
  }
}

void Vorticity1(AthenaArray<Real> &vor1, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro ;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke; k++){
      Real dphi2 = pcoord->dx3v(k-1) + pcoord->dx3v(k);
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dsinVphi = phydro->w(IVZ,k,j+1,i) * sin(pcoord->x2v(j+1))
            - phydro->w(IVZ,k,j-1,i) * sin(pcoord->x2v(j-1));
          Real dVtheta = phydro->w(IVY,k+1,j,i) - phydro->w(IVY,k-1,j,i);
          vor1(k,j,i) = r_inv * sint_inv * (dsinVphi/dtheta2 - dVtheta/dphi2)  ;
        }
      }
    }
  }
  else { // 2D
    for (int k=ks; k<=ke; k++){
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dsinVphi = phydro->w(IVZ,k,j+1,i) * sin(pcoord->x2v(j+1))
            - phydro->w(IVZ,k,j-1,i) * sin(pcoord->x2v(j-1));
          vor1(k,j,i) = r_inv * sint_inv * dsinVphi/dtheta2  ;
        }
      }
    }
  }
}

void Curl2(AthenaArray<Real> &J2, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Field *pfield = pmb->pfield;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke; k++){
      Real dphi2 = pcoord->dx3v(k-1) + pcoord->dx3v(k);
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
          Real dBr = pfield->bcc(IB1,k+1,j,i) - pfield->bcc(IB1,k-1,j,i);
          Real drBphi = pcoord->x1v(i+1) * pfield->bcc(IB3,k,j,i+1) 
            - pcoord->x1v(i-1) * pfield->bcc(IB3,k,j,i-1) ;
          J2(k,j,i) = r_inv * (sint_inv * dBr/dphi2 - drBphi/dr2)  ;
        }
      }
    }
  }
  else { // 2D
    for (int k=ks; k<=ke; k++){
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
          Real drBphi = pcoord->x1v(i+1) * pfield->bcc(IB3,k,j,i+1) 
            - pcoord->x1v(i-1) * pfield->bcc(IB3,k,j,i-1) ;
          J2(k,j,i) = r_inv * (- drBphi/dr2)  ;
        }
      }
    }
  }
}

void Vorticity2(AthenaArray<Real> &vor2, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke; k++){
      Real dphi2 = pcoord->dx3v(k-1) + pcoord->dx3v(k);
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
          Real dVr = phydro->w(IVX,k+1,j,i) - phydro->w(IVX,k-1,j,i);
          Real drVphi = pcoord->x1v(i+1) * phydro->w(IVZ,k,j,i+1) 
            - pcoord->x1v(i-1) * phydro->w(IVZ,k,j,i-1) ;
          vor2(k,j,i) = r_inv * (sint_inv * dVr/dphi2 - drVphi/dr2)  ;
        }
      }
    }
  }
  else { // 2D
    for (int k=ks; k<=ke; k++){
      for (int j=js; j<=je; j++){
        Real sint_inv = 1.0 / sin(pcoord->x2v(j));
        for (int i=is; i<=ie; i++){
          Real r_inv = 1.0/pcoord->x1v(i);
          Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
          Real drVphi = pcoord->x1v(i+1) * phydro->w(IVZ,k,j,i+1) 
            - pcoord->x1v(i-1) * phydro->w(IVZ,k,j,i-1) ;
          vor2(k,j,i) = r_inv * (- drVphi/dr2)  ;
        }
      }
    }
  }
}

void Curl3(AthenaArray<Real> &J3, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Field *pfield = pmb->pfield;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for (int k=ks; k<=ke; k++){
    for (int j=js; j<=je; j++){
      Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
      for (int i=is; i<=ie; i++){
        Real r_inv = 1.0/pcoord->x1v(i);
        Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
        Real drBtheta = pcoord->x1v(i+1) * pfield->bcc(IB2,k,j,i+1) 
                      - pcoord->x1v(i-1) * pfield->bcc(IB2,k,j,i-1) ;
        Real dBr = pfield->bcc(IB1,k,j+1,i) - pfield->bcc(IB1,k,j-1,i);
        J3(k,j,i) = r_inv * (drBtheta/dr2 - dBr/dtheta2)  ;
      }
    }
  }
}

void Vorticity3(AthenaArray<Real> &vor3, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for (int k=ks; k<=ke; k++){
    for (int j=js; j<=je; j++){
      Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
      for (int i=is; i<=ie; i++){
        Real r_inv = 1.0/pcoord->x1v(i);
        Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
        Real drVtheta = pcoord->x1v(i+1) * phydro->w(IVY,k,j,i+1) 
                      - pcoord->x1v(i-1) * phydro->w(IVY,k,j,i-1) ;
        Real dVr = phydro->w(IVX,k,j+1,i) - phydro->w(IVX,k,j-1,i);
        vor3(k,j,i) = r_inv * (drVtheta/dr2 - dVr/dtheta2)  ;
      }
    }
  }
}


void Der_R(AthenaArray<Real> &df, AthenaArray<Real> &f, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro ;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for (int k=ks; k<=ke; k++){
    for (int j=js; j<=je; j++){
      Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
      Real sint = sin(pcoord->x2v(j));
      Real cost = cos(pcoord->x2v(j));
      for (int i=is; i<=ie; i++){
        Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
        Real dfdr = (f(k,j,i+1)-f(k,j,i-1))/dr2;
        Real dfdth= (f(k,j+1,i)-f(k,j-1,i))/dtheta2;
        df(k,j,i) = sint * dfdr + cost/pcoord->x1v(i) * dfdth;
      }
    }
  }
}


void Der_z(AthenaArray<Real> &df, AthenaArray<Real> &f, MeshBlock *pmb){
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro ;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for (int k=ks; k<=ke; k++){
    for (int j=js; j<=je; j++){
      Real dtheta2 = pcoord->dx2v(j-1) + pcoord->dx2v(j);
      Real sint = sin(pcoord->x2v(j));
      Real cost = cos(pcoord->x2v(j));
      for (int i=is; i<=ie; i++){
        Real dr2 = pcoord->dx1v(i-1) + pcoord->dx1v(i);
        Real dfdr = (f(k,j,i+1)-f(k,j,i-1))/dr2;
        Real dfdth= (f(k,j+1,i)-f(k,j-1,i))/dtheta2;
        df(k,j,i) = cost * dfdr - sint/pcoord->x1v(i) * dfdth;
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
  // Get parameters for initial density and velocity
  rho_inf = pin->GetReal("problem", "rho_inf");
  flag_Bz = pin->GetInteger("problem","flag_Bz");
  flag_Bz_type = pin->GetInteger("problem","flag_Bz_type");
  beta_Bondi = pin->GetReal("problem", "beta_Bondi");
  amp = pin->GetReal("problem","amp");
  dfloor_pole = pin->GetReal("problem","dfloor_pole");
  pressure_floor = pin->GetReal("hydro","pfloor");

  R_Bondi = pin->GetReal("problem","R_Bondi");
  Rcyl_cen = pin->GetReal("problem","Rcyl_cen");
 
  eta_in = pin->GetReal("problem","eta_in");
  den_res_lim = pin->GetReal("problem","den_res_lim");
 
  time_AddingB = pin->GetReal("problem","time_AddingB");
  time_AddingNC = pin->GetReal("problem","time_AddingNC");
  time_AddingVis = pin->GetReal("problem","time_AddingVis");
  alpha = pin->GetReal("problem","alpha");
  den_eta = pin->GetReal("problem","den_eta");
  tem_cool = pin->GetReal("problem","tem_cool");
  rho_cool = pin->GetReal("problem","rho_cool");
  tem_lim = pin->GetReal("problem","tem_lim");
  tem_lim_hot = pin->GetReal("problem","tem_lim_hot");
  theta_start = pin->GetReal("problem","theta_start");
  theta_end   = pin->GetReal("problem","theta_end");

  tem_norm = pin->GetReal("problem","tem_norm");
  rho_norm = pin->GetReal("problem","rho_norm");
  time_norm = pin->GetReal("problem","time_norm");

  // setup boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, BCOuterX1);
  }

  // damping zone
  // f_den_damp=pin->GetReal("problem","f_den_damp");
  // f_w_damp=pin->GetReal("problem","f_w_damp");
  // f_Rin_torus=pin->GetReal("problem","f_Rin_torus");
  // f_tem_center=pin->GetReal("problem","f_tem_center");
  // den_center=pin->GetReal("problem","den_center");
  
  // f_damp_min=pin->GetReal("problem","f_damp_min"); // factor of the damping time
  // f_damp_max=pin->GetReal("problem","f_damp_max"); // factor of the damping time
  
  // AllocateUserHistoryOutput(17);
  // EnrollUserHistoryOutput(0,MagneticEnergyAroundStar,"emstar");
  // EnrollUserHistoryOutput(1,TotalMassAroundStar,"mstar");
  // EnrollUserHistoryOutput(2,GetStellarUnsignedMagFlux,"umflux");
  // EnrollUserHistoryOutput(3,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(4,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(5,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(6,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(7,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(8,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(9,GetAccretionRate,"dmdot");
  // EnrollUserHistoryOutput(10,GetAccretionRate,"dmdot");

  // EnrollUserHistoryOutput(11,GetSignedAccretionRate,"smdot");
  // EnrollUserHistoryOutput(12,GetSignedAccretionRate,"smdot");
  // EnrollUserHistoryOutput(13,GetSignedAccretionRate,"smdot");

  // EnrollUserHistoryOutput(14,GetOutflowMassLossRate,"omdot");
  // EnrollUserHistoryOutput(15,GetOutflowMassLossRate,"omdot");
  // EnrollUserHistoryOutput(16,GetOutflowMassLossRate,"omdot");

  // User-defined resistivity (anomalous resistivity)
  flag_res    = pin->GetInteger("problem","flag_res");
  if(flag_res != 0) {
    EnrollFieldDiffusivity(Diffusivity); // User-defined diffusivity
  }


  // User-defined viscosity (anomalous viscosity)
  flag_vis    = pin->GetInteger("problem","flag_vis");
  if(flag_vis != 0) {
    EnrollViscosityCoefficient(AlphaViscosity); // User-defined diffusivity
  }

  // user_out_var(shuto)
  // flag_uov = pin->GetInteger("problem","flag_uov");
  // std::string num_uov = pin->GetString("problem","num_output_uov");
  // dt_uov   =  pin->GetReal("output"+num_uov,"dt");
  // tlim_uov =  pin->GetReal("time","tlim");

  // Newton-cooling
  flag_cooling = pin->GetInteger("problem","flag_cooling");

  return;
}

//start(shuto)
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  // if(flag_uov == 0){
  //   nav_uov_add = (flag_res == 0) ? 2 : 3; // eta included when flag_res=1
  //   AllocateUserOutputVariables(nav_uov+nav_uov_add); // user_out_var
  //   //AllocateUserOutputVariables(nav_uov+6); // user_out_var
  // } else if (flag_uov == 1){
  //   AllocateUserOutputVariables(2*nav_uov); // user_out_var
  // }
  
  // // For inserting stellar dipole field
  // AllocateIntUserMeshBlockDataField(1);
  // iuser_meshblock_data[0].NewAthenaArray(1);
  // iuser_meshblock_data[0](0) = 0;

  // // Create AthenaArray for user_out_var
  // int nx1 = (ie-is)+1 + 2*(NGHOST);
  // int nx2 = (je-js)+1 + 2*(NGHOST);
  // int nx3 = (ke-ks)+1 + 2*(NGHOST);

  // J1.NewAthenaArray(nx3,nx2,nx1);
  // J2.NewAthenaArray(nx3,nx2,nx1);
  // J3.NewAthenaArray(nx3,nx2,nx1);

//   vor1.NewAthenaArray(nx3,nx2,nx1);
//   vor2.NewAthenaArray(nx3,nx2,nx1);
//   vor3.NewAthenaArray(nx3,nx2,nx1);

//   vzBR.NewAthenaArray(nx3,nx2,nx1);
//   vRBz.NewAthenaArray(nx3,nx2,nx1);
//   vpBz.NewAthenaArray(nx3,nx2,nx1);
//   vzBp.NewAthenaArray(nx3,nx2,nx1);
//   vRBp.NewAthenaArray(nx3,nx2,nx1);
//   vpBR.NewAthenaArray(nx3,nx2,nx1);

//   dvzBR_dz.NewAthenaArray(nx3,nx2,nx1);
//   dvRBz_dz.NewAthenaArray(nx3,nx2,nx1);
//   dvzBR_dR.NewAthenaArray(nx3,nx2,nx1);
//   dvRBz_dR.NewAthenaArray(nx3,nx2,nx1);
//   dvpBz_dz.NewAthenaArray(nx3,nx2,nx1);
//   dvzBp_dz.NewAthenaArray(nx3,nx2,nx1);
//   dvRBp_dR.NewAthenaArray(nx3,nx2,nx1);
//   dvpBR_dR.NewAthenaArray(nx3,nx2,nx1);

//   avrg.NewAthenaArray(nav_uov,nx2,nx1);
//   uov_tmp.NewAthenaArray(nav_uov,nx3,nx2,nx1);

//   den_surf.NewAthenaArray(nx3,nx2);
//   pr_surf.NewAthenaArray(nx3,nx2);
//   tempt.NewAthenaArray(nx3,nx2,nx1);
//   vrad_prev.NewAthenaArray(nx3,nx2,nx1);
//   dfloor=pin->GetReal("hydro","dfloor");

//   x1area.NewAthenaArray(nx1+1);

//   // for anomalous resistivity
//   face_area_.NewAthenaArray(nx1);
//   edge_len_.NewAthenaArray(nx1);
//   edge_len_m1_.NewAthenaArray(nx1);
//   Je_.x1e.NewAthenaArray(nx3+1,nx2+1,nx1);
//   Je_.x2e.NewAthenaArray(nx3+1,nx2,nx1+1);
//   Je_.x3e.NewAthenaArray(nx3,nx2+1,nx1+1);

//   if (block_size.nx3 > 1) { // if 3D
//     // Get ranks of processes arranged in the phi direction (granks)
//     LogicalLocation loc0;
//     long int lx3;
//     int nrbx3 = pmy_mesh->nrbx3; // meshblock number at the root level
//     int nbx3 = nrbx3 << (loc.level-pmy_mesh->root_level);
//     granks = new int[nbx3];
    
//     loc0.lx1 = loc.lx1, loc0.lx2 = loc.lx2, loc0.level = loc.level;
//     for (lx3=0;lx3<nbx3;lx3++){
//       loc0.lx3 = lx3;
//       MeshBlockTree *mbt = pmy_mesh->tree.FindMeshBlock(loc0);
//       granks[lx3] = mbt->gid_; // global MPI rank
//     }

//     // Create a new MPI communication group in which only processes arranged
//     // in the phi direction join (gcomm)
// #ifdef MPI_PARALLEL
//     MPI_Group group,group0;
//     int ngroup = nbx3;
//     MPI_Comm_group(MPI_COMM_WORLD, &group0);
//     MPI_Group_incl(group0,ngroup,granks,&group);
//     MPI_Comm_create_group(MPI_COMM_WORLD, group, 0, &gcomm);
// #endif
//   }

  return ;
}
//end(shuto)

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes field loop advection through pole.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gam = peos->GetGamma();
  Real en = 1.0/(gam - 1.0);
  // Real rho_inf = 1.0;
  Real tem_inf = 1.0/R_Bondi/gam; // R_Bondi = GM/cs^2 = GM/(gamma*tem)
  Real j_0     = sqrt(Rcyl_cen);  
  // Calculate magnetic field strength at Bondi radius  
  int f3=0;
  if (block_size.nx3 > 1) f3=1;
  std::srand(gid);

  // bz_Bondi = sqrt(2*rho*tem_inf/beta_Bondi);

  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r     = pcoord->x1v(i);
        Real theta = pcoord->x2v(j);
        Real Rcyl  = r * sin(theta) + 1e-6; // Cylindrical radius
        Real z     = r * cos(theta);
        Real v3    = j_0/Rcyl;
        Real R0    = Rcyl_cen;
        Real temp1 = R0/Rcyl - 0.5*pow(Rcyl/R0, -2);
        Real temp2 = 1+(gam-1)*R_Bondi/Rcyl_cen*temp1;
        Real rho   = rho_inf*pow(temp2, en);
        Real pr    = rho*tem_inf;
        // bz_Bondi = sqrt(2*rho*tem_inf/beta_Bondi);
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = rho*v3;
        phydro->u(IEN,k,j,i) = pr*en;
        phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))
                                     +SQR(phydro->u(IM2,k,j,i))
                                     +SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
         
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED){
    std::cout << "mag" << std::endl;

    Real R0    = Rcyl_cen;
    Real temp1 = R0/R_Bondi - 0.5*pow(R_Bondi/R0, -2);
    Real temp2 = 1+(gam-1)*R_Bondi/Rcyl_cen*temp1;
    Real rho_Bondi   = rho_inf*pow(temp2, en);
    bz_Bondi = sqrt(2*rho_Bondi*tem_inf/beta_Bondi);  
    SetBGMagneticField(this,is,ie,js,je,ks,ke);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                                      SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                                      SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));
        }
      }
    }
  }
  

  return;
}

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief User-defined work function for every time step
// contents of user_out_var (<> indicates the phi average):
// 1. <rho>
// 2. <p>
// 3. <rho^2>
// 4. <v_R>
// 5. <v_phi>
// 6. <v_z>
// 7. <rho*v_R>
// 8. <rho*v_phi>
// 9. <rho*v_z>
// 10. <rho*v_R*v_R>
// 11. <rho*v_R*v_phi>
// 12. <rho*v_R*v_z>
// 13. <rho*v_phi*v_phi>
// 14. <rho*v_phi*v_z>
// 15. <rho*v_z*v_z>
// 16. <B_R>
// 17. <B_phi>
// 18. <B_z>
// 19. <B_p> = <sqrt(B_r^2+B_theta^2)>
// 20. <B_R*B_R>
// 21. <B_R*B_phi>
// 22. <B_R*B_z>
// 23. <B_phi*B_phi>
// 24. <B_phi*B_z>
// 25. <B_z*B_z>
// 26. <v_R*B_R>
// 27. <v_R*B_phi>
// 28. <v_R*B_z>
// 29. <v_phi*B_R>
// 30. <v_phi*B_phi>
// 31. <v_phi*B_z>
// 32. <v_z*B_R>
// 33. <v_z*B_phi>
// 34. <v_z*B_z>
// 35. <v_phi*B_p>
// 36. <v_z*B_R^2>
// 37. <v_z*B_phi^2>
// 38. <v_R*B_R*B_z>
// 39. <v_phi*B_phi*B_z>
// 40. <p*v_z>
// 41. <p*v_z/rho>
// 42. <p/rho>
// 43. <J_R>
// 44. <J_phi>
// 45. <J_z>
// 46. <J_R*B_phi>
// 47. <J_R*B_z>
// 48. <J_phi*B_R>
// 49. <J_phi*B_z>
// 50. <J_z*B_R>
// 51. <J_z*B_phi>
// 52. w_r (r-comp     of vorticity)
// 53. w_t (theta-comp of vorticity)
// 54. w_p (phi-comp   of vorticity)
// 55. -dvzBR_dz
// 56.  dvRBz_dz
// 57.  vzBR/R - dvzBR_dR
// 58. -(vRBz/R - dvRBz_dR)
// 59.  dvpBz_dz
// 60.  dvzBp_dz
// 61. -dvRBp_dR
// 62.  dvpBR_dR
// 63. vrad
// 64. vtheta
// 65. v*w (total kinetic helicity)
// 66. B*J (total current helicity)
// in total, 66
// Time integration may be performed for all the quantities
// (so in this case the total number of the variables is doubled, 2*nav_uov)
//======================================================================================
//----------------------------------------------------------------------------------------
//!\f: magnetic field & newton-cooling switch
void MeshBlock::UserWorkInLoop()
{

//   int f3=0;
//   if (block_size.nx3 > 1) f3=1;

//      Real dt1 = pmy_mesh->dt;
//      Real gam = peos->GetGamma();

//   // Insert stellar magnetic field
//   if(pmy_mesh->time >= time_AddingB && iuser_meshblock_data[0](0) == 0){
//     iuser_meshblock_data[0](0) = 1;

//     Real rho_inf = 1.0;
//     Real tem_inf = 1.0/R_Bondi/gam; // R_Bondi = GM/cs^2 = GM/(gamma*tem)
//     Real r = Rcyl_c;
//     Real rho = rho_inf;
//     Real pr = rho_inf * tem_inf;
//     bz_Bondi = sqrt(2.0 * pr / beta_Bondi);
//     SetBGMagneticField(this,is,ie,js,je,ks,ke);
//     pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,
//                                        is-1,ie+1,js-1,je+1,ks-f3,ke+f3);
//   }

//   // Newton-cooling
//   if(pmy_mesh->time >= time_AddingNC && flag_cooling != 0){
//      // Step 1. Get the current temperature from w(IPR,k,j,i) and w(IDN,k,j,i) (T = p/rho)

//      Real kB = 1.38e-16;
//      Real Const =  kB*tem_norm/(gam-1.0)/rho_norm;

//      Real rho_inf = 1.0;
//      Real tem_inf = 1.0/R_Bondi/gam; // R_Bondi = GM/cs^2 = GM/(gamma*tem)
//      Real pr_inf  = rho_inf * tem_inf;
//      Real ent_inf = log(pr_inf/pow(rho_inf,gam));

//      //Real tem_lim = 2.5e-4;

//      Real lambda = 0.0;

//      for (int k=ks; k<=ke; ++k) {
//        for (int j=js; j<=je; ++j) {
//          for (int i=is; i<=ie; ++i) {
//            Real r     = pcoord->x1v(i);
//            Real theta = pcoord->x2v(j);
//            Real Rcyl  = r * sin(theta) + 1e-6; // Cylindrical radius
//            Real z     = r * cos(theta);
//            Real press = phydro->w(IPR,k,j,i);
//            Real rho = phydro->w(IDN,k,j,i);
//            Real tem = press/rho;
//            Real ent = log(press/pow(rho,gam));

//            if( (theta_start < theta) && (theta < theta_end) ){

//            if(ent > -6.85 || rho >= rho_cool){
//      // Step 2. Calculate tau(R,rho) from R = r*sin(theta) and rho
//               //Real vK = sqrt(1.0/Rcyl);
//               //Real omg = vK/Rcyl;
//               //Real P = 2*PI/omg;
//            if(tem*tem_norm <= 1e6){
//               lambda = 1e-22;
//            }else{
//               lambda = 1e-23;
//            }
//               Real t_cool = Const*tem/rho/lambda;
//               Real tau = t_cool/time_norm;

//         // Step 3. Integrate dT/dt = -(T-T*)/tau(R,rho)
//         //         --> T_new = T* + (T_old - T*) * exp(-dt/tau) ~ T* + (T_old - T*)(1-dt/tau)
//               Real dt1 = pmy_mesh->dt;
//                    tau = std::max(tau,3.0*dt1);

//               Real tem_new0 = tem_cool + (tem - tem_cool)*(1.0 - dt1/tau); 
//               Real tem_new = (tem_new0 >= tem_lim) ? tem_new0 : tem_lim;
      
//         // Step 4. Calculate new internal energy and pressure: eint_new = rho * T_new / (gamma-1), p_new = rho * T_new
//               Real p_new = rho * tem_new;
//               Real eint_new = p_new / (gam-1.0);
//         // Step 5. Calculate new total energy density: etot_new = eint_new + emag + ekin = (eint + emag + ekin) + (eint_new - eint)
//         //         --> etot_new = etot_old + (eint_new - eint)
//               Real etot_old = phydro->u(IEN,k,j,i);
//               Real eint = press / (gam-1.0);
//               Real etot_new = etot_old + (eint_new - eint);
//         // Step 6. Update p and etot: phydro->w(IPR,k,j,i) = pr_new, phydro->u(IEN,k,j,i) = etot_new
//               phydro->w(IPR,k,j,i) = p_new;
//               phydro->u(IEN,k,j,i) = etot_new;
//            }}
//          }}}
//      }

//     // artificial cooling
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
//       for (int i=is; i<=ie; ++i) {
//         Real &den= phydro->w(IDN,k,j,i);
//         Real &pr = phydro->w(IPR,k,j,i);
//         Real tem = pr/den;

//         Real den_lim_hot = rho_cool;

//         if(tem > tem_lim_hot && den < den_lim_hot){
//           // Backward Euler method is used
//           Real t_artrad = 2.0*dt1;
//           Real a1 = dt1/t_artrad;
//           Real b1 = 1.0/(1.0+a1);
//           tem = b1 * (tem + a1*tem_lim);
//           Real p_new = den * tem;
//           Real eint_new = p_new / (gam-1.0);
//           Real etot_old = phydro->u(IEN,k,j,i);
//           Real eint = pr / (gam-1.0);
//           Real etot_new = etot_old + (eint_new - eint);
//         // Step 6. Update p and etot: phydro->w(IPR,k,j,i) = pr_new, phydro->u(IEN,k,j,i) = etot_new
//           phydro->w(IPR,k,j,i) = p_new;
//           phydro->u(IEN,k,j,i) = etot_new;
//         }
//       }
//     }
//   }

// //  //Viscosity
// //  if(pmy_mesh->time >= time_AddingVis && flag_vis !=0 ){
// //
// //    AlphaViscosity(phdif,this,phydro->w,pfield->bc,is,ie,js,je,ks,ke);
// //
// //  }


  return;
}


/*void MeshBlock::UserWorkInLoop()
{

  int f3 = 0;
  if (block_size.nx3 > 1) f3=1;

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int kl = ks - f3*NGHOST;
  int ku = ke + f3*NGHOST;

  Real time1 = pmy_mesh->time;
  Real dt1 = pmy_mesh->dt;

  Real gam = peos->GetGamma();
  Real en = 1.0/(gam - 1.0);
  
  Real rmin = pmy_mesh->mesh_size.x1min;
  Real w_damp0 = f_w_damp * rmin;
  Real r_damp  = rmin + 0.5 * w_damp0;
  Real w_rad = 0.25 * w_damp0; // width of the transition region

  Real speed = sqrt(tem_center); // isothermal sound speed = 1
  Real t_cross = w_damp0 / speed;

  //  DampingLayer_NoVdamp(phydro,pcoord,dt1,rmin,r_damp,w_rad,t_cross,
  //il,iu,jl,ju,kl,ku);

  // Update the internal energy if INTERNAL_ENERGY_ENABLED is set
  if(INTERNAL_ENERGY_ENABLED){
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real ei = en*phydro->w(IPR,k,j,i);
//        phydro->u(IEI,k,j,i) = ei;
//        phydro->w(IEI,k,j,i) = ei;
      }
    }
  }}

  // Set boundary condition (this should be the same as sdbnd_iib)
  if(this->pbval->block_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")){
    BCInnerX1(this,pcoord,phydro->w,pfield->b,time1,dt1,is,ie,js,je,ks,ke,NGHOST);
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real bsq = SQR(pfield->bcc(IB1,k,j,i))
                 + SQR(pfield->bcc(IB2,k,j,i))
                 + SQR(pfield->bcc(IB3,k,j,i));

        const Real &den= phydro->w(IDN,k,j,i);
        const Real &v1 = phydro->w(IVX,k,j,i);
        const Real &v2 = phydro->w(IVY,k,j,i);
        const Real &v3 = phydro->w(IVZ,k,j,i);
        const Real &pr = phydro->w(IPR,k,j,i);

        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*v1;
        phydro->u(IM2,k,j,i) = den*v2;
        phydro->u(IM3,k,j,i) = den*v3;

        phydro->u(IEN,k,j,i) = 0.5 * den * (SQR(v1) + SQR(v2) + SQR(v3))
          + en * pr + 0.5 * bsq;
      }
    }
  }
  
  return;
}
*/

//start(shuto)
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
//   int f3 = 0;
//   if (block_size.nx3 > 1) f3=1;

//   // initialize avrg
//   for (int nv=0; nv<nav_uov; nv++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {
// #pragma omp simd
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//         avrg(nv,j,i) = 0.0 ;
//       }
//     }
//   }

//   // Calculate current density
//   pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,
//                                      is-1,ie+1,js-1,je+1,ks-f3,ke+f3);
//   Curl1(J1,this);
//   Curl2(J2,this);
//   Curl3(J3,this);

//   // Calculate vorticity (spherical coordinate)
//   Vorticity1(vor1,this);
//   Vorticity2(vor2,this);
//   Vorticity3(vor3,this);

//   // Make some quantities in a cylindrical coordinate
//   for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {
//       Real theta = pcoord->x2v(j);
// #pragma omp simd
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {

//         Real v_R   = sin(theta)*phydro->w(IVX,k,j,i) + cos(theta)*phydro->w(IVY,k,j,i) ;
//         Real v_phi = phydro->w(IVZ,k,j,i);
//         Real v_z   = cos(theta)*phydro->w(IVX,k,j,i) - sin(theta)*phydro->w(IVY,k,j,i) ;

//         Real B_R   = sin(theta)*pfield->bcc(IB1,k,j,i) + cos(theta)*pfield->bcc(IB2,k,j,i) ;
//         Real B_phi = pfield->bcc(IB3,k,j,i);
//         Real B_z   = cos(theta)*pfield->bcc(IB1,k,j,i) - sin(theta)*pfield->bcc(IB2,k,j,i) ;

//         // for R,z-component of induction eq
//         vzBR(k,j,i) = v_z * B_R;
//         vRBz(k,j,i) = v_R * B_z;
//         // for phi-component of induction eq
//         vpBz(k,j,i) = v_phi * B_z;
//         vzBp(k,j,i) = v_z * B_phi;
//         vRBp(k,j,i) = v_R * B_phi;
//         vpBR(k,j,i) = v_phi * B_R;
//       }
//     }
//   }

//   // Calculate derivatives
//   // for R-component of induction eq
//   Der_z(dvzBR_dz,vzBR,this);
//   Der_z(dvRBz_dz,vRBz,this);
//   // for z-component of induction eq
//   Der_R(dvzBR_dR,vzBR,this);
//   Der_R(dvRBz_dR,vRBz,this);
//   // for phi-component of induction eq  
//   Der_z(dvpBz_dz,vpBz,this);
//   Der_z(dvzBp_dz,vzBp,this);
//   Der_R(dvRBp_dR,vRBp,this);
//   Der_R(dvpBR_dR,vpBR,this);

//   // Prepare the output quantities
//   for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {

//       Real theta = pcoord->x2v(j);
// #pragma omp simd
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//         Real v_rad = phydro->w(IVX,k,j,i);
//         Real v_the = phydro->w(IVY,k,j,i);
//         Real v_phi = phydro->w(IVZ,k,j,i);

//         Real B_rad = pfield->bcc(IB1,k,j,i);
//         Real B_the = pfield->bcc(IB2,k,j,i);
//         Real B_phi = pfield->bcc(IB3,k,j,i);

//         Real J_rad = J1(k,j,i);
//         Real J_the = J2(k,j,i);
//         Real J_phi = J3(k,j,i);

//         Real v_R   = sin(theta)*v_rad + cos(theta)*v_the ;
//         Real v_z   = cos(theta)*v_rad - sin(theta)*v_the ;

//         Real B_R   = sin(theta)*B_rad + cos(theta)*B_the ;
//         Real B_z   = cos(theta)*B_rad - sin(theta)*B_the ;
//         Real B_p   = sqrt(B_R*B_R+B_z*B_z); // poloidal field strength

//         Real J_R   = sin(theta)*J_rad + cos(theta)*J_the ;
//         Real J_z   = cos(theta)*J_rad - sin(theta)*J_the ;

//         Real den = phydro->w(IDN,k,j,i);
//         Real pr  = phydro->w(IPR,k,j,i);

//         uov_tmp(0,k,j,i) = den; // No. 1
//         uov_tmp(1,k,j,i) = pr; // No. 2
//         uov_tmp(2,k,j,i) = pow(den,2); // No. 3

//         uov_tmp(3,k,j,i) = v_R;   // No. 4
//         uov_tmp(4,k,j,i) = v_phi; // No. 5
//         uov_tmp(5,k,j,i) = v_z;   // No. 6

//         uov_tmp(6,k,j,i) = den * v_R;   // No. 7
//         uov_tmp(7,k,j,i) = den * v_phi; // No. 8
//         uov_tmp(8,k,j,i) = den * v_z;   // No. 9

//         uov_tmp(9 ,k,j,i) = den * v_R * v_R;     // No. 10
//         uov_tmp(10,k,j,i) = den * v_R * v_phi;   // No. 11
//         uov_tmp(11,k,j,i) = den * v_R * v_z;     // No. 12
//         uov_tmp(12,k,j,i) = den * v_phi * v_phi; // No. 13
//         uov_tmp(13,k,j,i) = den * v_phi * v_z;   // No. 14
//         uov_tmp(14,k,j,i) = den * v_z * v_z;     // No. 15

//         uov_tmp(15,k,j,i) = B_R;   // No. 16
//         uov_tmp(16,k,j,i) = B_phi; // No. 17
//         uov_tmp(17,k,j,i) = B_z;   // No. 18
//         uov_tmp(18,k,j,i) = B_p;   // No. 19

//         uov_tmp(19,k,j,i) = B_R * B_R;     // No. 20
//         uov_tmp(20,k,j,i) = B_R * B_phi;   // No. 21
//         uov_tmp(21,k,j,i) = B_R * B_z;     // No. 22
//         uov_tmp(22,k,j,i) = B_phi * B_phi; // No. 23
//         uov_tmp(23,k,j,i) = B_phi * B_z;   // No. 24
//         uov_tmp(24,k,j,i) = B_z * B_z;     // No. 25

//         uov_tmp(25,k,j,i) = v_R * B_phi;     // No. 26
//         uov_tmp(26,k,j,i) = v_R * B_phi;     // No. 27
//         uov_tmp(27,k,j,i) = v_R * B_z;       // No. 28

//         uov_tmp(28,k,j,i) = v_phi * B_R;     // No. 29
//         uov_tmp(29,k,j,i) = v_phi * B_z;     // No. 30
//         uov_tmp(30,k,j,i) = v_phi * B_z;     // No. 31

//         uov_tmp(31,k,j,i) = v_z * B_R;       // No. 32
//         uov_tmp(32,k,j,i) = v_z * B_phi;     // No. 33
//         uov_tmp(33,k,j,i) = v_z * B_phi;     // No. 34

//         uov_tmp(34,k,j,i) = v_phi * B_p;     // No. 35

//         uov_tmp(35,k,j,i) = v_z * B_R * B_R;     // No. 36
//         uov_tmp(36,k,j,i) = v_z * B_phi * B_phi; // No. 37
//         uov_tmp(37,k,j,i) = v_R * B_R * B_z;     // No. 38
//         uov_tmp(38,k,j,i) = v_phi * B_phi * B_z; // No. 39

//         uov_tmp(39,k,j,i) = pr * v_z;  // No. 40
//         uov_tmp(40,k,j,i) = pr * v_z / den; // No. 41
//         uov_tmp(41,k,j,i) = pr / den; // No. 42

//         uov_tmp(42,k,j,i) = J_R;   // No. 43
//         uov_tmp(43,k,j,i) = J_phi; // No. 44
//         uov_tmp(44,k,j,i) = J_z;   // No. 45

//         uov_tmp(45,k,j,i) = J_R*B_phi; // No. 46
//         uov_tmp(46,k,j,i) = J_R*B_z;   // No. 47
//         uov_tmp(47,k,j,i) = J_phi*B_R; // No. 48
//         uov_tmp(48,k,j,i) = J_phi*B_z; // No. 49
//         uov_tmp(49,k,j,i) = J_z*B_R;   // No. 50
//         uov_tmp(50,k,j,i) = J_z*B_phi; // No. 51

//         Real vor_rad = vor1(k,j,i);
//         Real vor_the = vor2(k,j,i);
//         Real vor_phi = vor3(k,j,i);

//         uov_tmp(51,k,j,i) = vor_rad; // No. 52
//         uov_tmp(52,k,j,i) = vor_the; // No. 53
//         uov_tmp(53,k,j,i) = vor_phi; // No. 54

//         // for R-component of induction eq
//         uov_tmp(54,k,j,i) = -dvzBR_dz(k,j,i); // No. 55
//         uov_tmp(55,k,j,i) =  dvRBz_dz(k,j,i); // No. 56
//         // for z-component of induction eq
//         Real Rcyl = pcoord->x1v(i) * sin(theta);
//         uov_tmp(56,k,j,i) =   vzBR(k,j,i) / Rcyl - dvzBR_dR(k,j,i) ; // No. 57
//         uov_tmp(57,k,j,i) = -(vRBz(k,j,i) / Rcyl - dvRBz_dR(k,j,i)); // No. 58
//         // for phi-component of induction eq
//         uov_tmp(58,k,j,i) =  dvpBz_dz(k,j,i); // No. 59
//         uov_tmp(59,k,j,i) = -dvzBp_dz(k,j,i); // No. 60
//         uov_tmp(60,k,j,i) = -dvRBp_dR(k,j,i); // No. 61
//         uov_tmp(61,k,j,i) =  dvpBR_dR(k,j,i); // No. 62

//         uov_tmp(62,k,j,i) = v_rad; // No. 63
//         uov_tmp(63,k,j,i) = v_the; // No. 64

//         uov_tmp(64,k,j,i) = v_rad*vor_rad + v_the*vor_the + v_phi*vor_phi; // No. 65
//         uov_tmp(65,k,j,i) = B_rad*J_rad   + B_the*J_the   + B_phi*J_phi; // No. 66
//       }
//     }
//   }


//   // Sum data in the phi direction in this meshblock
//   for (int nv=0; nv<nav_uov; nv++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//         for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
//           avrg(nv,j,i) += uov_tmp(nv,k,j,i);
//         }
//       }
//     }
//   }

// #ifdef MPI_PARALLEL
//   if (f3 > 0) { // Only for 3D data
//   // Sum data of processes in gcomm
//   MPI_Allreduce(MPI_IN_PLACE, avrg.data(), avrg.GetSize(),
//         MPI_ATHENA_REAL, MPI_SUM, gcomm);

//   // Calculate the average values
//   int nx3 = (ke-ks)+1+2*NGHOST;
//   int nrbx3 = pmy_mesh->nrbx3; // meshblock number at the root level
//   int nbx3 = nrbx3 << (loc.level-pmy_mesh->root_level);

//   for (int nv=0; nv<nav_uov; nv++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {
// #pragma omp simd
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//         avrg(nv,j,i) /= (double) (nbx3 * nx3) ;
//       }
//     }
//   }
//   }
// #endif

//   // Pass the averaged data to user_out_var
//   // Phi average (No.1--nav_uov)
//   for (int nv=0; nv<nav_uov; nv++) {
//     for (int j=js-NGHOST; j<=je+NGHOST; j++) {
// #pragma omp simd
//       for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//         user_out_var(nv,ks,j,i) = avrg(nv,j,i);
//       }
//     }
//   }

//   // Other variables in user_out_var (3D kinetic/current helicity...)
//   if(flag_uov == 0){
//     for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
//       for (int j=js-NGHOST; j<=je+NGHOST; j++) {
// #pragma omp simd        
//         for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//           user_out_var(nav_uov  ,k,j,i) = uov_tmp(64,k,j,i); // No. nav_uov + 1 (kinetic helicity)
//           user_out_var(nav_uov+1,k,j,i) = uov_tmp(65,k,j,i); // No. nav_uov + 2 (current helicity)
//         }
//       }
//     }
//     }
//     // Ohmic (+ anomalous) resistivity outputed
//     if(flag_res == 1){
//     for (int k=ks-NGHOST*f3; k<=ke+NGHOST*f3; k++) {
//       for (int j=js-NGHOST; j<=je+NGHOST; j++) {
// #pragma omp simd
//         for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
//           user_out_var(nav_uov+2,k,j,i)
//             = pfield->fdif.etaB(0,k,j,i); // No. nav_uov + 3 (Ohmic eta)
//         }
//       }
//     }
//     }
  
  return;
}
//end(shuto)


//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: BCInnerX1

void BCInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
               Real time, Real dt,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  if(INTERNAL_ENERGY_ENABLED){
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        //prim(IVX,k,j,il-i) = 0.0; // zero vrad
        prim(IVX,k,j,il-i) = (prim(IVX,k,j,il) < 0.0) ? prim(IVX,k,j,il) : 0.0; // diode boundary
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il)*sqrt(pco->x1v(il)/pco->x1v(il-i));
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
//        prim(IEI,k,j,il-i) = prim(IEI,k,j,il);
      }
    }
  }
  } else {
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        //prim(IVX,k,j,il-i) = 0.0; // zero vrad
        prim(IVX,k,j,il-i) = (prim(IVX,k,j,il) < 0.0) ? prim(IVX,k,j,il) : 0.0; // diode boundary
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il)*sqrt(pco->x1v(il)/pco->x1v(il-i));
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
      }
    }
  }
  }
  if (MAGNETIC_FIELDS_ENABLED) { // free
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(il-i)) = b.x1f(k,j,il);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(il-i)) = b.x2f(k,j,il);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(il-i)) = b.x3f(k,j,il);
      }
    }}
  }
  
}

//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: BCOuterX1

void BCOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
               Real time, Real dt,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh) {


  if(INTERNAL_ENERGY_ENABLED){
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IVX,k,j,iu+i) = (prim(IVX,k,j,iu) > 0.0) ? prim(IVX,k,j,iu) : 0.0; // diode boundary
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu)*sqrt(pco->x1v(iu)/pco->x1v(iu+i));
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
//        prim(IEI,k,j,iu+i) = prim(IEI,k,j,iu);
      }
    }
  }
  } else {
  
  Real gam = pmb->peos->GetGamma();
  Real rho_inf = 1.0;
  Real tem_inf = 1.0/R_Bondi/gam;
  Real pr_inf = rho_inf * tem_inf; 
  Real c_inf = sqrt(gam*tem_inf);
  Real tmp = pow(2.0 / (5.0 - 3.0*gam),(5.0-3.0*gam)/(2.0*gam-2.0));
  Real Mdot_B = PI * tmp * R_Bondi * R_Bondi * rho_inf * c_inf; 
//  Real Vrad = 0.0037159977602672875;
  Real Vrad = -1.0 * Mdot_B / (4.0 * PI * pco->x1v(iu) * pco->x1v(iu)) / rho_inf;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,iu+i) = rho_inf;
        //prim(IVX,k,j,iu+i) = (prim(IVX,k,j,iu) > 0.0) ? prim(IVX,k,j,iu) : 0.0; // diode boundary
        prim(IVX,k,j,iu+i) = Vrad;
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        //prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu)*sqrt(pco->x1v(iu)/pco->x1v(iu+i));
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = pr_inf;
      }
    }
  }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(iu+i+1)) = b.x1f(k,j,(iu+1));
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(iu+i)) = b.x2f(k,j,iu);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(iu+i)) = b.x3f(k,j,iu);
      }
    }}
  }
  
}


// Damping layer without the velocity damp
void DampingLayer_NoVdamp(Hydro *phydro, Coordinates *pcoord,
			   Real dt1, Real rmin, Real r_damp, Real w_rad, Real t_cross,
			   int il, int iu, int jl, int ju, int kl, int ku)
{
  Real pr_center = den_center * tem_center;
  Real den_damp = f_den_damp * den_center;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real theta0 = pcoord->x2v(j);
      //for (int i=il; i<=iu; ++i) {
      for (int i=il; i<=iu-1; ++i) {
        Real r0 = pcoord->x1v(i);
        if(r0 < rmin + 2.0*w_rad){
         // Real &den = phydro->w(IDN,k,j,i);
         // Real &pr = phydro->w(IPR,k,j,i);
         // Real &den_p = phydro->w(IDN,k,j,i+1);
         // Real &pr_p = phydro->w(IPR,k,j,i+1);
         // tempt(k,j,i) = pr/den; // Save temperature
         // tempt(k,j,i+1) = pr_p/den_p;
          Real &vr = phydro->w(IVX,k,j,i);
          Real &vt = phydro->w(IVY,k,j,i);
          Real &vp = phydro->w(IVZ,k,j,i);
          
          // Density update
          //Real t_damp1 = MAX(MIN(f_damp_min*den/den_center,f_damp_max),f_damp_min) * t_cross;
          Real t_damp1 = dt1;
          // MAX(t_damp1,dt1); // Not to violate CFL condition
          Real t_damp1i_r = 1.0 / t_damp1 * 0.5 * (1.0 - tanh((r0-r_damp)/w_rad));

          // Backward Euler method is used
          Real a1 = dt1*t_damp1i_r;
          Real b1 = 1.0/(1.0+a1);

          Real r1 = r0*sin(theta0);
          Real vr0 = 0.0;
          Real vt0 = 0.0;
          Real vp0 = 1./sqrt(r0);
          if( vr > 0 ){
          vr = b1 * (vr + a1*vr0);
          }
          vt = b1 * (vt + a1*vt0);
          vp = b1 * (vp + a1*vp0);	  


          //den = b1 * (den + a1*den_center);
	  
          //pressure update
          //Real ent = pr/pow(den,5./3.);
          //Real ent_p = pr_p/pow(den_p,5./3.);
          //Real dent = ent_p - ent; // Entropy gradient before the update
          //Real dent_sign = (Real) SIGN1(dent);

          //Real fac_den = 0.5 * (tanh((den-den_damp)/den_damp*10.) + 1.0);

          //Real fac_ent0 = 0.5 * (dent_sign + 1.0);
          //Real fac_ent1 = 1.0 + (fac_ent0 - 1.0)*fac_den;
          //Real a2 = fac_ent1 * a1 ;
          //Real b2 = 1.0/(1.0+a2);
          
          //pr = b2 * (pr + a2*pr_center);
          //pr = (pr > pressure_floor) ? pr : pressure_floor;
        }
      }
    }
  }

  return;
}

// anomalous resistivity
void Diffusivity(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                   const AthenaArray<Real> &bmag, const int is, const int ie,
                   const int js, const int je, const int ks, const int ke)
{
  MeshBlock* pmy_block = pfdif->pmy_block;
  Field *pfield = pmy_block->pfield;
  Coordinates *pcoord = pmy_block->pcoord;

  Real gam = pmb->peos->GetGamma();
  Real theta1 = 0.05235987756; // 3deg
  Real theta2 = 3.0892327761; // 177deg

  Real r_res = 3.0;
  Real w_res = 0.1;
  //Real alpha = 0.1;
  Real tem = tem_cool;
  Real Cs = sqrt(gam*tem);
  //Real den_eta = 1e4;
  //Real eta_in = 0.1;//plasmabeta=1e4
  //Real eta_in = 0.01;//plasmabeta=1e5
  //Real eta_in = ;//plasmabeta=1e6
  // dx2 = r*dtheta = 1 * (pi/256) = 0.0123
  // Rm = t_diff / t_A ~ 1
  // t_diff = dx2^2 / eta_in
  // t_A = dx2/V_A = dx2 / (B/sqrt(rho)) = dx2*sqrt(rho)/B
  // t_diff ~ t_A --> dx2^2 / eta_in = dx2 * sqrt(rho)/B --> eta_in = dx2 * B / sqrt(rho) = 0.0123 * 0.01 / sqrt(10) = 3e-5

  int f3=0;
  if (pmy_block->block_size.nx3 > 1) f3=1;

  for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        Real theta = pcoord->x2v(j);
#pragma omp simd
        for(int i=is; i<=ie; i++){
          Real radius = pcoord->x1v(i);
          Real &den = pmb->phydro->w(IDN,k,j,i);
          //Real &pr = phydro->w(IPR,k,j,i);
          //Real tem = 0.001;
          //Real Cs = sqrt(gamma*tem);
          Real Rcyl  = radius * sin(theta) + 1e-6; // Cylindrical radius
          Real vK = sqrt(1.0/Rcyl);
          Real omg = vK/Rcyl;
          Real Hp = Cs/omg;
          Real eta_disk0 = alpha * Cs * Hp;

          Real &etaB = pfdif->etaB(FieldDiffusion::DiffProcess::ohmic,k,j,i); // etaB will be activated/created only when eta_ohm is given in input file.

          if(pmb->pmy_mesh->time >= time_AddingB){
              // etaB = pfdif->eta_ohm; // give a uniform resistivity
              etaB = eta_in * 0.5 * (-tanh((radius - r_res)/w_res)+1.0) + eta_disk0 * 0.5 * (tanh((den - den_eta)/(den_eta*w_res))+1.0);
              // etaB = pfdif->eta_ohm; // give a uniform resistivity
              // etaB += eta_in * 0.5 * (-tanh((radius - r_res)/w_res)+1.0);
              // set zero resistivity around the poles
              etaB = ((theta1 < theta) && (theta < theta2)) ? etaB : 0.0;
              etaB = (den > den_res_lim) ? etaB : 0.0;
          }
        }
      }
  }
  return;
}


void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                    const AthenaArray<Real> &bc, int is, int ie, int js, int je, int ks, int ke)
{
  //MeshBlock* pmy_block = pmb->pmy_block;
  //Field *pfield = pmy_block->pfield;
  Coordinates *pcoord = pmb->pcoord;

  Real gam = pmb->peos->GetGamma();

  Real w_res = 0.1;
  //Real alpha = 0.01;
  Real tem = tem_cool; //tem_cool
  Real Cs = sqrt(gam*tem);
  //Real den_eta = 1e4;

  int f3=0;
  //if (pmy_block->block_size.nx3 > 1) f3=1;
  if (pmb->block_size.nx3 > 1) f3=1;

  for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        Real theta = pcoord->x2v(j);
#pragma omp simd
        for(int i=is; i<=ie; i++){
          Real radius = pcoord->x1v(i);
          Real &den = pmb->phydro->w(IDN,k,j,i);
          //Real &pr = phydro->w(IPR,k,j,i);
          //Real tem = 0.001;
          //Real Cs = sqrt(gamma*tem);
          Real Rcyl  = radius * sin(theta) + 1e-6; // Cylindrical radius
          Real vK = sqrt(1.0/Rcyl);
          Real omg = vK/Rcyl;
          Real Hp = Cs/omg;
          Real nu_disk = alpha * Cs * Hp;

          Real &nu = phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i); // nu will be activated/created only when nu_iso is given in input file.
          if(pmb->pmy_mesh->time >= time_AddingVis){
             nu = nu_disk * 0.5 * (tanh((den - den_eta)/(den_eta*w_res))+1.0);
             nu = (den > den_res_lim) ? nu : 0.0;
          }
        } 
      }
    }
  return;
}
