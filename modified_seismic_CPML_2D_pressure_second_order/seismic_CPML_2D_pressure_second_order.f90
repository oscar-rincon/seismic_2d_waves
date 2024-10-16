!
! SEISMIC_CPML Version 1.1.3, July 2018.
!
! Copyright CNRS, France.
! Contributor: Dimitri Komatitsch, komatitsch aT lma DOT cnrs-mrs DOT fr
!
! This software is a computer program whose purpose is to solve
! the two-dimensional heterogeneous isotropic acoustic wave equation
! using a finite-difference method with Convolutional Perfectly Matched
! Layer (C-PML) conditions.
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".

  program seismic_CPML_2D_pressure

! 2D acoustic finite-difference code in pressure formulation
! with Convolutional-PML (C-PML) absorbing conditions for an heterogeneous isotropic acoustic medium

! Dimitri Komatitsch, CNRS, Marseille, July 2018.

! The pressure wave equation in an inviscid heterogeneous fluid is:
!
! 1/Kappa d2p / dt2 = div(grad(p) / rho) = d(1/rho dp/dx)/dx + d(1/rho dp/dy)/dy
!
! (see for instance Komatitsch and Tromp, Geophysical Journal International, vol. 149, p. 390-412 (2002), equations (19) and (21))
!
! The second-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used:
!
!            ^ y
!            |
!            |
!
!            +-------------------+
!            |                   |
!            |                   |
!            |                   |
!            |                   |
!            |                   |
!      dp/dy +---------+         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            +---------+---------+  ---> x
!            p       dp/dx
!

! The C-PML implementation is based in part on formulas given in Roden and Gedney (2000).
! If you use this code for your own research, please cite some (or all) of these
! articles:
!
! @ARTICLE{MaKoEz08,
! author = {Roland Martin and Dimitri Komatitsch and Abdela\^aziz Ezziani},
! title = {An unsplit convolutional perfectly matched layer improved at grazing
! incidence for seismic wave equation in poroelastic media},
! journal = {Geophysics},
! year = {2008},
! volume = {73},
! pages = {T51-T61},
! number = {4},
! doi = {10.1190/1.2939484}}
!
! @ARTICLE{MaKo09,
! author = {Roland Martin and Dimitri Komatitsch},
! title = {An unsplit convolutional perfectly matched layer technique improved
! at grazing incidence for the viscoelastic wave equation},
! journal = {Geophysical Journal International},
! year = {2009},
! volume = {179},
! pages = {333-344},
! number = {1},
! doi = {10.1111/j.1365-246X.2009.04278.x}}
!
! @ARTICLE{MaKoGe08,
! author = {Roland Martin and Dimitri Komatitsch and Stephen D. Gedney},
! title = {A variational formulation of a stabilized unsplit convolutional perfectly
! matched layer for the isotropic or anisotropic seismic wave equation},
! journal = {Computer Modeling in Engineering and Sciences},
! year = {2008},
! volume = {37},
! pages = {274-304},
! number = {3}}
!
! @ARTICLE{KoMa07,
! author = {Dimitri Komatitsch and Roland Martin},
! title = {An unsplit convolutional {P}erfectly {M}atched {L}ayer improved
!          at grazing incidence for the seismic wave equation},
! journal = {Geophysics},
! year = {2007},
! volume = {72},
! number = {5},
! pages = {SM155-SM167},
! doi = {10.1190/1.2757586}}
!
! The original CPML technique for Maxwell's equations is described in:
!
! @ARTICLE{RoGe00,
! author = {J. A. Roden and S. D. Gedney},
! title = {Convolution {PML} ({CPML}): {A}n Efficient {FDTD} Implementation
!          of the {CFS}-{PML} for Arbitrary Media},
! journal = {Microwave and Optical Technology Letters},
! year = {2000},
! volume = {27},
! number = {5},
! pages = {334-339},
! doi = {10.1002/1098-2760(20001205)27:5 < 334::AID-MOP14>3.0.CO;2-A}}

!
! To display the 2D results as color images, use:
!
!   " display image*.gif " or " gimp image*.gif "
!
! or
!
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vx*.gif allfiles_Vx.gif "
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vy*.gif allfiles_Vy.gif "
!   then " display allfiles_Vx.gif " or " gimp allfiles_Vx.gif "
!   then " display allfiles_Vy.gif " or " gimp allfiles_Vy.gif "
!

! IMPORTANT : all our CPML codes work fine in single precision as well (which is significantly faster).
!             If you want you can thus force automatic conversion to single precision at compile time
!             or change all the declarations and constants in the code from double precision to single.

  implicit none
  character(len=100) :: file_name
! flags to add PML layers to the edges of the grid
  logical, parameter :: USE_PML_XMIN = .true.
  logical, parameter :: USE_PML_XMAX = .true.
  logical, parameter :: USE_PML_YMIN = .true.
  logical, parameter :: USE_PML_YMAX = .true.

! total number of grid points in each direction of the grid
  integer, parameter :: NX = 300
  integer, parameter :: NY = 300

! size of a grid cell
  double precision, parameter :: DELTAX = 5.0d0
  double precision, parameter :: DELTAY = DELTAX

! thickness of the PML layer in grid points
  integer, parameter :: NPOINTS_PML = 5

! P-velocity and density
! the unrelaxed value is the value at frequency = 0 (the relaxed value would be the value at frequency = +infinity)
  double precision, parameter :: cp_unrelaxed = 2500.d0
  double precision, parameter :: density = 2200.d0

! total number of time steps
  integer, parameter :: NSTEP = 8001

! time step in seconds
  double precision, parameter :: DELTAT = 0.0001!5.0d-4!5.2d-4

! parameters for the source
  double precision, parameter :: f0 = 20.d0 
  double precision, parameter :: t0 = 1.20d0 / f0
  double precision, parameter :: factor = 1.d4

! source (in pressure)
  !double precision, parameter :: xsource = 750.d0
  !double precision, parameter :: ysource = 750.d0
  !integer, parameter :: ISOURCE = xsource / DELTAX + 1
  ! integer, parameter :: JSOURCE = ysource / DELTAY + 1
  double precision, parameter :: ysource = 100.d0
  integer, parameter :: JSOURCE =10!ysource / DELTAY + 1
  integer, parameter :: ISOURCE_START = 1
  integer, parameter :: ISOURCE_END = NX-1

! receivers
  integer, parameter :: NREC = 1
!! DK DK I use 2301 here instead of 2300 in order to fall exactly on a grid point
  double precision, parameter :: xdeb = 800.d0   ! first receiver x in meters
  double precision, parameter :: ydeb = 750.d0   ! first receiver y in meters
  double precision, parameter :: xfin = 800.d0   ! last receiver x in meters
  double precision, parameter :: yfin = 750.d0   ! last receiver y in meters

! display information on the screen from time to time
  integer, parameter :: IT_DISPLAY = 125

! value of PI
  double precision, parameter :: PI = 3.141592653589793238462643d0

! zero
  double precision, parameter :: ZERO = 0.d0

! large value for maximum
  double precision, parameter :: HUGEVAL = 1.d+30

! threshold above which we consider that the code became unstable
  double precision, parameter :: STABILITY_THRESHOLD = 1.d+25

! main arrays
  double precision, dimension(NX,NY) :: pressure_past,pressure_present,pressure_future, &
      pressure_xx,pressure_yy,dpressurexx_dx,dpressureyy_dy,kappa_unrelaxed,rho,Kronecker_source

! to interpolate material parameters or velocity at the right location in the staggered grid cell
  double precision :: rho_half_x,rho_half_y

! power to compute d0 profile
  double precision, parameter :: NPOWER = 2.d0

! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
  double precision, parameter :: K_MAX_PML = 1.d0
  double precision, parameter :: ALPHA_MAX_PML = 2.d0*PI*(f0/2.d0) ! from Festa and Vilotte

! arrays for the memory variables
! could declare these arrays in PML only to save a lot of memory, but proof of concept only here
  double precision, dimension(NX,NY) :: &
      memory_dpressure_dx, &
      memory_dpressure_dy, &
      memory_dpressurexx_dx, &
      memory_dpressureyy_dy

  double precision :: &
      value_dpressure_dx, &
      value_dpressure_dy, &
      value_dpressurexx_dx, &
      value_dpressureyy_dy

! 1D arrays for the damping profiles
  double precision, dimension(NX) :: d_x,K_x,alpha_x,a_x,b_x,d_x_half,K_x_half,alpha_x_half,a_x_half,b_x_half
  double precision, dimension(NY) :: d_y,K_y,alpha_y,a_y,b_y,d_y_half,K_y_half,alpha_y_half,a_y_half,b_y_half

  double precision :: thickness_PML_x,thickness_PML_y,xoriginleft,xoriginright,yoriginbottom,yorigintop
  double precision :: Rcoef,d0_x,d0_y,xval,yval,abscissa_in_PML,abscissa_normalized

! for the source
  double precision :: a,t,source_term

! for receivers
  double precision xspacerec,yspacerec,distval,dist
  integer, dimension(NREC) :: ix_rec,iy_rec
  double precision, dimension(NREC) :: xrec,yrec
  integer :: myNREC

! for seismograms
  double precision, dimension(NSTEP,NREC) :: sispressure

  integer :: i,j,it,irec

  double precision :: Courant_number,pressurenorm

  double precision, parameter :: x_center = NX / 2.0
  double precision, parameter :: y_center = NY / 2.0
  double precision, parameter :: radius = 50
  double precision, parameter :: kappa_inside = 0.0!density * cp_unrelaxed * cp_unrelaxed / 2.0  ! Different kappa inside the circle

 
 
!---
!--- program starts here
!---
print *, 'JSOURCE = ', JSOURCE
  print *
  print *,'2D acoustic finite-difference code in pressure formulation with C-PML'
  print *

! display size of the model
  print *
  print *,'NX = ',NX
  print *,'NY = ',NY
  print *
  print *,'size of the model along X = ',(NX - 1) * DELTAX
  print *,'size of the model along Y = ',(NY - 1) * DELTAY
  print *
  print *,'Total number of grid points = ',NX * NY
  print *

!--- define profile of absorption in PML region

! thickness of the PML layer in meters
  thickness_PML_x = NPOINTS_PML * DELTAX
  thickness_PML_y = NPOINTS_PML * DELTAY

! reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
  Rcoef = 0.0001d0!0.001d0

! check that NPOWER is okay
  if (NPOWER < 1) stop 'NPOWER must be greater than 1'

! compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
  d0_x = - (NPOWER + 1) * cp_unrelaxed * log(Rcoef) / (2.d0 * thickness_PML_x)
  d0_y = - (NPOWER + 1) * cp_unrelaxed * log(Rcoef) / (2.d0 * thickness_PML_y)

  print *,'d0_x = ',d0_x
  print *,'d0_y = ',d0_y
  print *

  d_x(:) = ZERO
  d_x_half(:) = ZERO
  K_x(:) = 1.d0
  K_x_half(:) = 1.d0
  alpha_x(:) = ZERO
  alpha_x_half(:) = ZERO
  a_x(:) = ZERO
  a_x_half(:) = ZERO

  d_y(:) = ZERO
  d_y_half(:) = ZERO
  K_y(:) = 1.d0
  K_y_half(:) = 1.d0
  alpha_y(:) = ZERO
  alpha_y_half(:) = ZERO
  a_y(:) = ZERO
  a_y_half(:) = ZERO

! damping in the X direction

! origin of the PML layer (position of right edge minus thickness, in meters)
  xoriginleft = thickness_PML_x
  xoriginright = (NX-1)*DELTAX - thickness_PML_x

  do i = 1,NX

! abscissa of current grid point along the damping profile
    xval = DELTAX * dble(i-1)

!---------- left edge
    if (USE_PML_XMIN) then

! define damping profile at the grid points
      abscissa_in_PML = xoriginleft - xval
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = xoriginleft - (xval + DELTAX/2.d0)
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x_half(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

!---------- right edge
    if (USE_PML_XMAX) then

! define damping profile at the grid points
      abscissa_in_PML = xval - xoriginright
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = xval + DELTAX/2.d0 - xoriginright
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x_half(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

! just in case, for -5 at the end
    if (alpha_x(i) < ZERO) alpha_x(i) = ZERO
    if (alpha_x_half(i) < ZERO) alpha_x_half(i) = ZERO

    b_x(i) = exp(- (d_x(i) / K_x(i) + alpha_x(i)) * DELTAT)
    b_x_half(i) = exp(- (d_x_half(i) / K_x_half(i) + alpha_x_half(i)) * DELTAT)

! this to avoid division by zero outside the PML
    if (abs(d_x(i)) > 1.d-6) a_x(i) = d_x(i) * (b_x(i) - 1.d0) / (K_x(i) * (d_x(i) + K_x(i) * alpha_x(i)))
    if (abs(d_x_half(i)) > 1.d-6) a_x_half(i) = d_x_half(i) * &
      (b_x_half(i) - 1.d0) / (K_x_half(i) * (d_x_half(i) + K_x_half(i) * alpha_x_half(i)))

  enddo

! damping in the Y direction

! origin of the PML layer (position of right edge minus thickness, in meters)
  yoriginbottom = thickness_PML_y
  yorigintop = (NY-1)*DELTAY - thickness_PML_y

  do j = 1,NY

! abscissa of current grid point along the damping profile
    yval = DELTAY * dble(j-1)

!---------- bottom edge
    if (USE_PML_YMIN) then

! define damping profile at the grid points
      abscissa_in_PML = yoriginbottom - yval
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.d0)
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y_half(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

!---------- top edge
    if (USE_PML_YMAX) then

! define damping profile at the grid points
      abscissa_in_PML = yval - yorigintop
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = yval + DELTAY/2.d0 - yorigintop
      if (abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y_half(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

    b_y(j) = exp(- (d_y(j) / K_y(j) + alpha_y(j)) * DELTAT)
    b_y_half(j) = exp(- (d_y_half(j) / K_y_half(j) + alpha_y_half(j)) * DELTAT)

! this to avoid division by zero outside the PML
    if (abs(d_y(j)) > 1.d-6) a_y(j) = d_y(j) * (b_y(j) - 1.d0) / (K_y(j) * (d_y(j) + K_y(j) * alpha_y(j)))
    if (abs(d_y_half(j)) > 1.d-6) a_y_half(j) = d_y_half(j) * &
      (b_y_half(j) - 1.d0) / (K_y_half(j) * (d_y_half(j) + K_y_half(j) * alpha_y_half(j)))

  enddo

! ! compute the Lame parameter and density
!   do j = 1,NY
!     do i = 1,NX
!       rho(i,j) = density
!       kappa_unrelaxed(i,j) = density*cp_unrelaxed*cp_unrelaxed
!     enddo
!   enddo

! compute the Lame parameter and density
do j = 1, NY
  do i = 1, NX
    rho(i, j) = density
    ! Check if the point (i, j) is inside the circle
    if ((i - x_center)**2 + (j - y_center)**2 <= radius**2) then
      kappa_unrelaxed(i, j) = kappa_inside
    else
      kappa_unrelaxed(i, j) = density * cp_unrelaxed * cp_unrelaxed
    end if
  end do
end do


 
 

! ! print position of the source
!   print *,'Position of the source:'
!   print *
!   print *,'x = ',xsource
!   print *,'y = ',ysource
!   print *

! ! define location of the source
!   Kronecker_source(:,:) = 0.d0
!   Kronecker_source(ISOURCE,JSOURCE) = 1.d0
! define location of the source along the full central line
Kronecker_source(:,:) = 0.d0
do j = 1, NX
    Kronecker_source(JSOURCE, j) = 1.d0
end do

! define location of receivers
  print *,'There are ',nrec,' receivers'
  print *
  if (NREC > 1) then
! this is to avoid a warning with GNU gfortran at compile time about division by zero when NREC = 1
    myNREC = NREC
    xspacerec = (xfin-xdeb) / dble(myNREC-1)
    yspacerec = (yfin-ydeb) / dble(myNREC-1)
  else
    xspacerec = 0.d0
    yspacerec = 0.d0
  endif
  do irec=1,nrec
    xrec(irec) = xdeb + dble(irec-1)*xspacerec
    yrec(irec) = ydeb + dble(irec-1)*yspacerec
  enddo

! find closest grid point for each receiver
  do irec=1,nrec
    dist = HUGEVAL
    do j = 1,NY
    do i = 1,NX
      distval = sqrt((DELTAX*dble(i-1) - xrec(irec))**2 + (DELTAY*dble(j-1) - yrec(irec))**2)
      if (distval < dist) then
        dist = distval
        ix_rec(irec) = i
        iy_rec(irec) = j
      endif
    enddo
    enddo
    print *,'receiver ',irec,' x_target,y_target = ',xrec(irec),yrec(irec)
    print *,'closest grid point found at distance ',dist,' in i,j = ',ix_rec(irec),iy_rec(irec)
    print *
  enddo

! check the Courant stability condition for the explicit time scheme
! R. Courant et K. O. Friedrichs et H. Lewy (1928)
  Courant_number = cp_unrelaxed * DELTAT * sqrt(1.d0/DELTAX**2 + 1.d0/DELTAY**2)
  print *,'Courant number is ',Courant_number
  print *
  if (Courant_number > 2.d0) stop 'time step is too large, simulation will be unstable'

! suppress old files (can be commented out if "call system" is missing in your compiler)
  call system('rm -f Vx_*.dat Vy_*.dat image*.pnm image*.gif')

! initialize arrays
  pressure_present(:,:) = ZERO
  pressure_past(:,:) = ZERO

! PML
  memory_dpressure_dx(:,:) = ZERO
  memory_dpressure_dy(:,:) = ZERO
  memory_dpressurexx_dx(:,:) = ZERO
  memory_dpressureyy_dy(:,:) = ZERO

! initialize seismograms
  sispressure(:,:) = ZERO

!---
!---  beginning of time loop
!---

  do it = 1,NSTEP

! compute the first spatial derivatives divided by density

    do j = 1,NY
      do i = 1,NX-1
      value_dpressure_dx = (pressure_present(i+1,j) - pressure_present(i,j)) / DELTAX

      memory_dpressure_dx(i,j) = b_x_half(i) * memory_dpressure_dx(i,j) + a_x_half(i) * value_dpressure_dx

      value_dpressure_dx = value_dpressure_dx / K_x_half(i) + memory_dpressure_dx(i,j)

      rho_half_x = 0.5d0 * (rho(i+1,j) + rho(i,j))
      pressure_xx(i,j) = value_dpressure_dx / rho_half_x
      enddo
    enddo

    do j = 1,NY-1
      do i = 1,NX
      value_dpressure_dy = (pressure_present(i,j+1) - pressure_present(i,j)) / DELTAY

      memory_dpressure_dy(i,j) = b_y_half(j) * memory_dpressure_dy(i,j) + a_y_half(j) * value_dpressure_dy

      value_dpressure_dy = value_dpressure_dy / K_y_half(j) + memory_dpressure_dy(i,j)

      rho_half_y = 0.5d0 * (rho(i,j+1) + rho(i,j))
      pressure_yy(i,j) = value_dpressure_dy / rho_half_y
      enddo
    enddo

! compute the second spatial derivatives

    do j = 1,NY
      do i = 2,NX
      value_dpressurexx_dx = (pressure_xx(i,j) - pressure_xx(i-1,j)) / DELTAX

      memory_dpressurexx_dx(i,j) = b_x(i) * memory_dpressurexx_dx(i,j) + a_x(i) * value_dpressurexx_dx

      value_dpressurexx_dx = value_dpressurexx_dx / K_x(i) + memory_dpressurexx_dx(i,j)

      dpressurexx_dx(i,j) = value_dpressurexx_dx
      enddo
    enddo

    do j = 2,NY
      do i = 1,NX
      value_dpressureyy_dy = (pressure_yy(i,j) - pressure_yy(i,j-1)) / DELTAY

      memory_dpressureyy_dy(i,j) = b_y(j) * memory_dpressureyy_dy(i,j) + a_y(j) * value_dpressureyy_dy

      value_dpressureyy_dy = value_dpressureyy_dy / K_y(j) + memory_dpressureyy_dy(i,j)

      dpressureyy_dy(i,j) = value_dpressureyy_dy
      enddo
    enddo

! add the source (pressure located at a given grid point)
  a = pi*pi*f0*f0
  t = dble(it-1)*DELTAT

! Gaussian
! source_term = - factor * exp(-a*(t-t0)**2) / (2.d0 * a)

! first derivative of a Gaussian
! source_term = factor * (t-t0)*exp(-a*(t-t0)**2)

! Ricker source time function (second derivative of a Gaussian)
  source_term = factor * (1.d0 - 2.d0*a*(t-t0)**2)*exp(-a*(t-t0)**2)

! apply the time evolution scheme
! we apply it everywhere, including at some points on the edges of the domain that have not be calculated above,
! which is of course wrong (or more precisely undefined), but this does not matter because these values
! will be erased by the Dirichlet conditions set on these edges below
  pressure_future(:,:) = - pressure_past(:,:) + 2.d0 * pressure_present(:,:) + &
                                  DELTAT*DELTAT * ((dpressurexx_dx(:,:) + dpressureyy_dy(:,:)) * kappa_unrelaxed(:,:) + &
                                  4.d0 * PI * cp_unrelaxed**2 * source_term * Kronecker_source(:,:))

! apply Dirichlet conditions at the bottom of the C-PML layers,
! which is the right condition to implement in order for C-PML to remain stable at long times

! Dirichlet condition for pressure on the left boundary
  pressure_future(1,:) = ZERO

! Dirichlet condition for pressure on the right boundary
  pressure_future(NX,:) = ZERO

! Dirichlet condition for pressure on the bottom boundary
  pressure_future(:,1) = ZERO

! Dirichlet condition for pressure on the top boundary
  pressure_future(:,NY) = ZERO

! store seismograms
  do irec = 1,NREC
    sispressure(it,irec) = pressure_future(ix_rec(irec),iy_rec(irec))
  enddo

! output information
  if (mod(it,IT_DISPLAY) == 0 .or. it == 5) then

! print maximum of pressure and of norm of velocity
    pressurenorm = maxval(abs(pressure_future))
    print *,'Time step # ',it,' out of ',NSTEP
    print *,'Time: ',sngl((it-1)*DELTAT),' seconds'
    print *,'Max absolute value of pressure = ',pressurenorm
    print *
! check stability of the code, exit if unstable
    if (pressurenorm > STABILITY_THRESHOLD) stop 'code became unstable and blew up'

    !call create_color_image(pressure_future,NX,NY,it,ISOURCE,JSOURCE,ix_rec,iy_rec,nrec, &
    !                     NPOINTS_PML,USE_PML_XMIN,USE_PML_XMAX,USE_PML_YMIN,USE_PML_YMAX,3)

    ! Guardar pressure_future en un archivo externo
    write(file_name, "('results/pressure_future_', I0, '.txt')") it
    call save_pressure_future(pressure_future, NX, NY, file_name)
  endif



! move new values to old values (the present becomes the past, the future becomes the present)
  pressure_past(:,:) = pressure_present(:,:)
  pressure_present(:,:) = pressure_future(:,:)

  enddo   ! end of the time loop

! Crear el nombre del archivo de salida
file_name = 'results/kappa_unrelaxed.txt'

! Llamar a la subrutina para guardar kappa_unrelaxed
call save_kappa_unrelaxed(kappa_unrelaxed, NX, NY, file_name)

! save seismograms
  call write_seismograms(sispressure,NSTEP,NREC,DELTAT,t0)

  print *
  print *,'End of the simulation'
  print *

  end program seismic_CPML_2D_pressure

!----
!----  save the seismograms in ASCII text format
!----

  subroutine write_seismograms(sispressure,nt,nrec,DELTAT,t0)

  implicit none

  integer nt,nrec
  double precision DELTAT,t0

  double precision sispressure(nt,nrec)

  integer irec,it

  character(len=100) file_name

! pressure
  do irec=1,nrec
    write(file_name,"('results/pressure_file_',i3.3,'.dat')") irec
    open(unit=11,file=file_name,status='unknown')
    do it=1,nt
!     write(11,*) sngl(dble(it-1)*DELTAT - t0),' ',sngl(sispressure(it,irec))
      write(11,*) sngl(dble(it-1)*DELTAT - t0 + DELTAT/2.d0),' ',sngl(sispressure(it,irec))
!     write(11,*) sngl(dble(it-1)*DELTAT - DELTAT - t0),' ',sngl(sispressure(it,irec))
    enddo
    close(11)
  enddo

  end subroutine write_seismograms

!----
!----  routine to create a color image of a given vector component
!----  the image is created in PNM format and then converted to GIF
!----

  subroutine create_color_image(image_data_2D,NX,NY,it,ISOURCE,JSOURCE,ix_rec,iy_rec,nrec, &
              NPOINTS_PML,USE_PML_XMIN,USE_PML_XMAX,USE_PML_YMIN,USE_PML_YMAX,field_number)

  implicit none

! non linear display to enhance small amplitudes for graphics
  double precision, parameter :: POWER_DISPLAY = 0.30d0

! amplitude threshold above which we draw the color point
  double precision, parameter :: cutvect = 0.01d0

! use black or white background for points that are below the threshold
  logical, parameter :: WHITE_BACKGROUND = .true.

! size of cross and square in pixels drawn to represent the source and the receivers
  integer, parameter :: width_cross = 5, thickness_cross = 1, size_square = 3

  integer NX,NY,it,field_number,ISOURCE,JSOURCE,NPOINTS_PML,nrec
  logical USE_PML_XMIN,USE_PML_XMAX,USE_PML_YMIN,USE_PML_YMAX

  double precision, dimension(NX,NY) :: image_data_2D

  integer, dimension(nrec) :: ix_rec,iy_rec

  integer :: ix,iy,irec

  character(len=100) :: file_name,system_command

  integer :: R, G, B

  double precision :: normalized_value,max_amplitude

! open image file and create system command to convert image to more convenient format
! use the "convert" command from ImageMagick http://www.imagemagick.org
  ! if (field_number == 1) then
  !   write(file_name,"('image',i6.6,'_Vx.pnm')") it
  !   write(system_command,"('convert image',i6.6,'_Vx.pnm image',i6.6,'_Vx.gif ; rm image',i6.6,'_Vx.pnm')") it,it,it
  ! else if (field_number == 2) then
  !   write(file_name,"('image',i6.6,'_Vy.pnm')") it
  !   write(system_command,"('convert image',i6.6,'_Vy.pnm image',i6.6,'_Vy.gif ; rm image',i6.6,'_Vy.pnm')") it,it,it
  ! else if (field_number == 3) then
  !   write(file_name,"('image',i6.6,'_pressure.pnm')") it
  !   write(system_command,"('convert image',i6.6,'_pressure.pnm image',i6.6,'_pressure.gif ; rm image',i6.6,'_pressure.pnm')") &
  !                              it,it,it
  ! endif
  
  open(unit=27, file=file_name, status='unknown')

  write(27,"('P3')") ! write image in PNM P3 format

  write(27,*) NX,NY ! write image size
  write(27,*) '255' ! maximum value of each pixel color

! compute maximum amplitude
  max_amplitude = maxval(abs(image_data_2D))

! image starts in upper-left corner in PNM format
  do iy=NY,1,-1
    do ix=1,NX

! define data as vector component normalized to [-1:1] and rounded to nearest integer
! keeping in mind that amplitude can be negative
    normalized_value = image_data_2D(ix,iy) / max_amplitude

! suppress values that are outside [-1:+1] to avoid small edge effects
    if (normalized_value < -1.d0) normalized_value = -1.d0
    if (normalized_value > 1.d0) normalized_value = 1.d0

! draw an orange cross to represent the source
    if ((ix >= ISOURCE - width_cross .and. ix <= ISOURCE + width_cross .and. &
        iy >= JSOURCE - thickness_cross .and. iy <= JSOURCE + thickness_cross) .or. &
       (ix >= ISOURCE - thickness_cross .and. ix <= ISOURCE + thickness_cross .and. &
        iy >= JSOURCE - width_cross .and. iy <= JSOURCE + width_cross)) then
      R = 255
      G = 157
      B = 0

! display two-pixel-thick black frame around the image
  else if (ix <= 2 .or. ix >= NX-1 .or. iy <= 2 .or. iy >= NY-1) then
      R = 0
      G = 0
      B = 0

! display edges of the PML layers
  else if ((USE_PML_XMIN .and. ix == NPOINTS_PML) .or. &
          (USE_PML_XMAX .and. ix == NX - NPOINTS_PML) .or. &
          (USE_PML_YMIN .and. iy == NPOINTS_PML) .or. &
          (USE_PML_YMAX .and. iy == NY - NPOINTS_PML)) then
      R = 255
      G = 150
      B = 0

! suppress all the values that are below the threshold
    else if (abs(image_data_2D(ix,iy)) <= max_amplitude * cutvect) then

! use a black or white background for points that are below the threshold
      if (WHITE_BACKGROUND) then
        R = 255
        G = 255
        B = 255
      else
        R = 0
        G = 0
        B = 0
      endif

! represent regular image points using red if value is positive, blue if negative
    else if (normalized_value >= 0.d0) then
      R = nint(255.d0*normalized_value**POWER_DISPLAY)
      G = 0
      B = 0
    else
      R = 0
      G = 0
      B = nint(255.d0*abs(normalized_value)**POWER_DISPLAY)
    endif

! draw a green square to represent the receivers
  do irec = 1,nrec
    if ((ix >= ix_rec(irec) - size_square .and. ix <= ix_rec(irec) + size_square .and. &
        iy >= iy_rec(irec) - size_square .and. iy <= iy_rec(irec) + size_square) .or. &
       (ix >= ix_rec(irec) - size_square .and. ix <= ix_rec(irec) + size_square .and. &
        iy >= iy_rec(irec) - size_square .and. iy <= iy_rec(irec) + size_square)) then
! use dark green color
      R = 30
      G = 180
      B = 60
    endif
  enddo

! write color pixel
    write(27,"(i3,' ',i3,' ',i3)") R,G,B

    enddo
  enddo

! close file
  close(27)

! call the system to convert image to Gif (can be commented out if "call system" is missing in your compiler)
  call system(system_command)

  end subroutine create_color_image

  subroutine save_pressure_future(pressure_future, NX, NY, file_name)
    implicit none
    integer, intent(in) :: NX, NY
    double precision, dimension(NX, NY), intent(in) :: pressure_future
    character(len=*), intent(in) :: file_name

    integer :: i, j
    open(unit=27, file=file_name, status='unknown')
    do j = 1, NY
      do i = 1, NX
        write(27, *) i, j, pressure_future(i, j)
      end do
    end do
    close(27)
  end subroutine save_pressure_future

subroutine save_kappa_unrelaxed(kappa_unrelaxed, NX, NY, file_name)
  implicit none
  integer, intent(in) :: NX, NY
  double precision, dimension(NX, NY), intent(in) :: kappa_unrelaxed
  character(len=*), intent(in) :: file_name

  integer :: i, j
  ! Abrir el archivo para escritura
  open(unit=27, file=file_name, status='unknown')
  
  ! Escribir los índices y los datos de kappa_unrelaxed en el archivo
  do j = 1, NY
    do i = 1, NX
      write(27, *) i, j, kappa_unrelaxed(i, j)
    end do
  end do
  
  ! Cerrar el archivo
  close(27)
end subroutine save_kappa_unrelaxed