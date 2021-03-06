      SUBROUTINE BMG2_SymStd_UTILS_matvec( &
     &                       K, SO, QF, Q, II, JJ,&
     &                       KF, IFD, NStncl&
     &                       ) BIND(C, NAME='BMG2_SymStd_UTILS_matvec')

! ======================================================================
!  --------------------
!   DESCRIPTION:
!  --------------------
!
!     BMG_SymStd_UTILS_matvec computes the matrix-vector product
!     for the CG routine using the BoxMG stencil.
!
! ======================================================================
! $license_flag$
! ======================================================================
!  --------------------
!   INPUT:
!  --------------------
!
!     K         K is the grid number.
!     SO        Refer to BMG2_SymStd_SOLVE_boxmg
!     QF        Refer to BMG2_SymStd_SOLVE_boxmg
!     RES       Refer to BMG2_SymStd_SOLVE_boxmg
!     II        Number of grid points in x direction, including
!               two fictitious points.
!     JJ        Number of grid points in y direction, including
!               two fictitious points.
!     KF        index of the finest grid
!     IFD       Refer to BMG2_SymStd_SOLVE_boxmg
!     IRELAX    Refer to BMG2_SymStd_SOLVE_boxmg
!
! ======================================================================
!  --------------------
!   INPUT/OUTPUT:
!  --------------------
!
!     Q         Refer to BOXMG.
!
! ======================================================================
!  --------------------
!   OUTPUT:
!  --------------------
!
!
!
! ======================================================================
!  --------------------
!   LOCAL:
!  --------------------
!
!
! ======================================================================
      USE ModInterface
      IMPLICIT NONE

! -----------------------------
!     Includes
!
      INCLUDE 'mpif.h'
      INCLUDE 'MSG_f90.h'


      INCLUDE 'BMG_constants_f90.h'
      INCLUDE 'BMG_stencils_f90.h'
      INCLUDE 'BMG_workspace_f90.h'

! ----------------------------
!     Argument Declarations
!
      INTEGER(len_t), VALUE :: II, JJ
      INTEGER(C_INT), VALUE :: NStncl, IFD, K, KF
      REAL(real_t) :: Q(II,JJ), QF(II,JJ), SO(II+1,JJ+1,NStncl)

! ----------------------------
!     Local Declarations
!
      INTEGER I, I1, J, J1

! ======================================================================

      J1=JJ-1
      I1=II-1

!     ------------------------------------------------------------------


      IF ( K.LT.KF .OR. IFD.NE.1 ) THEN
         !
         !  9-point stencil
         !

         DO J=2,J1
            DO I=2,I1
               QF(I,J) =  SO(I  ,J  ,KO )*Q(I  ,J)&
     &                  - SO(I  ,J  ,KW )*Q(I-1,J)&
     &                  - SO(I+1,J  ,KW )*Q(I+1,J)&
     &                  - SO(I  ,J  ,KS )*Q(I  ,J-1)&
     &                  - SO(I  ,J+1,KS )*Q(I  ,J+1)&
     &                  - SO(I  ,J  ,KSW)*Q(I-1,J-1)&
     &                  - SO(I+1,J  ,KNW)*Q(I+1,J-1)&
     &                  - SO(I  ,J+1,KNW)*Q(I-1,J+1)&
     &                  - SO(I+1,J+1,KSW)*Q(I+1,J+1)
            ENDDO
         ENDDO

         !
      ELSE
         !
         !  5-point stencil
         !

         DO J=2,J1
            DO I=2,I1
               QF(I,J) =  SO(I  ,J  ,KO)*Q(I  ,J)&
     &                  - SO(I  ,J  ,KW)*Q(I-1,J)&
     &                  - SO(I+1,J  ,KW)*Q(I+1,J)&
     &                  - SO(I  ,J  ,KS)*Q(I  ,J-1)&
     &                  - SO(I  ,J+1,KS)*Q(I  ,J+1)
            ENDDO
         ENDDO
         !
      ENDIF

! ======================================================================

      RETURN
      END
