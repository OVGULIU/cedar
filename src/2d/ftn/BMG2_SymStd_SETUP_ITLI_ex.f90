      SUBROUTINE BMG2_SymStd_SETUP_ITLI_ex(&
     &                SO, SOC, CI,&
     &                IIF, JJF, IIC, JJC, IFD, NStncl,IPN&
     &                ) BIND(C, NAME='BMG2_SymStd_SETUP_ITLI_ex')

! ======================================================================
!  --------------------
!   DESCRIPTION:
!  --------------------
!
!     BMG2_SymStd_SETUP_cg_ITLI constructs the Galerkin (variational)
!     coarse-grid operator on the coasre grid, KC, given the fine-grid
!     stencil, SO, and interpolation stencil, CI, on the fine-grid, KF.
!
! ======================================================================
! $license_flag$
! ======================================================================
!  --------------------
!   INPUT:
!  --------------------
!
!
! ======================================================================
!  --------------------
!   INPUT/OUTPUT:
!  --------------------
!
!
! ======================================================================
!  --------------------
!   OUTPUT:
!  --------------------
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
      INCLUDE 'BMG_constants_f90.h'
      INCLUDE 'BMG_stencils_f90.h'
      INCLUDE 'BMG_parameters_f90.h'

! ---------------------------
!    Argument Declarations:
!
      integer(len_t), value :: IIC, IIF, JJC, JJF
      integer(c_int), value :: IFD, NStncl, IPN
      real(real_t) :: CI(IIC,JJC,8), SO(IIF,JJF,NStncl), SOC(IIC,JJC,5)

! --------------------------
!     Local Declarations:
!
      integer(len_t) :: IC, I, IIC1, IICF, IICF1, IICF2, IIF1
      integer(len_t) :: JC, J, JJC1, JJCF, JJCF1, JJCF2, JJF1
      integer :: PER_x, PER_y, PER_xy
      real(real_t) :: CE, CEA, CENW, CFNW, CN, CNE, CNW, CO, COA, CONW,&
     &          COSW, CS, CSA, CSE, CSEA, CSENW, CSNW, CSWA, CSSW, CSW,&
     &          CSWSW, CW, CWA, CWSW

! ======================================================================

! -----------------------------------
!     Useful indexing bounds:
! -----------------------------------

      IIC1=IIC-1
      JJC1=JJC-1

      IIF1=IIF-1
      JJF1=JJF-1

      IICF=(IIF-2)/2+3
      JJCF=(JJF-2)/2+3
      IICF1=IICF-1
      JJCF1=JJCF-1
      IICF2=IICF-2
      JJCF2=JJCF-2

      IPN = IABS(IPN)
      PER_x = IABS(BMG_BCs_def_per_x)
      PER_y = IABS(BMG_BCs_def_per_y)
      PER_xy = IABS(BMG_BCs_def_per_xy)

      !IF ( IFD.NE.1 .OR. KF.LT.NOG ) THEN
      IF ( IFD.NE.1 ) THEN

!******************************
!     begin computation of grid kc difference operator when kf differenc
!     operator is nine point unless kc. ge. NOG
!
         J=0
         DO JC=2,JJC1
            J=J+2
            I=0
            DO IC=2,IIC1
               I=I+2
               CO = SO(I,J+1,KNW)*CI(IC,JC+1,LSW)&
     &              + SO(I,J,KW)*CI(IC,JC,LL)&
     &              + SO(I,J,KSW)*CI(IC,JC,LNW)
               CS=SO(I,J-1,KW)*CI(IC,JC,LNW)+SO(I,J,KNW)&
     &              *CI(IC,JC,LL)
               CSW=-SO(I-1,J-1,KO)*CI(IC,JC,LNW)&
     &              +SO(I-1,J-1,KW)*CI(IC-1,JC,LA)+SO(I-1,J,KNW)&
     &              +SO(I-1,J,KS)*CI(IC,JC,LL)
               CW=-SO(I-1,J,KO)*CI(IC,JC,LL)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LNW)+SO(I-1,J,KSW)*CI(IC-1,JC,LA)&
     &              +SO(I-1,J,KW)+SO(I-1,J+1,KNW)*CI(IC-1,JC+1,LB)&
     &              +SO(I-1,J+1,KS)*CI(IC,JC+1,LSW)
               CNW=-SO(I-1,J+1,KO)*CI(IC,JC+1,LSW)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC,LL)+SO(I-1,J+1,KSW)+SO(I-1,J+1,KW)&
     &              *CI(IC-1,JC+1,LB)
               CN=SO(I,J+1,KSW)*CI(IC,JC,LL)+SO(I,J+1,KW)&
     &              *CI(IC,JC+1,LSW)
               SOC(IC,JC,KW)=CO+CI(IC,JC,LA)*CS+CI(IC,JC,LNE)*CSW&
     &              +CI(IC,JC,LR)*CW+CI(IC,JC+1,LSE)*CNW&
     &              +CI(IC,JC+1,LB)*CN
               COSW=SO(I,J,KSW)*CI(IC,JC,LSW)
               CSSW=SO(I,J-1,KSW)*CI(IC,JC-1,LL)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LSW)
               CSWSW=-SO(I-1,J-1,KO)*CI(IC,JC,LSW)&
     &              +SO(I-1,J-1,KS)*CI(IC,JC-1,LL)+SO(I-1,J-1,KSW)&
     &              +SO(I-1,J-1,KW)*CI(IC-1,JC,LB)
               CWSW=SO(I-1,J,KS)*CI(IC,JC,LSW)+SO(I-1,J,KSW)&
     &              *CI(IC-1,JC,LB)
               SOC(IC,JC,KSW)=COSW+CI(IC,JC,LA)*CSSW+CI(IC,JC,LNE)&
     &              *CSWSW+CI(IC,JC,LR)*CWSW
               COA=SO(I,J,KSW)*CI(IC,JC,LSE)+SO(I,J,KS)&
     &              *CI(IC,JC,LB)+SO(I+1,J,KNW)*CI(IC+1,JC,LSW)
               CEA=SO(I+1,J,KS)*CI(IC+1,JC,LSW)+SO(I+1,J,KSW)&
     &              *CI(IC,JC,LB)
               CSEA=-SO(I+1,J-1,KO)*CI(IC+1,JC,LSW)+SO(I+1,J-1,KS)&
     &              *CI(IC+1,JC-1,LL)+SO(I+1,J-1,KSW)+SO(I+1,J-1,KW)&
     &              *CI(IC,JC,LB)
               CSA=-SO(I,J-1,KO)*CI(IC,JC,LB)+SO(I+1,J-1,KW)&
     &              *CI(IC+1,JC,LSW)+SO(I+1,J-1,KNW)*CI(IC+1,JC-1,LL)&
     &              +SO(I,J-1,KS)+SO(I,J-1,KSW)*CI(IC,JC-1,LR)&
     &              +SO(I,J-1,KW)*CI(IC,JC,LSE)
               CSWA=-SO(I-1,J-1,KO)*CI(IC,JC,LSE)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LB)+SO(I,J-1,KNW)+SO(I-1,J-1,KS)&
     &              *CI(IC,JC-1,LR)
               CWA=SO(I,J,KNW)*CI(IC,JC,LB)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LSE)
               SOC(IC,JC,KS)=COA+CI(IC+1,JC,LL)*CEA+CI(IC+1,JC,LNW)&
     &              *CSEA+CI(IC,JC,LA)*CSA+CI(IC,JC,LNE)&
     &              *CSWA+CI(IC,JC,LR)*CWA

               CONW=SO(I-1,J,KNW)*CI(IC,JC,LSE)
               CENW=SO(I,J,KNW)*CI(IC,JC,LB)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LSE)
               CSENW=-SO(I-1,J-1,KO)*CI(IC,JC,LSE)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LB)+SO(I,J-1,KNW)+SO(I-1,J-1,KS)&
     &              *CI(IC,JC-1,LR)
               CFNW=SO(I-1,J-1,KW)*CI(IC,JC,LSE)+SO(I-1,J-1,KNW)&
     &              *CI(IC,JC-1,LR)
               SOC(IC,JC,KNW)=CONW+CI(IC,JC,LL)*CENW+CI(IC,JC,LNW)&
     &              *CSENW+CI(IC-1,JC,LA)*CFNW
            ENDDO
         ENDDO

         J=0
         DO JC=2,JJC1
            J=J+2
            I=0
            DO IC=2,IIC1
               I=I+2
               CO=SO(I,J,KW)*CI(IC,JC,LR)+SO(I,J+1,KNW)&
     &              *CI(IC,JC+1,LSE)+SO(I,J+1,KS)*CI(IC,JC+1,LB)&
     &              +SO(I+1,J+1,KSW)*CI(IC+1,JC+1,LSW)+SO(I+1,J,KW)&
     &              *CI(IC+1,JC,LL)+SO(I+1,J,KNW)*CI(IC+1,JC,LNW)&
     &              +SO(I,J,KS)*CI(IC,JC,LA)+SO(I,J,KSW)&
     &              *CI(IC,JC,LNE)-SO(I,J,KO)
               CW=-SO(I-1,J,KO)*CI(IC,JC,LR)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC+1,LSE)+SO(I,J+1,KSW)*CI(IC,JC+1,LB)&
     &              +SO(I,J,KW)+SO(I,J,KNW)*CI(IC,JC,LA)&
     &              +SO(I-1,J,KS)*CI(IC,JC,LNE)
               CNW=-SO(I-1,J+1,KO)*CI(IC,JC+1,LSE)+SO(I,J+1,KW)&
     &              *CI(IC,JC+1,LB)+SO(I,J+1,KNW)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC,LR)
               CN=-SO(I,J+1,KO)*CI(IC,JC+1,LB)+SO(I+1,J+1,KNW)&
     &              *CI(IC+1,JC,LL)+SO(I,J+1,KS)+SO(I,J+1,KSW)&
     &              *CI(IC,JC,LR)+SO(I,J+1,KW)*CI(IC,JC+1,LSE)&
     &              +SO(I+1,J+1,KW)*CI(IC+1,JC+1,LSW)
               CNE=-SO(I+1,J+1,KO)*CI(IC+1,JC+1,LSW)+SO(I+1,J+1,KS)&
     &              *CI(IC+1,JC,LL)+SO(I+1,J+1,KSW)+SO(I+1,J+1,KW)&
     &              *CI(IC,JC+1,LB)
               CE=-SO(I+1,J,KO)*CI(IC+1,JC,LL)+SO(I+1,J,KS)&
     &              *CI(IC+1,JC,LNW)+SO(I+1,J,KSW)*CI(IC,JC,LA)&
     &              +SO(I+1,J,KW)+SO(I+1,J+1,KNW)*CI(IC,JC+1,LB)&
     &              +SO(I+1,J+1,KS)*CI(IC+1,JC+1,LSW)
               CSE=-SO(I+1,J-1,KO)*CI(IC+1,JC,LNW)+SO(I+1,J-1,KW)&
     &              *CI(IC,JC,LA)+SO(I+1,J,KNW)+SO(I+1,J,KS)&
     &              *CI(IC+1,JC,LL)
               CS=-SO(I,J-1,KO)*CI(IC,JC,LA)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LNE)+SO(I,J,KNW)*CI(IC,JC,LR)&
     &              +SO(I,J,KS)+SO(I+1,J,KSW)*CI(IC+1,JC,LL)&
     &              +SO(I+1,J-1,KW)*CI(IC+1,JC,LNW)
               CSW=-SO(I-1,J-1,KO)*CI(IC,JC,LNE)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LR)+SO(I,J,KSW)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LA)
               SOC(IC,JC,KO)=-CI(IC,JC+1,LSE)*CNW-CI(IC,JC+1,LB)*CN&
     &              -CI(IC+1,JC+1,LSW)*CNE-CI(IC,JC,LR)*CW-CO&
     &              -CI(IC+1,JC,LL)*CE-CI(IC,JC,LNE)*CSW&
     &              -CI(IC,JC,LA)*CS-CI(IC+1,JC,LNW)*CSE
            ENDDO
         ENDDO

      IF(IPN.EQ.BMG_BCs_definite .OR. IPN.EQ.BMG_BCs_indef_nonper)&
     & RETURN

         IF(IPN.EQ.PER_y .OR. IPN.EQ.PER_xy) THEN
            DO IC=1,IIC
               SOC(IC,JJC,KS)=SOC(IC,2,KS)
               SOC(IC,JJC,KW)=SOC(IC,2,KW)
               SOC(IC,JJC,KNW)=SOC(IC,2,KNW)
               SOC(IC,JJC,KSW)=SOC(IC,2,KSW)
               SOC(IC,JJC,KO)=SOC(IC,2,KO)
               SOC(IC,1,KO)=SOC(IC,JJC1,KO)
               SOC(IC,1,KW)=SOC(IC,JJC1,KW)
               SOC(IC,1,KS)=SOC(IC,JJC1,KS)
               SOC(IC,1,KSW)=SOC(IC,JJC1,KSW)
               SOC(IC,1,KNW)=SOC(IC,JJC1,KNW)
            ENDDO
         ENDIF
         IF(IPN.EQ.PER_x .OR. IPN.EQ.PER_xy) THEN
            DO JC=1,JJC
               SOC(IIC,JC,KW)=SOC(2,JC,KW)
               SOC(IIC,JC,KS)=SOC(2,JC,KS)
               SOC(IIC,JC,KNW)=SOC(2,JC,KNW)
               SOC(IIC,JC,KSW)=SOC(2,JC,KSW)
               SOC(IIC,JC,KO)=SOC(2,JC,KO)
               SOC(1,JC,KW)=SOC(IIC1,JC,KW)
               SOC(1,JC,KS)=SOC(IIC1,JC,KS)
               SOC(1,JC,KSW)=SOC(IIC1,JC,KSW)
               SOC(1,JC,KNW)=SOC(IIC1,JC,KNW)
               SOC(1,JC,KO)=SOC(IIC1,JC,KO)
            ENDDO
         ENDIF

!   end of computation of kc difference operator when kf difference
!   operator is nine point
!******************************

      ELSE

!******************************
!   begin computation of kc difference operator when kf difference
!   operator is five point unless kc.ge.icoef
!
         J=0
         DO JC=2,JJC1
            J=J+2
            I=0
            DO IC=2,IIC1
               I=I+2
               CO=SO(I,J,KW)*CI(IC,JC,LL)
               CS=SO(I,J-1,KW)*CI(IC,JC,LNW)
               CSW=-SO(I-1,J-1,KO)*CI(IC,JC,LNW)+SO(I-1,J-1,KW)&
     &              *CI(IC-1,JC,LA)+SO(I-1,J,KS)*CI(IC,JC,LL)
               CW=-SO(I-1,J,KO)*CI(IC,JC,LL)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LNW)+SO(I-1,J,KW)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC+1,LSW)
               CNW=-SO(I-1,J+1,KO)*CI(IC,JC+1,LSW)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC,LL)+SO(I-1,J+1,KW)*CI(IC-1,JC+1,LB)
               CN=SO(I,J+1,KW)*CI(IC,JC+1,LSW)
               SOC(IC,JC,KW)=CO+CI(IC,JC,LA)*CS+CI(IC,JC,LNE)*CSW&
     &              +CI(IC,JC,LR)*CW+CI(IC,JC+1,LSE)*CNW&
     &              +CI(IC,JC+1,LB)*CN
               CSSW=SO(I,J-1,KW)*CI(IC,JC,LSW)
               CSWSW=-SO(I-1,J-1,KO)*CI(IC,JC,LSW)&
     &              +SO(I-1,J-1,KS)*CI(IC,JC-1,LL)+SO(I-1,J-1,KW)&
     &              *CI(IC-1,JC,LB)
               CWSW=SO(I-1,J,KS)*CI(IC,JC,LSW)
               SOC(IC,JC,KSW)=CI(IC,JC,LA)*CSSW+CI(IC,JC,LNE)&
     &              *CSWSW+CI(IC,JC,LR)*CWSW
               COA=SO(I,J,KS)*CI(IC,JC,LB)
               CEA=SO(I+1,J,KS)*CI(IC+1,JC,LSW)
               CSEA=-SO(I+1,J-1,KO)*CI(IC+1,JC,LSW)+SO(I+1,J-1,KS)&
     &              *CI(IC+1,JC-1,LL)+SO(I+1,J-1,KW)*CI(IC,JC,LB)
               CSA=-SO(I,J-1,KO)*CI(IC,JC,LB)+SO(I+1,J-1,KW)&
     &              *CI(IC+1,JC,LSW)+SO(I,J-1,KS)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LSE)
               CSWA=-SO(I-1,J-1,KO)*CI(IC,JC,LSE)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LB)+SO(I-1,J-1,KS)*CI(IC,JC-1,LR)
               CWA=SO(I-1,J,KS)*CI(IC,JC,LSE)
               SOC(IC,JC,KS)=COA+CI(IC+1,JC,LL)*CEA+CI(IC+1,JC,LNW)&
     &              *CSEA+CI(IC,JC,LA)*CSA+CI(IC,JC,LNE)*CSWA&
     &              +CI(IC,JC,LR)*CWA
               CENW=SO(I-1,J,KS)*CI(IC,JC,LSE)
               CSENW=-SO(I-1,J-1,KO)*CI(IC,JC,LSE)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LB)+SO(I-1,J-1,KS)*CI(IC,JC-1,LR)
               CSNW=SO(I-1,J-1,KW)*CI(IC,JC,LSE)
               SOC(IC,JC,KNW)=CI(IC,JC,LL)*CENW+CI(IC,JC,LNW)*CSENW&
     &              +CI(IC-1,JC,LA)*CSNW
               CO=SO(I,J,KW)*CI(IC,JC,LR)+SO(I,J+1,KS)&
     &              *CI(IC,JC+1,LB)+SO(I+1,J,KW)*CI(IC+1,JC,LL)&
     &              +SO(I,J,KS)*CI(IC,JC,LA)-SO(I,J,KO)
               CW=-SO(I-1,J,KO)*CI(IC,JC,LR)+SO(I-1,J+1,KS)&
     &              *CI(IC,JC+1,LSE)+SO(I,J,KW)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LNE)
               CNW=-SO(I-1,J+1,KO)*CI(IC,JC+1,LSE)+SO(I,J+1,KW)&
     &              *CI(IC,JC+1,LB)+SO(I-1,J+1,KS)*CI(IC,JC,LR)
               CN=-SO(I,J+1,KO)*CI(IC,JC+1,LB)+SO(I,J+1,KS)&
     &              +SO(I,J+1,KW)*CI(IC,JC+1,LSE)+SO(I+1,J+1,KW)&
     &              *CI(IC+1,JC+1,LSW)
               CNE=-SO(I+1,J+1,KO)*CI(IC+1,JC+1,LSW)+SO(I+1,J+1,KS)&
     &              *CI(IC+1,JC,LL)+SO(I+1,J+1,KW)*CI(IC,JC+1,LB)
               CE=-SO(I+1,J,KO)*CI(IC+1,JC,LL)+SO(I+1,J,KS)&
     &              *CI(IC+1,JC,LNW)+SO(I+1,J,KW)+SO(I+1,J+1,KS)&
     &              *CI(IC+1,JC+1,LSW)
               CSE=-SO(I+1,J-1,KO)*CI(IC+1,JC,LNW)+SO(I+1,J-1,KW)&
     &              *CI(IC,JC,LA)+SO(I+1,J,KS)*CI(IC+1,JC,LL)
               CS=-SO(I,J-1,KO)*CI(IC,JC,LA)+SO(I,J-1,KW)&
     &              *CI(IC,JC,LNE)+SO(I,J,KS)+SO(I+1,J-1,KW)&
     &              *CI(IC+1,JC,LNW)
               CSW=-SO(I-1,J-1,KO)*CI(IC,JC,LNE)+SO(I-1,J,KS)&
     &              *CI(IC,JC,LR)+SO(I,J-1,KW)*CI(IC,JC,LA)
               SOC(IC,JC,KO)=-CI(IC,JC+1,LSE)*CNW-CI(IC,JC+1,LB)*CN&
     &              -CI(IC+1,JC+1,LSW)*CNE-CI(IC,JC,LR)*CW-CO&
     &              -CI(IC+1,JC,LL)*CE-CI(IC,JC,LNE)*CSW&
     &              -CI(IC,JC,LA)*CS-CI(IC+1,JC,LNW)*CSE
            ENDDO
         ENDDO

      IF(IPN.EQ.BMG_BCs_definite .OR. IPN.EQ.BMG_BCs_indef_nonper)&
     & RETURN

         IF(IPN.EQ.PER_y .OR. IPN.EQ.PER_xy) THEN
            DO IC=1,IIC
               SOC(IC,JJC,KS)=SOC(IC,2,KS)
               SOC(IC,JJC,KW)=SOC(IC,2,KW)
               SOC(IC,JJC,KNW)=SOC(IC,2,KNW)
               SOC(IC,JJC,KSW)=SOC(IC,2,KSW)
               SOC(IC,JJC,KO)=SOC(IC,2,KO)
               SOC(IC,1,KO)=SOC(IC,JJC1,KO)
               SOC(IC,1,KW)=SOC(IC,JJC1,KW)
               SOC(IC,1,KS)=SOC(IC,JJC1,KS)
               SOC(IC,1,KSW)=SOC(IC,JJC1,KSW)
               SOC(IC,1,KNW)=SOC(IC,JJC1,KNW)
            ENDDO
         ENDIF
         IF(IPN.EQ.PER_x .OR. IPN.EQ.PER_xy) THEN
            DO JC=1,JJC
               SOC(IIC,JC,KW)=SOC(2,JC,KW)
               SOC(IIC,JC,KS)=SOC(2,JC,KS)
               SOC(IIC,JC,KNW)=SOC(2,JC,KNW)
               SOC(IIC,JC,KSW)=SOC(2,JC,KSW)
               SOC(IIC,JC,KO)=SOC(2,JC,KO)
               SOC(1,JC,KW)=SOC(IIC1,JC,KW)
               SOC(1,JC,KS)=SOC(IIC1,JC,KS)
               SOC(1,JC,KSW)=SOC(IIC1,JC,KSW)
               SOC(1,JC,KNW)=SOC(IIC1,JC,KNW)
               SOC(1,JC,KO)=SOC(IIC1,JC,KO)
            ENDDO
         ENDIF
      ENDIF

!$$$      WRITE(*,*) 'FOR KC = ',KC
!$$$      DO JC = 2,JJC1
!$$$         WRITE(*,*) 'FOR JC ',JC
!$$$         DO IC = 2,IIC1
!$$$            WRITE(*,*) 'FOR IC ',IC
!$$$            WRITE(*,*) -SOC(IC,JC+1,KNW),-SOC(IC,JC+1,KS),
!$$$     &           -SOC(IC+1,JC+1,KSW)
!$$$            WRITE(*,*) -SOC(IC,JC,KW),SOC(IC,JC,KO),-SOC(IC+1,JC,KW)
!$$$            WRITE(*,*) -SOC(IC,JC,KSW),-SOC(IC,JC,KS),
!$$$     &           -SOC(IC+1,JC,KNW)
!$$$            WRITE(*,*) -SOC(IC,JC+1,KNW)-SOC(IC,JC+1,KS)
!$$$     &           -SOC(IC+1,JC+1,KSW)
!$$$     &        -SOC(IC,JC,KW)+SOC(IC,JC,KO)-SOC(IC+1,JC,KW)
!$$$     &        -SOC(IC,JC,KSW)-SOC(IC,JC,KS)
!$$$     &           -SOC(IC+1,JC,KNW)
!$$$         ENDDO
!$$$      ENDDO

!   end of computation of grid kc difference operator, when kf
!   difference operator is five point
!******************************

      RETURN
      END
