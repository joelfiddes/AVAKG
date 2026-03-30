"""
Numba-accelerated replacements for the performance-critical Cython kernels
in avaframe.com1DFA (DFAfunctionsCython / DFAToolsCython).

Usage
-----
    from pipeline_steps.com1dfa_fast import patch_avaframe
    patch_avaframe()          # monkey-patches the Cython module
    # then run com1DFA as usual

Ported functions
----------------
Helper (DFAToolsCython):
    norm, norm2, normalize, crossProd, scalProd,
    getCells, getWeights, getCellAndWeights,
    getScalar, getVector, SamosATfric,
    samosProjectionIteratrive, reprojectVelocity

Kernels (DFAfunctionsCython):
    _compute_force_sph_kernel   (computeGradC inner loop, SPH options 1 & 2)
    _compute_force_kernel       (computeForceC inner loop, samosAT friction)
    _update_fields_kernel       (updateFieldsC inner loop)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numba import njit, prange

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Helper functions  (from DFAToolsCython)
# ---------------------------------------------------------------------------

@njit(cache=True)
def norm(x, y, z):
    """Euclidean norm of (x, y, z)."""
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True)
def norm2(x, y, z):
    """Squared Euclidean norm of (x, y, z)."""
    return x * x + y * y + z * z


@njit(cache=True)
def normalize(x, y, z):
    """Return unit vector in direction of (x, y, z).
    If the vector is zero-length the original values are returned (matches
    Cython behaviour where the output variables are left uninitialised when
    norme == 0)."""
    norme = math.sqrt(x * x + y * y + z * z)
    if norme > 0.0:
        return x / norme, y / norme, z / norme
    return x, y, z


@njit(cache=True)
def crossProd(ux, uy, uz, vx, vy, vz):
    """Cross product u x v."""
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx
    return wx, wy, wz


@njit(cache=True)
def scalProd(ux, uy, uz, vx, vy, vz):
    """Scalar (dot) product u . v."""
    return ux * vx + uy * vy + uz * vz


@njit(cache=True)
def getCells(x, y, ncols, nrows, csz):
    """Return flat index of the lower-left grid cell containing (x, y).
    Returns -1 if the point is outside the domain."""
    Lx = x / csz
    Ly = y / csz
    Lx0 = int(math.floor(Lx))
    Ly0 = int(math.floor(Ly))
    if Lx0 < 0 or Ly0 < 0 or Lx0 + 1 >= ncols or Ly0 + 1 >= nrows:
        return -1
    return Ly0 * ncols + Lx0


@njit(cache=True)
def getWeights(x, y, iCell, csz, ncols, interpOption):
    """Bilinear interpolation weights for a point relative to the lower-left
    cell *iCell*.  interpOption: 0=nearest, 1=equal, 2=bilinear."""
    Lx0 = iCell % ncols
    Ly0 = iCell // ncols
    Lx = x / csz
    Ly = y / csz
    dx = Lx - Lx0
    dy = Ly - Ly0

    if interpOption == 0:
        dx = float(round(dx))
        dy = float(round(dy))
    elif interpOption == 1:
        dx = 0.5
        dy = 0.5

    w0 = (1.0 - dx) * (1.0 - dy)
    w1 = dx * (1.0 - dy)
    w2 = (1.0 - dx) * dy
    w3 = dx * dy
    return w0, w1, w2, w3


@njit(cache=True)
def getCellAndWeights(x, y, ncols, nrows, csz, interpOption):
    """Return (Lx0, Ly0, iCell, w0, w1, w2, w3)."""
    iCell = getCells(x, y, ncols, nrows, csz)
    if iCell < 0:
        return 0, 0, -1, 0.0, 0.0, 0.0, 0.0
    w0, w1, w2, w3 = getWeights(x, y, iCell, csz, ncols, interpOption)
    Lx0 = iCell % ncols
    Ly0 = iCell // ncols
    return Lx0, Ly0, iCell, w0, w1, w2, w3


@njit(cache=True)
def getScalar(Lx0, Ly0, w0, w1, w2, w3, V):
    """Interpolate a 2-D scalar field *V* to a point given cell coords and
    bilinear weights."""
    return (V[Ly0, Lx0] * w0 +
            V[Ly0, Lx0 + 1] * w1 +
            V[Ly0 + 1, Lx0] * w2 +
            V[Ly0 + 1, Lx0 + 1] * w3)


@njit(cache=True)
def getVector(Lx0, Ly0, w0, w1, w2, w3, Nx, Ny, Nz):
    """Interpolate a 2-D vector field (Nx, Ny, Nz) to a point."""
    nx = (Nx[Ly0, Lx0] * w0 + Nx[Ly0, Lx0 + 1] * w1 +
          Nx[Ly0 + 1, Lx0] * w2 + Nx[Ly0 + 1, Lx0 + 1] * w3)
    ny = (Ny[Ly0, Lx0] * w0 + Ny[Ly0, Lx0 + 1] * w1 +
          Ny[Ly0 + 1, Lx0] * w2 + Ny[Ly0 + 1, Lx0 + 1] * w3)
    nz = (Nz[Ly0, Lx0] * w0 + Nz[Ly0, Lx0 + 1] * w1 +
          Nz[Ly0 + 1, Lx0] * w2 + Nz[Ly0 + 1, Lx0 + 1] * w3)
    return nx, ny, nz


@njit(cache=True)
def SamosATfric(rho, tau0, Rs0, mu, kappa, B, R, v, p, h):
    """SamosAT basal shear stress."""
    Rs = rho * v * v / (p + 0.001)
    div = h / R
    if div < 1.0:
        div = 1.0
    div = math.log(div) / kappa + B
    tau = tau0 + p * mu * (1.0 + Rs0 / (Rs0 + Rs)) + rho * v * v / (div * div)
    return tau


@njit(cache=True)
def samosProjectionIteratrive(xOld, yOld, zOld, ZDEM, nxArray, nyArray,
                               nzArray, csz, ncols, nrows, interpOption,
                               reprojectionIterations):
    """Samos-style iterative projection of a point onto the DEM surface.
    Returns (xNew, yNew, iCell, Lx0, Ly0, w0, w1, w2, w3)."""
    xNew = xOld
    yNew = yOld
    zNew = zOld
    iCell = getCells(xNew, yNew, ncols, nrows, csz)
    if iCell < 0:
        return xNew, yNew, iCell, -1, -1, 0.0, 0.0, 0.0, 0.0
    w0, w1, w2, w3 = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
    Lx0 = iCell % ncols
    Ly0 = iCell // ncols
    Lxi = Lx0
    Lyi = Ly0

    while reprojectionIterations > 0:
        reprojectionIterations -= 1
        # vertical projection
        zTemp = getScalar(Lx0, Ly0, w0, w1, w2, w3, ZDEM)
        # normal at cell centre (equal weights 0.25)
        nx, ny, nz = getVector(Lx0, Ly0, 0.25, 0.25, 0.25, 0.25,
                               nxArray, nyArray, nzArray)
        nx, ny, nz = normalize(nx, ny, nz)
        # zn: (xNew-xNew)*nx is 0, (yNew-yNew)*ny is 0
        zn = (zTemp - zNew) * nz
        xNew = xNew + zn * nx
        yNew = yNew + zn * ny
        zNew = zNew + zn * nz

        iCell = getCells(xNew, yNew, ncols, nrows, csz)
        if iCell < 0:
            return xNew, yNew, iCell, -1, -1, 0.0, 0.0, 0.0, 0.0
        w0, w1, w2, w3 = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
        Lx0 = iCell % ncols
        Ly0 = iCell // ncols
        Lxj = Lx0
        Lyj = Ly0

        if Lxi == Lxj and Lyi == Lyj:
            return xNew, yNew, iCell, Lx0, Ly0, w0, w1, w2, w3
        # update "previous" cell indices for next iteration
        Lxi = Lxj
        Lyi = Lyj

    return xNew, yNew, iCell, Lx0, Ly0, w0, w1, w2, w3


@njit(cache=True)
def reprojectVelocity(uxNew, uyNew, uzNew, nxNew, nyNew, nzNew,
                       velMagMin, keep):
    """Re-project velocity onto the tangent plane of the topography.
    Returns (ux, uy, uz, uMag)."""
    uMag = norm(uxNew, uyNew, uzNew)
    uN = scalProd(uxNew, uyNew, uzNew, nxNew, nyNew, nzNew)
    uxNew = uxNew - uN * nxNew
    uyNew = uyNew - uN * nyNew
    uzNew = uzNew - uN * nzNew
    uMagNew = norm(uxNew, uyNew, uzNew)
    if uMag > 0.0 and keep == 1:
        fac = uMag / (uMagNew + velMagMin)
        uxNew = uxNew * fac
        uyNew = uyNew * fac
        uzNew = uzNew * fac
    return uxNew, uyNew, uzNew, uMag


# ---------------------------------------------------------------------------
# 2.  _compute_force_sph_kernel  (port of computeGradC, SPH options 1 & 2)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _compute_force_sph_kernel(
    # particle arrays
    xArray, yArray, zArray,
    uxArray, uyArray, uzArray,
    hArray, mass, gEff,
    # neighbour search
    indPartInCell, partInCell,
    nColsNeighbourGrid, nRowsNeighbourGrid, cszNeighbourGrid,
    # DEM normal grid
    nxArray, nyArray, nzArray,
    nColsNormal, nRowsNormal, cszNormal,
    # physics
    rho, minRKern, velMagMin, gravAcc, interpOption,
    viscOption, SPHoption, gradient,
):
    """Compute lateral SPH forces for all particles (SPH options 1 & 2).

    Returns (GHX, GHY, GHZ) arrays of shape (N,).
    """
    N = xArray.shape[0]
    rKernel = cszNeighbourGrid
    rK5 = rKernel * rKernel * rKernel * rKernel * rKernel
    facKernel = 10.0 / (math.pi * rK5)
    dfacKernel = -3.0 * facKernel

    GHX = np.zeros(N, dtype=np.float64)
    GHY = np.zeros(N, dtype=np.float64)
    GHZ = np.zeros(N, dtype=np.float64)

    for k in prange(N):
        gradhX = 0.0
        gradhY = 0.0
        gradhZ = 0.0

        x = xArray[k]
        y = yArray[k]
        z = zArray[k]
        ux = uxArray[k]
        uy = uyArray[k]
        uz = uzArray[k]
        hk = hArray[k]
        mk = mass[k]

        # locate particle in SPH neighbour grid
        indx = int(round(x / cszNeighbourGrid))
        indy = int(round(y / cszNeighbourGrid))

        gravAcc3 = gravAcc  # default for option 1

        if SPHoption >= 2:
            Lx0, Ly0, iCell, w0, w1, w2, w3 = getCellAndWeights(
                x, y, nColsNormal, nRowsNormal, cszNormal, interpOption)
            nx, ny, nz = getVector(Lx0, Ly0, w0, w1, w2, w3,
                                   nxArray, nyArray, nzArray)
            nx, ny, nz = normalize(nx, ny, nz)
            gravAcc3 = gEff[k]

        # row range for neighbour search
        lInd = -1
        rInd = 2
        if indy == 0:
            lInd = 0
        if indy == nRowsNeighbourGrid - 1:
            rInd = 1

        for n in range(lInd, rInd):
            ic = (indx - 1) + nColsNeighbourGrid * (indy + n)
            imax = ic
            row_start = nColsNeighbourGrid * (indy + n)
            row_end = nColsNeighbourGrid * (indy + n + 1)
            if imax < row_start:
                imax = row_start
            imin = ic + 3
            if imin > row_end:
                imin = row_end
            iPstart = indPartInCell[imax]
            iPend = indPartInCell[imin]

            for p in range(iPstart, iPend):
                l = partInCell[p]
                if k == l:
                    continue

                dx = xArray[l] - x
                dy = yArray[l] - y
                dz = zArray[l] - z

                # ---- SPH OPTION 1 ----
                if SPHoption == 1:
                    dz = 0.0
                    r = norm(dx, dy, dz)
                    if r < minRKern * rKernel:
                        r = minRKern * rKernel
                    if r < rKernel:
                        hr = rKernel - r
                        dwdr = dfacKernel * hr * hr
                        mdwdrr = mass[l] * dwdr / r
                        gradhX += mdwdrr * dx
                        gradhY += mdwdrr * dy
                        gradhZ += mdwdrr * dz
                        gravAcc3 = gravAcc

                # ---- SPH OPTION 2 ----
                elif SPHoption == 2:
                    r = norm(dx, dy, dz)
                    if r < minRKern * rKernel:
                        fac_min = minRKern * rKernel
                        dx = fac_min * dx
                        dy = fac_min * dy
                        dz = fac_min * dz
                        r = fac_min
                    if r < rKernel:
                        hl = hArray[l]
                        hr = rKernel - r
                        dwdrr = dfacKernel * hr * hr / r
                        ml = mass[l]
                        area = ml / (rho * hl)
                        flux = gravAcc3 * hl
                        pikl = 0.0
                        if viscOption == 2:
                            dux = uxArray[l] - ux
                            duy = uyArray[l] - uy
                            duz = uzArray[l] - uz
                            ck = math.sqrt(gravAcc3 * hk)
                            cl = math.sqrt(gravAcc3 * hl)
                            lambdakl = (ck + cl) * 0.5
                            pikl = -lambdakl * scalProd(dux, duy, duz,
                                                        dx, dy, dz) / r
                        val = (flux + pikl) * dwdrr * area
                        gradhX += val * dx
                        gradhY += val * dy
                        gradhZ += val * dz

        # Convert gradient to force (or leave as gradient)
        if gradient == 1:
            if SPHoption == 2:
                if gravAcc3 != 0.0:
                    GHX[k] = gradhX / gravAcc3
                    GHY[k] = gradhY / gravAcc3
                    GHZ[k] = gradhZ / gravAcc3
            else:
                GHX[k] = -gradhX / rho
                GHY[k] = -gradhY / rho
                GHZ[k] = -gradhZ / rho
        else:
            if SPHoption == 2:
                GHX[k] = gradhX * mk
                GHY[k] = gradhY * mk
                GHZ[k] = gradhZ * mk
            else:
                GHX[k] = gradhX * gravAcc3 / rho * mk
                GHY[k] = gradhY * gravAcc3 / rho * mk
                GHZ[k] = gradhZ * gravAcc3 / rho * mk

    return GHX, GHY, GHZ


# ---------------------------------------------------------------------------
# 3.  _compute_force_kernel  (port of computeForceC, samosAT friction)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _compute_force_kernel(
    # particle state (read/write for ux/uy/uz/mass due to entrainment)
    xArray, yArray, zArray,
    uxArray, uyArray, uzArray,
    hArray, mass, indXDEM, indYDEM,
    totalEnthalpyArray,
    # DEM
    ZDEM, nxArray, nyArray, nzArray, outOfDEM,
    ncols, nrows, csz,
    # fields
    VX, VY, VZ,
    entrMassRaster, entrEnthRaster,
    detRaster, cResRaster,
    # config scalars
    rho, rhoEnt, gravAcc, depMin, velMagMin, dt,
    interpOption, explicitFriction, reprojMethod, reprojectionIterations,
    thresholdProjection, curvAccInFriction, curvAccInTangent, curvAccInGradient,
    viscOption, subgridMixingFactor, resistanceType,
    # samosAT friction parameters
    frictType,
    tau0SamosAt, Rs0SamosAt, muSamosAt, kappaSamosAt, BSamosAt, RSamosAt,
    # entrainment
    entEroEnergy, entShearResistance, entDefResistance,
    # voellmy / coulomb (unused when frictType==1 but passed for generality)
    muVoellmy, xsiVoellmy,
    muCoulomb,
):
    """Compute gravity + friction forces on each particle.

    This is the inner loop of ``computeForceC``.  Only SamosAT friction
    (frictType == 1), Coulomb (2) and Voellmy (3) are ported; other types
    default to tau = 0.

    Returns
    -------
    forceX, forceY, forceZ : float64[N]
    forceFrict : float64[N]
    gEff : float64[N]
    dM : float64[N]  (entrained mass per particle)
    """
    nPart = xArray.shape[0]

    forceX = np.zeros(nPart, dtype=np.float64)
    forceY = np.zeros(nPart, dtype=np.float64)
    forceZ = np.zeros(nPart, dtype=np.float64)
    forceFrict = np.zeros(nPart, dtype=np.float64)
    gEffOut = np.zeros(nPart, dtype=np.float64)
    dM = np.zeros(nPart, dtype=np.float64)

    for k in prange(nPart):
        m = mass[k]
        x = xArray[k]
        y = yArray[k]
        z = zArray[k]
        h = hArray[k]
        if h < depMin:
            h = depMin
        ux = uxArray[k]
        uy = uyArray[k]
        uz = uzArray[k]
        indCellX = indXDEM[k]
        indCellY = indYDEM[k]

        areaPart = m / (h * rho)

        # get cell and weights
        Lx0, Ly0, iCell, w0, w1, w2, w3 = getCellAndWeights(
            x, y, ncols, nrows, csz, interpOption)

        # normal at particle location
        nx, ny, nz = getVector(Lx0, Ly0, w0, w1, w2, w3,
                               nxArray, nyArray, nzArray)
        nx, ny, nz = normalize(nx, ny, nz)

        # estimated end position for curvature
        xEnd = x + dt * ux
        yEnd = y + dt * uy
        zEnd = z + dt * uz

        # default: use start-point normal for end as well
        LxEnd0 = Lx0
        LyEnd0 = Ly0
        wEnd0 = w0
        wEnd1 = w1
        wEnd2 = w2
        wEnd3 = w3

        if reprojMethod == 0:
            iCellEnd = getCells(xEnd, yEnd, ncols, nrows, csz)
            if iCellEnd >= 0 and outOfDEM[iCellEnd] == 0:
                LxEnd0_t, LyEnd0_t, iCellEnd_t, we0, we1, we2, we3 = \
                    getCellAndWeights(xEnd, yEnd, ncols, nrows, csz, interpOption)
                if iCellEnd_t >= 0:
                    LxEnd0 = LxEnd0_t
                    LyEnd0 = LyEnd0_t
                    wEnd0 = we0
                    wEnd1 = we1
                    wEnd2 = we2
                    wEnd3 = we3
        elif reprojMethod == 2:
            xEndP, yEndP, iCellEnd, LxE, LyE, we0, we1, we2, we3 = \
                samosProjectionIteratrive(
                    xEnd, yEnd, zEnd, ZDEM, nxArray, nyArray, nzArray,
                    csz, ncols, nrows, interpOption, reprojectionIterations)
            if iCellEnd >= 0 and outOfDEM[iCellEnd] == 0:
                LxEnd0 = LxE
                LyEnd0 = LyE
                wEnd0 = we0
                wEnd1 = we1
                wEnd2 = we2
                wEnd3 = we3

        # normal at estimated end
        nxEnd, nyEnd, nzEnd = getVector(LxEnd0, LyEnd0,
                                        wEnd0, wEnd1, wEnd2, wEnd3,
                                        nxArray, nyArray, nzArray)
        nxEnd, nyEnd, nzEnd = normalize(nxEnd, nyEnd, nzEnd)

        # averaged normal
        nxAvg, nyAvg, nzAvg = normalize(nx + nxEnd, ny + nyEnd, nz + nzEnd)

        # curvature acceleration
        accNormCurv = (ux * (nxEnd - nx) + uy * (nyEnd - ny) +
                       uz * (nzEnd - nz)) / dt

        # gravity normal component
        gravAccNorm = -gravAcc * nzAvg
        effAccNorm = gravAccNorm + curvAccInFriction * accNormCurv

        # effective gravity for pressure gradient
        if curvAccInGradient == 1:
            if effAccNorm <= 0.0:
                gEffOut[k] = -effAccNorm
            else:
                gEffOut[k] = 0.0
        else:
            gEffOut[k] = -gravAccNorm

        # tangential gravity force
        gravAccTangX = -(gravAccNorm + curvAccInTangent * accNormCurv) * nxAvg
        gravAccTangY = -(gravAccNorm + curvAccInTangent * accNormCurv) * nyAvg
        gravAccTangZ = (-gravAcc -
                        (gravAccNorm + curvAccInTangent * accNormCurv) * nzAvg)
        forceX[k] = gravAccTangX * m
        forceY[k] = gravAccTangY * m
        forceZ[k] = gravAccTangZ * m

        # velocity magnitude
        uMag = norm(ux, uy, uz)

        # friction
        tau = 0.0
        if -effAccNorm >= 0.0:
            sigmaB = -effAccNorm * rho * h
            if frictType == 1:
                tau = SamosATfric(rho, tau0SamosAt, Rs0SamosAt, muSamosAt,
                                  kappaSamosAt, BSamosAt, RSamosAt,
                                  uMag, sigmaB, h)
            elif frictType == 2:
                tau = muCoulomb * sigmaB
            elif frictType == 3:
                tau = muVoellmy * sigmaB + rho * uMag * uMag * gravAcc / xsiVoellmy

        forceBotTang = -areaPart * tau
        if explicitFriction == 1:
            forceFrict[k] = -forceBotTang
        else:
            uMagRes = uMag if uMag >= velMagMin else velMagMin
            forceFrict[k] = -forceBotTang / uMagRes

        # entrainment (simplified -- ploughing & erosion)
        entrMassCell = entrMassRaster[indCellY, indCellX]
        dm = 0.0
        if entrMassCell > 0.0 and uMag > 0.0:
            if entEroEnergy > 0.0:
                dm = areaPart * tau * uMag * dt / entEroEnergy
            else:
                width = math.sqrt(areaPart)
                ABotSwiped = width * uMag * dt
                dm = entrMassCell * ABotSwiped

            # energy loss
            areaEntrPart = areaPart
            dEnergyEntr = areaEntrPart * entShearResistance + dm * entDefResistance

            # momentum conservation on entrainment
            frac = m / (m + dm)
            ux = ux * frac
            uy = uy * frac
            uz = uz * frac
            m = m + dm

            if dEnergyEntr > 0.0:
                dis = 1.0 - dEnergyEntr / (0.5 * m * (uMag * uMag + velMagMin))
                if dis < 0.0:
                    dis = 0.0
                ux = ux * dis
                uy = uy * dis
                uz = uz * dis

        mass[k] = m
        dM[k] = dm
        uxArray[k] = ux
        uyArray[k] = uy
        uzArray[k] = uz

    return forceX, forceY, forceZ, forceFrict, gEffOut, dM


# ---------------------------------------------------------------------------
# 4.  _update_fields_kernel  (port of updateFieldsC)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _update_fields_kernel(
    # particle arrays
    xArray, yArray,
    uxArray, uyArray, uzArray,
    mass, massDet, massEnt,
    trajectoryAngleArray,
    # stopped particles
    xStoppedArray, yStoppedArray, mStoppedArray,
    # DEM info
    ncols, nrows, csz, interpOption,
    areaRaster,
    # config
    rho, rhoEnt,
    computeTA, computeKE, computeP,
    # peak fields (updated in-place)
    PFV, PP, PFT, PTA, PKE,
    DMDet,
):
    """Scatter particle data to Euler grid fields.

    This kernel is serial because multiple particles write to the same
    grid cells (write conflicts).

    Returns
    -------
    FTBilinear, VBilinear, VXBilinear, VYBilinear, VZBilinear : 2D arrays
    PBilinear, kineticEnergy, travelAngleField : 2D arrays
    FTDetBilinear, FTStopBilinear, FTEntBilinear : 2D arrays
    MassBilinear : 2D array
    hBB : 1D array  (flow thickness per particle)
    """
    nPart = xArray.shape[0]

    MassBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    MassDetBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    MassStopBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    MassEntBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    MomBilinearX = np.zeros((nrows, ncols), dtype=np.float64)
    MomBilinearY = np.zeros((nrows, ncols), dtype=np.float64)
    MomBilinearZ = np.zeros((nrows, ncols), dtype=np.float64)
    VBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    VXBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    VYBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    VZBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    FTBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    PBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    kineticEnergy = np.zeros((nrows, ncols), dtype=np.float64)
    travelAngleField = np.zeros((nrows, ncols), dtype=np.float64)
    FTDetBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    FTStopBilinear = np.zeros((nrows, ncols), dtype=np.float64)
    FTEntBilinear = np.zeros((nrows, ncols), dtype=np.float64)

    # index offsets for the 4 bilinear neighbours
    ind1_arr = np.array([0, 1, 0, 1], dtype=np.int32)
    ind2_arr = np.array([0, 0, 1, 1], dtype=np.int32)

    # ---- scatter particles to grid ----
    for k in range(nPart):
        x = xArray[k]
        y = yArray[k]
        ux_k = uxArray[k]
        uy_k = uyArray[k]
        uz_k = uzArray[k]
        m = mass[k]
        dmDet = massDet[k]
        dmEnt = massEnt[k]

        Lx0, Ly0, iCell, w0, w1, w2, w3 = getCellAndWeights(
            x, y, ncols, nrows, csz, interpOption)

        w_arr = np.empty(4, dtype=np.float64)
        w_arr[0] = w0
        w_arr[1] = w1
        w_arr[2] = w2
        w_arr[3] = w3

        # travel angle (nearest neighbour)
        if computeTA:
            indx_ta = int(round(x / csz))
            indy_ta = int(round(y / csz))
            if (0 <= indx_ta < ncols and 0 <= indy_ta < nrows):
                ta = trajectoryAngleArray[k]
                if ta > travelAngleField[indy_ta, indx_ta]:
                    travelAngleField[indy_ta, indx_ta] = ta

        for i in range(4):
            indx = Lx0 + ind1_arr[i]
            indy = Ly0 + ind2_arr[i]
            mwi = m * w_arr[i]
            dmDetWi = dmDet * w_arr[i]
            dmEntWi = dmEnt * w_arr[i]
            MassBilinear[indy, indx] += mwi
            MassDetBilinear[indy, indx] += dmDetWi
            MassEntBilinear[indy, indx] += dmEntWi
            MomBilinearX[indy, indx] += mwi * ux_k
            MomBilinearY[indy, indx] += mwi * uy_k
            MomBilinearZ[indy, indx] += mwi * uz_k

    # ---- scatter stopped particles ----
    nStopped = xStoppedArray.shape[0]
    for l_s in range(nStopped):
        xStop = xStoppedArray[l_s]
        yStop = yStoppedArray[l_s]
        mStop = -mStoppedArray[l_s]  # negative for the flow

        Lx0, Ly0, iCell, w0, w1, w2, w3 = getCellAndWeights(
            xStop, yStop, ncols, nrows, csz, interpOption)

        w_arr2 = np.empty(4, dtype=np.float64)
        w_arr2[0] = w0
        w_arr2[1] = w1
        w_arr2[2] = w2
        w_arr2[3] = w3

        for i in range(4):
            indx = Lx0 + ind1_arr[i]
            indy = Ly0 + ind2_arr[i]
            mwi = mStop * w_arr2[i]
            MassStopBilinear[indy, indx] += mwi

    # ---- compute derived fields on the grid ----
    for j in range(nrows):
        for i in range(ncols):
            m = MassBilinear[j, i]
            DMDet[j, i] = DMDet[j, i] + MassDetBilinear[j, i]

            if m > 0.0:
                FTBilinear[j, i] = m / (areaRaster[j, i] * rho)
                VXBilinear[j, i] = MomBilinearX[j, i] / m
                VYBilinear[j, i] = MomBilinearY[j, i] / m
                VZBilinear[j, i] = MomBilinearZ[j, i] / m
                vx = VXBilinear[j, i]
                vy = VYBilinear[j, i]
                vz = VZBilinear[j, i]
                vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
                VBilinear[j, i] = vmag
                if vmag > PFV[j, i]:
                    PFV[j, i] = vmag
                if FTBilinear[j, i] > PFT[j, i]:
                    PFT[j, i] = FTBilinear[j, i]
                if computeP:
                    p = rho * vmag * vmag
                    PBilinear[j, i] = p
                    if p > PP[j, i]:
                        PP[j, i] = p
                if computeTA:
                    if travelAngleField[j, i] > PTA[j, i]:
                        PTA[j, i] = travelAngleField[j, i]
                if computeKE:
                    ke = 0.5 * m * vmag * vmag
                    kineticEnergy[j, i] = ke
                    if ke > PKE[j, i]:
                        PKE[j, i] = ke

            FTDetBilinear[j, i] = -MassDetBilinear[j, i] / (areaRaster[j, i] * rho)
            FTStopBilinear[j, i] = -MassStopBilinear[j, i] / (areaRaster[j, i] * rho)
            FTEntBilinear[j, i] = -MassEntBilinear[j, i] / (areaRaster[j, i] * rhoEnt)

    # ---- interpolate flow thickness back to particles ----
    hBB = np.zeros(nPart, dtype=np.float64)
    for k in range(nPart):
        x = xArray[k]
        y = yArray[k]
        Lx0, Ly0, iCell, w0, w1, w2, w3 = getCellAndWeights(
            x, y, ncols, nrows, csz, interpOption)
        hBB[k] = getScalar(Lx0, Ly0, w0, w1, w2, w3, FTBilinear)

    return (FTBilinear, VBilinear, VXBilinear, VYBilinear, VZBilinear,
            PBilinear, kineticEnergy, travelAngleField,
            FTDetBilinear, FTStopBilinear, FTEntBilinear,
            MassBilinear,
            hBB)


# ---------------------------------------------------------------------------
# 5.  patch_avaframe()  --  monkey-patch wrapper
# ---------------------------------------------------------------------------

def patch_avaframe():
    """Monkey-patch avaframe's Cython modules so that the hot inner loops
    use the Numba kernels above instead.

    Call this once before running a com1DFA simulation.
    """
    import avaframe.com1DFA.DFAfunctionsCython as DFAfunC

    # -- keep references to originals so we can delegate for unsupported paths
    _orig_computeForceC = DFAfunC.computeForceC
    _orig_computeForceSPHC = DFAfunC.computeForceSPHC
    _orig_updateFieldsC = DFAfunC.updateFieldsC

    # ----------------------------------------------------------------
    # Patched computeForceSPHC
    # ----------------------------------------------------------------
    def patched_computeForceSPHC(cfg, particles, force, dem, sphOption, gradient=0):
        if sphOption > 2:
            # option 3 not ported -- fall back
            return _orig_computeForceSPHC(cfg, particles, force, dem, sphOption, gradient)

        headerNeighbourGrid = dem['headerNeighbourGrid']
        headerNormalGrid = dem['header']

        GHX, GHY, GHZ = _compute_force_sph_kernel(
            np.ascontiguousarray(particles['x']),
            np.ascontiguousarray(particles['y']),
            np.ascontiguousarray(particles['z']),
            np.ascontiguousarray(particles['ux']),
            np.ascontiguousarray(particles['uy']),
            np.ascontiguousarray(particles['uz']),
            np.ascontiguousarray(particles['h']),
            np.ascontiguousarray(particles['m']),
            np.ascontiguousarray(particles['gEff']),
            np.ascontiguousarray(particles['indPartInCell'], dtype=np.int32),
            np.ascontiguousarray(particles['partInCell'], dtype=np.int32),
            int(headerNeighbourGrid['ncols']),
            int(headerNeighbourGrid['nrows']),
            float(headerNeighbourGrid['cellsize']),
            np.ascontiguousarray(dem['Nx']),
            np.ascontiguousarray(dem['Ny']),
            np.ascontiguousarray(dem['Nz']),
            int(headerNormalGrid['ncols']),
            int(headerNormalGrid['nrows']),
            float(headerNormalGrid['cellsize']),
            cfg.getfloat('rho'),
            cfg.getfloat('minRKern'),
            cfg.getfloat('velMagMin'),
            cfg.getfloat('gravAcc'),
            cfg.getint('interpOption'),
            cfg.getint('viscOption'),
            int(sphOption),
            int(gradient),
        )

        force['forceSPHX'] = np.asarray(GHX)
        force['forceSPHY'] = np.asarray(GHY)
        force['forceSPHZ'] = np.asarray(GHZ)
        return particles, force

    # ----------------------------------------------------------------
    # Patched computeForceC
    # ----------------------------------------------------------------
    def patched_computeForceC(cfg, particles, fields, dem, frictType, resistanceType):
        # For friction types we haven't ported, fall back
        if frictType not in (1, 2, 3):
            return _orig_computeForceC(cfg, particles, fields, dem, frictType, resistanceType)

        nPart = particles['nPart']
        csz = dem['header']['cellsize']
        nrows = dem['header']['nrows']
        ncols = dem['header']['ncols']

        outOfDEM_flat = np.array(dem['outOfDEM'], dtype=np.uint8).ravel()

        forceX, forceY, forceZ, forceFrict, gEffOut, dM = \
            _compute_force_kernel(
                np.ascontiguousarray(particles['x']),
                np.ascontiguousarray(particles['y']),
                np.ascontiguousarray(particles['z']),
                np.ascontiguousarray(particles['ux']),
                np.ascontiguousarray(particles['uy']),
                np.ascontiguousarray(particles['uz']),
                np.ascontiguousarray(particles['h']),
                np.ascontiguousarray(particles['m']),
                np.ascontiguousarray(particles['indXDEM'], dtype=np.int32),
                np.ascontiguousarray(particles['indYDEM'], dtype=np.int32),
                np.ascontiguousarray(particles['totalEnthalpy']),
                np.ascontiguousarray(dem['rasterData']),
                np.ascontiguousarray(dem['Nx']),
                np.ascontiguousarray(dem['Ny']),
                np.ascontiguousarray(dem['Nz']),
                outOfDEM_flat,
                ncols, nrows, csz,
                np.ascontiguousarray(fields['Vx']),
                np.ascontiguousarray(fields['Vy']),
                np.ascontiguousarray(fields['Vz']),
                np.ascontiguousarray(fields['entrMassRaster']),
                np.ascontiguousarray(fields['entrEnthRaster']),
                np.ascontiguousarray(fields['detRaster']),
                np.ascontiguousarray(fields['cResRaster']),
                cfg.getfloat('rho'),
                cfg.getfloat('rhoEnt'),
                cfg.getfloat('gravAcc'),
                cfg.getfloat('depMin'),
                cfg.getfloat('velMagMin'),
                float(particles['dt']),
                cfg.getint('interpOption'),
                cfg.getint('explicitFriction'),
                cfg.getint('reprojMethodForce'),
                cfg.getint('reprojectionIterations'),
                cfg.getfloat('thresholdProjection'),
                cfg.getfloat('curvAccInFriction'),
                cfg.getfloat('curvAccInTangent'),
                cfg.getint('curvAccInGradient'),
                cfg.getint('viscOption'),
                cfg.getfloat('subgridMixingFactor'),
                int(resistanceType),
                int(frictType),
                cfg.getfloat('tau0samosat'),
                cfg.getfloat('Rs0samosat'),
                cfg.getfloat('musamosat'),
                cfg.getfloat('kappasamosat'),
                cfg.getfloat('Bsamosat'),
                cfg.getfloat('Rsamosat'),
                cfg.getfloat('entEroEnergy'),
                cfg.getfloat('entShearResistance'),
                cfg.getfloat('entDefResistance'),
                cfg.getfloat('muvoellmy'),
                cfg.getfloat('xsivoellmy'),
                cfg.getfloat('mucoulomb'),
            )

        # pack results
        force = {}
        force['dM'] = np.asarray(dM)
        force['dMDet'] = np.zeros(nPart, dtype=np.float64)
        force['forceX'] = np.asarray(forceX)
        force['forceY'] = np.asarray(forceY)
        force['forceZ'] = np.asarray(forceZ)
        force['forceFrict'] = np.asarray(forceFrict)
        particles['gEff'] = np.asarray(gEffOut)
        particles['curvAcc'] = np.zeros(nPart, dtype=np.float64)
        # ux/uy/uz/m are modified in-place by the kernel
        particles['dmDet'] = np.zeros(nPart, dtype=np.float64)
        particles['dmEnt'] = np.asarray(dM)

        # update entrainment raster (serial -- same cell can be hit multiple
        # times)
        areaRaster = dem['areaRaster']
        entrMassRaster = fields['entrMassRaster']
        indXDEM = particles['indXDEM']
        indYDEM = particles['indYDEM']
        for k in range(nPart):
            ix = indXDEM[k]
            iy = indYDEM[k]
            emc = entrMassRaster[iy, ix]
            areaCell = areaRaster[iy, ix]
            emc = emc - dM[k] / areaCell
            if emc < 0.0:
                emc = 0.0
            entrMassRaster[iy, ix] = emc
        fields['entrMassRaster'] = entrMassRaster

        return particles, force, fields

    # ----------------------------------------------------------------
    # Patched updateFieldsC
    # ----------------------------------------------------------------
    def patched_updateFieldsC(cfg, particles, dem, fields):
        header = dem['header']
        nrows = header['nrows']
        ncols = header['ncols']
        csz = float(header['cellsize'])
        interpOption = cfg.getint('interpOption')
        rho = cfg.getfloat('rho')
        rhoEnt = cfg.getfloat('rhoEnt')

        computeTA = bool(fields['computeTA'])
        computeKE = bool(fields['computeKE'])
        computeP = bool(fields['computeP'])

        # ensure contiguous arrays
        xArr = np.ascontiguousarray(particles['x'])
        yArr = np.ascontiguousarray(particles['y'])
        uxArr = np.ascontiguousarray(particles['ux'])
        uyArr = np.ascontiguousarray(particles['uy'])
        uzArr = np.ascontiguousarray(particles['uz'])
        mArr = np.ascontiguousarray(particles['m'])
        mDetArr = np.ascontiguousarray(particles['dmDet'])
        mEntArr = np.ascontiguousarray(particles['dmEnt'])
        taArr = np.ascontiguousarray(particles['trajectoryAngle'])
        areaRaster = np.ascontiguousarray(dem['areaRaster'])

        xStop = np.ascontiguousarray(particles['stoppedParticles']['x'])
        yStop = np.ascontiguousarray(particles['stoppedParticles']['y'])
        mStop = np.ascontiguousarray(particles['stoppedParticles']['m'])

        PFV = np.ascontiguousarray(fields['pfv'])
        PP = np.ascontiguousarray(fields['ppr'])
        PFT = np.ascontiguousarray(fields['pft'])
        PTA = np.ascontiguousarray(fields['pta'])
        PKE = np.ascontiguousarray(fields['pke'])
        DMDet = np.ascontiguousarray(fields['dmDet'])

        (FTBilinear, VBilinear, VXBilinear, VYBilinear, VZBilinear,
         PBilinear, kineticEnergy, travelAngleField,
         FTDetBilinear, FTStopBilinear, FTEntBilinear,
         MassBilinear,
         hBB) = _update_fields_kernel(
            xArr, yArr, uxArr, uyArr, uzArr,
            mArr, mDetArr, mEntArr, taArr,
            xStop, yStop, mStop,
            ncols, nrows, csz, interpOption,
            areaRaster,
            rho, rhoEnt,
            computeTA, computeKE, computeP,
            PFV, PP, PFT, PTA, PKE,
            DMDet,
        )

        fields['FM'] = np.asarray(MassBilinear)
        fields['FV'] = np.asarray(VBilinear)
        fields['Vx'] = np.asarray(VXBilinear)
        fields['Vy'] = np.asarray(VYBilinear)
        fields['Vz'] = np.asarray(VZBilinear)
        fields['FT'] = np.asarray(FTBilinear)
        fields['pfv'] = np.asarray(PFV)
        fields['pft'] = np.asarray(PFT)
        fields['dmDet'] = np.asarray(DMDet)
        fields['FTStop'] = np.asarray(FTStopBilinear)
        fields['FTDet'] = np.asarray(FTDetBilinear)
        fields['FTEnt'] = np.asarray(FTEntBilinear)

        if computeP:
            fields['ppr'] = np.asarray(PP)
            fields['P'] = np.asarray(PBilinear)
        if computeTA:
            fields['TA'] = np.asarray(travelAngleField)
            fields['pta'] = np.asarray(PTA)
        if computeKE:
            fields['pke'] = np.asarray(PKE)

        particles['h'] = np.asarray(hBB)

        return particles, fields

    # ---- Apply patches ----
    DFAfunC.computeForceC = patched_computeForceC
    DFAfunC.computeForceSPHC = patched_computeForceSPHC
    DFAfunC.updateFieldsC = patched_updateFieldsC

    log.info("com1dfa_fast: monkey-patched DFAfunctionsCython with Numba kernels")
