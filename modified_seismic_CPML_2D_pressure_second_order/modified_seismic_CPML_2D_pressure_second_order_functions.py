import numpy as np
from numba import njit


@njit
def apply_pml_x(NX, DELTAX, USE_PML_XMIN, USE_PML_XMAX, xoriginleft, xoriginright, thickness_PML_x, d0_x, K_MAX_PML, ALPHA_MAX_PML, ZERO, NPOWER, DELTAT, d_x, K_x, alpha_x, d_x_half, K_x_half, alpha_x_half, b_x, b_x_half, a_x, a_x_half):
    for i in range(0, NX):

        # Abscisa del punto de la malla actual a lo largo del perfil de amortiguamiento
        xval = DELTAX * np.float64(i - 1)

        # ---------- Borde izquierdo
        if USE_PML_XMIN:

            # Definir perfil de amortiguamiento en los puntos de la malla
            abscissa_in_PML = xoriginleft - xval
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized**NPOWER
                K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

            # Definir perfil de amortiguamiento en la mitad de los puntos de la malla
            abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0)
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized**NPOWER
                K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # ---------- Borde derecho
        if USE_PML_XMAX:

            # Definir perfil de amortiguamiento en los puntos de la malla
            abscissa_in_PML = xval - xoriginright
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized**NPOWER
                K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

            # Definir perfil de amortiguamiento en la mitad de los puntos de la malla
            abscissa_in_PML = xval + DELTAX / 2.0 - xoriginright
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized**NPOWER
                K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        alpha_x[i] = max(alpha_x[i], ZERO)
        alpha_x_half[i] = max(alpha_x_half[i], ZERO)

        # Calcular coeficientes de amortiguamiento
        b_x[i] = np.exp(- (d_x[i] / K_x[i] + alpha_x[i]) * DELTAT)
        b_x_half[i] = np.exp(- (d_x_half[i] / K_x_half[i] + alpha_x_half[i]) * DELTAT)

        a_x[i] = d_x[i] * (b_x[i] - 1.0) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i])) if abs(d_x[i]) > 1.0e-6 else 0.0
        a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i])) if abs(d_x_half[i]) > 1.0e-6 else 0.0


@njit
def apply_pml_y(NY, DELTAY, USE_PML_YMIN, USE_PML_YMAX, yoriginbottom, yorigintop, ZERO, thickness_PML_y, d0_y, NPOWER, K_MAX_PML, ALPHA_MAX_PML, DELTAT, 
              d_y, K_y, alpha_y, d_y_half, K_y_half, alpha_y_half, b_y, b_y_half, a_y, a_y_half):
    for j in range(NY):
        # Abscisa del punto de la malla actual a lo largo del perfil de amortiguamiento
        yval = DELTAY * (j - 1)

        # ---------- Borde inferior
        if USE_PML_YMIN:
            # Definir perfil de amortiguamiento en los puntos de la malla
            abscissa_in_PML = yoriginbottom - yval
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized**NPOWER
                K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

            # Definir perfil de amortiguamiento en la mitad de los puntos de la malla
            abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0)
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized**NPOWER
                K_y_half[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # ---------- Borde superior
        if USE_PML_YMAX:
            # Definir perfil de amortiguamiento en los puntos de la malla
            abscissa_in_PML = yval - yorigintop
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized**NPOWER
                K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

            # Definir perfil de amortiguamiento en la mitad de los puntos de la malla
            abscissa_in_PML = yval + DELTAY / 2.0 - yorigintop
            if abscissa_in_PML >= ZERO:
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized**NPOWER
                K_y_half[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
                alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        b_y[j] = np.exp(- (d_y[j] / K_y[j] + alpha_y[j]) * DELTAT)
        b_y_half[j] = np.exp(- (d_y_half[j] / K_y_half[j] + alpha_y_half[j]) * DELTAT)

        # Evitar división por cero fuera de la PML
        if abs(d_y[j]) > 1.0e-6:
            a_y[j] = d_y[j] * (b_y[j] - 1.0) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
        if abs(d_y_half[j]) > 1.0e-6:
            a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))

@njit 
def find_nearest_grid_points(NREC, NX, NY, DELTAX, DELTAY, xrec, yrec, ix_rec, iy_rec, HUGEVAL):
    for irec in range(NREC):
        dist = HUGEVAL
        for j in range(1, NY + 1):
            for i in range(1, NX + 1):
                distval = np.sqrt((DELTAX * (i - 1) - xrec[irec])**2 + (DELTAY * (j - 1) - yrec[irec])**2)
                if distval < dist:
                    dist = distval
                    ix_rec[irec] = i
                    iy_rec[irec] = j

@njit
def time_loop(NSTEP, NX, NY, DELTAX, DELTAY, DELTAT, f0, t0, factor, cp_unrelaxed, kappa_unrelaxed, Kronecker_source, 
              b_x_half, a_x_half, K_x_half, b_y_half, a_y_half, K_y_half, b_x, a_x, K_x, b_y, a_y, K_y, 
              pressure_present, pressure_past, pressure_future, memory_dpressure_dx, memory_dpressure_dy, 
              memory_dpressurexx_dx, memory_dpressureyy_dy, dpressurexx_dx, dpressureyy_dy, rho, pressure_xx, pressure_yy):
    
    # Precalcular constantes fuera del bucle
    a = np.pi**2 * f0**2  # factor de la fuente
    stored_pressures = np.zeros((NSTEP, NX, NY))  # Array para almacenar las presiones en el receptor


    for it in range(NSTEP):
        # Calcular las diferencias de presión en x (evitando el bucle)
        value_dpressure_dx = (pressure_present[1:NX, 0:NY-1] - pressure_present[0:NX-1, 0:NY-1]) / DELTAX

        stored_pressures[it] = np.copy(pressure_present)


        # Actualizar memory_dpressure_dx y ajustar value_dpressure_dx
        for i in range(NX-1):
            for j in range(NY-1):
                memory_dpressure_dx[i, j] = b_x_half[i] * memory_dpressure_dx[i, j] + a_x_half[i] * value_dpressure_dx[i, j]
                value_dpressure_dx[i, j] = value_dpressure_dx[i, j] / K_x_half[i] + memory_dpressure_dx[i, j]

        # Calcular rho_half_x y pressure_xx en un solo bucle
        rho_half_x = np.empty((NX-1, NY-1), dtype=np.float64)
        for i in range(NX-1):
            for j in range(NY-1):
                rho_half_x[i, j] = 0.5 * (rho[i+1, j] + rho[i, j])
                pressure_xx[i, j] = value_dpressure_dx[i, j] / rho_half_x[i, j]

        # Calcular value_dpressure_dy, actualizar memory_dpressure_dy, y calcular rho_half_y en un solo bucle
        value_dpressure_dy = (pressure_present[:, 1:NY-1] - pressure_present[:, 0:NY-2]) / DELTAY
        rho_half_y = np.empty((NX, NY-2), dtype=np.float64)
        for i in range(NX):
            for j in range(NY-2):
                memory_dpressure_dy[i, j] = b_y_half[j] * memory_dpressure_dy[i, j] + a_y_half[j] * value_dpressure_dy[i, j]
                value_dpressure_dy[i, j] = value_dpressure_dy[i, j] / K_y_half[j] + memory_dpressure_dy[i, j]
                rho_half_y[i, j] = 0.5 * (rho[i, j+1] + rho[i, j])
                if rho_half_y[i, j] == 0:
                    rho_half_y[i, j] = np.finfo(np.float64).eps
                pressure_yy[i, j] = value_dpressure_dy[i, j] / rho_half_y[i, j]

        # Combinar bucles de memory_dpressurexx_dx, ajustar value_dpressurexx_dx y actualizar dpressurexx_dx
        value_dpressurexx_dx = (pressure_xx[1:NX, 0:NY-1] - pressure_xx[0:NX-1, 0:NY-1]) / DELTAX
        for i in range(1, NX):
            for j in range(NY-1):
                memory_dpressurexx_dx[i, j] = b_x[i] * memory_dpressurexx_dx[i, j] + a_x[i] * value_dpressurexx_dx[i-1, j]
                value_dpressurexx_dx[i-1, j] = value_dpressurexx_dx[i-1, j] / K_x[i] + memory_dpressurexx_dx[i, j]
                dpressurexx_dx[i, j] = value_dpressurexx_dx[i-1, j]

        # Calcular value_dpressureyy_dy, actualizar memory_dpressureyy_dy y actualizar dpressureyy_dy en un solo bucle
        value_dpressureyy_dy = (pressure_yy[:, 1:NY-1] - pressure_yy[:, 0:NY-2]) / DELTAY
        for i in range(NX):
            for j in range(1, NY-1):
                memory_dpressureyy_dy[i, j] = b_y[j] * memory_dpressureyy_dy[i, j] + a_y[j] * value_dpressureyy_dy[i, j-1]
                value_dpressureyy_dy[i, j-1] = value_dpressureyy_dy[i, j-1] / K_y[j] + memory_dpressureyy_dy[i, j]
                dpressureyy_dy[i, j] = value_dpressureyy_dy[i, j-1]

        # add the source (pressure located at a given grid point)
        t = (it - 1) * DELTAT

        # Ricker source time function (second derivative of a Gaussian)
        source_term = factor * (1.0 - 2.0 * a * (t - t0)**2) * np.exp(-a * (t - t0)**2)

        # apply the time evolution scheme
        for i in range(NX):
            for j in range(NY):
                pressure_future[i, j] = -pressure_past[i, j] + 2.0 * pressure_present[i, j] + \
                                        DELTAT * DELTAT * ((dpressurexx_dx[i, j] + dpressureyy_dy[i, j]) * kappa_unrelaxed[i, j] + \
                                        4.0 * np.pi * cp_unrelaxed**2 * source_term * Kronecker_source[i, j])

        # apply Dirichlet conditions at the edges of the domain
        for j in range(NY):
            pressure_future[0, j] = 0.0  # Dirichlet condition for pressure on the left boundary
            pressure_future[NX - 1, j] = 0.0  # Dirichlet condition for pressure on the right boundary
        for i in range(NX):
            pressure_future[i, 0] = 0.0  # Dirichlet condition for pressure on the bottom boundary
            pressure_future[i, NY - 1] = 0.0  # Dirichlet condition for pressure on the top boundary

        #Update pressure_past and pressure_present
        for i in range(NX):
            for j in range(NY):
                pressure_past[i, j] = pressure_present[i, j]
                pressure_present[i, j] = pressure_future[i, j]

        if it % 500 == 0:
            print('Iteración:', it)      
    return stored_pressures                                               