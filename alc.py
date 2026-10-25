import numpy as np

# =========================================================
# LABORATORIO 1 - Numeros de maquina
# =========================================================

def error(x, y):
    #calcula el error de aproximar x usando y
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return abs(x - y)


def error_relativo(x, y):
    #calcula el error relativo de aproximar x usando y
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return error(x, y) / abs(x)


def matricesIguales(A, B):
    #devuelve True si dos matrices son iguales, o False en caso contrario
    atol = 1e-8
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    # Si tienen distinta forma (filas, columnas) no pueden ser iguales
    if np.shape(A) != np.shape(B):
        return False

    filas, columnas = np.shape(A)

    # Comparamos elemento a elemento usando slicing/indexado
    for i in range(filas):
        for j in range(columnas):
            if error(A[i, j], B[i, j]) > atol:
                return False

    return True


# =========================================================
# LABORATORIO 2 - Transformaciones lineales
# =========================================================

def multiplicar(A, B):
    #Funcion auxiliar para multiplicar matrices
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    # Si viene un vector (1D), lo llevamos a formato matriz (2D)
    if np.ndim(A) == 1:
        A = np.reshape(A, (1, np.shape(A)[0]))
    if np.ndim(B) == 1:
        B = np.reshape(B, (np.shape(B)[0], 1))

    m, n = np.shape(A)
    n2, p = np.shape(B)

    C = np.zeros((m, p))  # matriz de salida
    for i in range(m):
        for j in range(p):
            suma = 0.0
            for k in range(n):
                suma = suma + A[i, k] * B[k, j]
            C[i, j] = suma

    return C


def rota(theta):
    # Retorna una matriz que rota un vector en angulo theta
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def escala(s):
    # Da una matriz de nxn, con el tamaño de s
    s = np.array(s, dtype=np.float64)
    n = np.shape(s)[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = s[i]
    return T


def rota_y_escala(theta, s):
    # Retorna una matriz de 2x2 que rota el vector en theta y dsp lo escala en un factor s
    return multiplicar(escala(s), rota(theta))


def afin(theta, s, b):
    # Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    # luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b
    A = rota_y_escala(theta, s)
    b = np.array(b, dtype=np.float64)

    M = np.zeros((3, 3))
    M[0:2, 0:2] = A
    M[0:2, 2] = b
    M[2, 2] = 1

    return M


def trans_afin(v, theta, s, b):
    # Retorna el vector w resultante de aplicar la transformacion afin a v
    v = np.array(v, dtype=np.float64)
    M = afin(theta, s, b)

    v_hom = np.ones(3)
    v_hom[0:2] = v

    res = multiplicar(M, v_hom)
    res = np.reshape(res, (3,))

    return res[0:2]


# =========================================================
# LABORATORIO 3 - Normas y condicion de matrices
# =========================================================

def norma(x, p):
    x = np.array(x, dtype=np.float64)
    n = np.shape(x)[0]

    if p == 'inf':
        return max(abs(x[i]) for i in range(n))

    res = 0.0
    for i in range(n):
        res += abs(x[i]) ** p
    return res ** (1 / p)


def normaliza(X, p):
    resultado = []
    for x in X:
        x = np.array(x, dtype=np.float64)
        resultado.append(x / norma(x, p))
    return resultado


def normaMatMC(A, q, p, Np):
    A = np.array(A, dtype=np.float64)
    filas, columnas = np.shape(A)
    n = columnas
    max_norma = 0.0
    x_max = None
    for _ in range(Np):
        x = np.random.uniform(-1, 1, n)
        x = x / norma(x, p)
        Ax = A @ x
        norma_actual = norma(Ax, q)
        if norma_actual > max_norma:
            max_norma = norma_actual
            x_max = x
    return max_norma, x_max


def normaExactaValor(A, valor):
    n, m = np.shape(A)

    if valor == 1:
        columnas = []
        for j in range(m):
            suma = 0.0
            for i in range(n):
                suma += abs(A[i, j])
            columnas.append(suma)
        return max(columnas)
    elif valor == 'inf':
        filas = []
        for i in range(n):
            suma = 0.0
            for j in range(m):
                suma += abs(A[i, j])
            filas.append(suma)
        return max(filas)
    else:
        return None


def normaExacta(A, p=[1, 'inf']):
    A = np.array(A, dtype=np.float64)

    if isinstance(p, list):
        return [normaExactaValor(A, valor) for valor in p]
    else:
        return normaExactaValor(A, p)


def condMC(A, p, Np=1000):
    A = np.array(A, dtype=np.float64)
    Ainv = inversa(A)
    normaA, _ = normaMatMC(A, p, p, Np)
    normaAinv, _ = normaMatMC(Ainv, p, p, Np)
    return normaA * normaAinv


def condExacta(A, p):
    A = np.array(A, dtype=np.float64)
    Ainv = inversa(A)
    return normaExacta(A, p) * normaExacta(Ainv, p)


# =========================================================
# LABORATORIO 4 - Descomposicion LU
# =========================================================

def calculaLU(A):
    if A is None:
        return None, None, 0

    A = np.array(A, dtype=np.float64)

    if np.ndim(A) != 2:
        return None, None, 0

    n, m = np.shape(A)
    if n != m:
        return None, None, 0

    op = 0
    Atilde = np.copy(A)

    for k in range(n - 1):
        if Atilde[k, k] == 0:
            return None, None, 0
        for i in range(k + 1, n):
            m_ik = Atilde[i, k] / Atilde[k, k]
            op += 1
            Atilde[i, k] = m_ik
            for j in range(k + 1, n):
                Atilde[i, j] = Atilde[i, j] - m_ik * Atilde[k, j]
                op += 2

    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = Atilde[i, j]
            else:
                U[i, j] = Atilde[i, j]

    return L, U, op


def res_tri(L, b, inferior=True):
    L = np.array(L, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = np.shape(L)[0]
    x = np.zeros(n)

    if inferior:
        for i in range(n):
            suma = 0.0
            for j in range(i):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]
    else:
        for i in range(n - 1, -1, -1):
            suma = 0.0
            for j in range(i + 1, n):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]

    return x


def inversa(A):
    # Nota: esta version (basada en calculaLU) reemplaza a la del Labo 3
    A = np.array(A, dtype=np.float64)
    n = np.shape(A)[0]

    L, U, op = calculaLU(A)
    if L is None:
        return None

    for i in range(n):
        if U[i, i] == 0:
            return None  # A es singular, no tiene inversa

    Ainv = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        y = res_tri(L, e_i, inferior=True)
        x = res_tri(U, y, inferior=False)
        Ainv[:, i] = x

    return Ainv


def calculaLDV(A):
    A = np.array(A, dtype=np.float64)
    n = np.shape(A)[0]

    resultado_LU = calculaLU(A)
    if resultado_LU is None:
        return None
    L, U, op = resultado_LU

    resultado_LU2 = calculaLU(U.T)
    if resultado_LU2 is None:
        return None
    L2, U2, op2 = resultado_LU2

    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = U2[i, i]

    V = L2.T

    return L, D, V


def esSDP(A, atol=1e-8):
    A = np.array(A, dtype=np.float64)
    n = np.shape(A)[0]

    for i in range(n):
        for j in range(n):
            if abs(A[i, j] - A[j, i]) > atol:
                return False

    resultado = calculaLDV(A)
    if resultado is None:
        return False
    L, D, V = resultado

    for i in range(n):
        if D[i, i] <= atol:
            return False

    return True


# =========================================================
# LABORATORIO 5 - Descomposicion QR
# =========================================================

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    A = np.array(A, dtype=np.float64)
    n, m = np.shape(A)
    if n != m:
        return None

    op = 0
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    mantener = []  # indices de las columnas que efectivamente se guardan

    for j in range(n):
        qj_tilde = np.copy(A[:, j])

        # proyecto y resto la parte de qj_tilde sobre cada vector ya guardado
        for k in mantener:
            rkj = 0.0
            for i in range(n):
                rkj += Q[i, k] * qj_tilde[i]
                op += 2  # 1 mult + 1 suma
            R[k, j] = rkj
            for i in range(n):
                qj_tilde[i] = qj_tilde[i] - rkj * Q[i, k]
                op += 2  # 1 mult + 1 resta

        norma_qtilde = 0.0
        for i in range(n):
            norma_qtilde += qj_tilde[i] ** 2
            op += 2
        norma_qtilde = norma_qtilde ** 0.5
        op += 1

        if norma_qtilde > tol:
            R[j, j] = norma_qtilde
            for i in range(n):
                Q[i, j] = qj_tilde[i] / norma_qtilde
                op += 1
            mantener.append(j)
        # si no se mantiene, la columna j queda descartada (no se agrega a Q ni a R)

    r = len(mantener)
    Q_hat = np.zeros((n, r))
    R_hat = np.zeros((r, n))
    for pos, j in enumerate(mantener):
        Q_hat[:, pos] = Q[:, j]
        R_hat[pos, :] = R[j, :]

    if retorna_nops:
        return Q_hat, R_hat, op
    else:
        return Q_hat, R_hat


def QR_con_HH(A, tol=1e-12, extras=False):
    A = np.array(A, dtype=np.float64)
    m, n = np.shape(A)
    if m < n:
        return None

    R = np.copy(A)
    Q = np.eye(m)

    R_matrices = []
    Q_matrices = []

    for k in range(n):
        largo = m - k
        x = np.zeros(largo)
        for i in range(largo):
            x[i] = R[k + i, k]

        norma_x = 0.0
        for i in range(largo):
            norma_x += x[i] ** 2
        norma_x = norma_x ** 0.5

        if x[0] >= 0:
            signo = 1.0
        else:
            signo = -1.0
        alpha = -signo * norma_x

        u = np.copy(x)
        u[0] = u[0] - alpha

        norma_u = 0.0
        for i in range(largo):
            norma_u += u[i] ** 2
        norma_u = norma_u ** 0.5

        if norma_u > tol:
            for i in range(largo):
                u[i] = u[i] / norma_u

            # Hk = I_largo - 2 u u^T
            Hk = np.eye(largo)
            for i in range(largo):
                for j2 in range(largo):
                    Hk[i, j2] = Hk[i, j2] - 2 * u[i] * u[j2]

            # H_tilde_k: identidad de m x m con el bloque Hk metido abajo a la derecha
            H_tilde = np.eye(m)
            for i in range(largo):
                for j2 in range(largo):
                    H_tilde[k + i, k + j2] = Hk[i, j2]

            # R <- H_tilde @ R  (a mano)
            R_nueva = np.zeros((m, n))
            for i in range(m):
                for j2 in range(n):
                    suma = 0.0
                    for p in range(m):
                        suma += H_tilde[i, p] * R[p, j2]
                    R_nueva[i, j2] = suma
            R = R_nueva

            # Q <- Q @ H_tilde^T  (a mano)
            Q_nueva = np.zeros((m, m))
            for i in range(m):
                for j2 in range(m):
                    suma = 0.0
                    for p in range(m):
                        suma += Q[i, p] * H_tilde[j2, p]  # (H_tilde^T)[p,j2] = H_tilde[j2,p]
                    Q_nueva[i, j2] = suma
            Q = Q_nueva

        if extras:
            R_matrices.append(np.copy(R))
            Q_matrices.append(np.copy(Q))

    if extras:
        extra_info = {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
        return Q, R, extra_info
    else:
        return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol=tol)
    else:
        return None


# =========================================================
# LABORATORIO 6 - Autovalores y autovectores, metodo de la potencia
# =========================================================

def metpot2k(A, tol=1e-15, K=1000):
    A = np.array(A, dtype=np.float64)
    n = np.shape(A)[0]

    def matvec(x):
        resultado = np.zeros(n)
        for i in range(n):
            suma = 0.0
            for j in range(n):
                suma += A[i, j] * x[j]
            resultado[i] = suma
        return resultado

    def prod_interno(x, y):
        res = 0.0
        for i in range(n):
            res += x[i] * y[i]
        return res

    def fA(x):
        wp = matvec(x)
        norma_wp = 0.0
        for i in range(n):
            norma_wp += wp[i] ** 2
        norma_wp = norma_wp ** 0.5
        if norma_wp > 0:
            return wp / norma_wp
        else:
            return np.zeros(n)

    def fA2(x):
        return fA(fA(x))

    v = np.random.uniform(-1, 1, n)
    vtilde = fA2(v)
    e = prod_interno(vtilde, v)
    k = 0

    while abs(e - 1) > tol and k < K:
        v = vtilde
        vtilde = fA2(v)
        e = prod_interno(vtilde, v)
        k += 1

    lam = prod_interno(vtilde, matvec(vtilde))

    return vtilde, lam, k


def diagRH(A, tol=1e-15, K=1000):
    A = np.array(A, dtype=np.float64)
    if np.ndim(A) != 2:
        return None
    n, m = np.shape(A)
    if n != m:
        return None
    for i in range(n):
        for j in range(n):
            if abs(A[i, j] - A[j, i]) > 1e-8:
                return None

    def matmul(X, Y):
        filasX, colsX = np.shape(X)
        _, colsY = np.shape(Y)
        resultado = np.zeros((filasX, colsY))
        for i in range(filasX):
            for j in range(colsY):
                suma = 0.0
                for p in range(colsX):
                    suma += X[i, p] * Y[p, j]
                resultado[i, j] = suma
        return resultado

    v1, lam1, k1 = metpot2k(A, tol, K)

    e1 = np.zeros(n)
    e1[0] = 1.0
    dif = e1 - v1

    norma_dif2 = 0.0
    for i in range(n):
        norma_dif2 += dif[i] ** 2

    Hv1 = np.eye(n)
    if norma_dif2 > 0:
        for i in range(n):
            for j in range(n):
                Hv1[i, j] = Hv1[i, j] - 2 * dif[i] * dif[j] / norma_dif2

    B = matmul(matmul(Hv1, A), Hv1.T)

    if n == 2:
        S = Hv1
        D = B
        return S, D

    Atilde = B[1:n, 1:n]
    Stilde, Dtilde = diagRH(Atilde, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = lam1
    D[1:n, 1:n] = Dtilde

    bloque = np.eye(n)
    bloque[1:n, 1:n] = Stilde

    S = matmul(Hv1, bloque)

    return S, D


# =========================================================
# LABORATORIO 7 - Matrices de transicion, Markov
# =========================================================

def transiciones_al_azar_continuas(n):
    T = np.random.random((n, n))
    for j in range(n):
        suma_col = 0.0
        for i in range(n):
            suma_col += T[i, j]
        if suma_col > 0:
            for i in range(n):
                T[i, j] = T[i, j] / suma_col
    return T


def transiciones_al_azar_uniformes(n, thres):
    R = np.random.random((n, n))
    T = np.zeros((n, n))
    for j in range(n):
        indices_no_cero = []
        for i in range(n):
            if R[i, j] <= thres:
                indices_no_cero.append(i)
        if len(indices_no_cero) == 0:
            # columna nula: forzamos que quede al menos un elemento no nulo
            # (si no, la columna no suma 1 y T no es de Markov)
            idx_min = 0
            val_min = R[0, j]
            for i in range(1, n):
                if R[i, j] < val_min:
                    val_min = R[i, j]
                    idx_min = i
            indices_no_cero = [idx_min]
        valor = 1.0 / len(indices_no_cero)
        for i in indices_no_cero:
            T[i, j] = valor
    return T


def nucleo(A, tol=1e-15):
    A = np.array(A, dtype=np.float64)
    m, n = np.shape(A)

    def matmul(X, Y):
        filasX, colsX = np.shape(X)
        _, colsY = np.shape(Y)
        resultado = np.zeros((filasX, colsY))
        for i in range(filasX):
            for j in range(colsY):
                suma = 0.0
                for p in range(colsX):
                    suma += X[i, p] * Y[p, j]
                resultado[i, j] = suma
        return resultado

    AtA = matmul(A.T, A)

    resultado_diag = diagRH(AtA, tol=tol)
    if resultado_diag is None:
        return None
    S, D = resultado_diag

    vectores = []
    for i in range(n):
        if abs(D[i, i]) <= tol:
            vectores.append(S[:, i])

    return np.array(vectores).T


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    A_dict = {}
    if len(listado) == 0:
        return [A_dict, (m_filas, n_columnas)]

    indices_i, indices_j, valores = listado
    cantidad = len(indices_i)
    for k in range(cantidad):
        i = indices_i[k]
        j = indices_j[k]
        val = valores[k]
        if abs(val) >= tol:
            A_dict[(i, j)] = val

    return [A_dict, (m_filas, n_columnas)]


def multiplica_rala_vector(A, v):
    A_dict, (m_filas, n_columnas) = A
    es_lista = isinstance(v, list)
    v_arr = np.array(v, dtype=np.float64)

    w = np.zeros(m_filas)
    for (i, j), valor in A_dict.items():
        w[i] = w[i] + valor * v_arr[j]

    if es_lista:
        return list(w)
    else:
        return w

# =========================================================
# LABORATORIO 8 - Descomposicion SVD
# =========================================================

def svd_reducida(A, k="max", tol=1e-15):
    A = np.array(A, dtype=np.float64)
    m, n = np.shape(A)

    def matmul(X, Y):
        filasX, colsX = np.shape(X)
        _, colsY = np.shape(Y)
        resultado = np.zeros((filasX, colsY))
        for i in range(filasX):
            for j in range(colsY):
                suma = 0.0
                for p in range(colsX):
                    suma += X[i, p] * Y[p, j]
                resultado[i, j] = suma
        return resultado

    def norma_vec(v):
        s = 0.0
        largo = np.shape(v)[0]
        for i in range(largo):
            s += v[i] ** 2
        return s ** 0.5

    if m >= n:
        X = A
    else:
        X = A.T

    XtX = matmul(X.T, X)

    resultado_diag = diagRH(XtX, tol=tol)
    if resultado_diag is None:
        return None
    S, D = resultado_diag

    p = np.shape(XtX)[0]
    r_max = p if k == "max" else min(k, p)

    r_eff = 0
    for i in range(p):
        if r_eff >= r_max:
            break
        if D[i, i] <= tol:
            break
        r_eff += 1

    V_chico = S[:, :r_eff]

    B = matmul(X, V_chico)

    filas_B = np.shape(B)[0]
    sigma = np.zeros(r_eff)
    U_grande = np.zeros((filas_B, r_eff))
    for i in range(r_eff):
        s_i = norma_vec(B[:, i])
        sigma[i] = s_i
        U_grande[:, i] = B[:, i] / s_i

    if m >= n:
        hatU = U_grande
        hatV = V_chico
    else:
        hatU = V_chico
        hatV = U_grande

    return hatU, sigma, hatV

#%%

# =========================================================
# TRABAJO PRÁCTICO
# =========================================================

# =========================================================
# TP - EJERCICIO 2: Calibración con puntos 3D (DLT / Tsai)
# =========================================================

def obtenerQ3d(imgPts, objPts):
    imgPts = np.array(imgPts, dtype=np.float64)
    objPts = np.array(objPts, dtype=np.float64)
    n = np.shape(objPts)[0]

    Q = np.zeros((2 * n, 12))
    for i in range(n):
        u = imgPts[i, 0]
        v = imgPts[i, 1]

        # Punto en coordenadas homogéneas P_i = [U, V, W, 1]
        P = np.ones(4)
        P[0:3] = objPts[i, :]

        # Fila de la ecuación de u:  [ P^T, 0^T, -u P^T ]
        Q[2 * i, 0:4] = P
        Q[2 * i, 8:12] = -u * P

        # Fila de la ecuación de v:  [ 0^T, P^T, -v P^T ]
        Q[2 * i + 1, 4:8] = P
        Q[2 * i + 1, 8:12] = -v * P

    return Q


def resolverHomogeneo(Q, tol=1e-14, maxIt=1000):
    Q = np.array(Q, dtype=np.float64)

    # Q^T Q es n x n, simétrica y semidefinida positiva
    QtQ = multiplicar(Q.T, Q)
    n = np.shape(QtQ)[0]

    # Factorizamos una sola vez: Q^T Q = L U
    L, U, _ = calculaLU(QtQ)

    # Vector inicial cualquiera, normalizado
    x = np.ones(n) / np.sqrt(n)

    for _ in range(maxIt):
        # Resolvemos (Q^T Q) y = x  ->  L z = x  y luego  U y = z
        z = res_tri(L, x, inferior=True)
        y = res_tri(U, z, inferior=False)
        y = y / norma(y, 2)

        # Si x e y tienen la misma dirección (salvo signo), convergió
        if 1 - np.abs(np.sum(x * y)) < tol:
            x = y
            break
        x = y

    return x


def resolverDLT3D(Q, tol=1e-14, maxIt=1000):
    m = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(m, (3, 4))


def resolverDLT2D(Q, tol=1e-14, maxIt=1000):
    h = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(h, (3, 3))


def devolverMatricesCalibracion3d(M, alfasIguales=False, sinSesgo=False):
    M = np.array(M, dtype=np.float64)
    a1 = M[0, 0:3]
    a2 = M[1, 0:3]
    a3 = M[2, 0:3]
    b = M[:, 3]

    # 1) Escala: ||rho a3|| = 1. El signo se elige para que t3 = rho*b3 > 0
    #    (la mira queda delante de la cámara)
    rho = 1 / norma(a3, 2)
    if rho * b[2] < 0:
        rho = -rho

    # 2) Tercera fila de R
    r3 = rho * a3

    # 3) Punto principal: componente de las filas 1 y 2 en la dirección de r3
    u0 = np.sum(rho * a1 * r3)
    v0 = np.sum(rho * a2 * r3)

    # 4) Fila 2:  rho a2 = alpha_v r2 + v0 r3
    w2 = rho * a2 - v0 * r3
    alfa_v = norma(w2, 2)
    r2 = w2 / alfa_v

    # 5) Fila 1:  rho a1 = alpha_u r1 + gamma r2 + u0 r3
    gamma = np.sum(rho * a1 * r2)
    w1 = rho * a1 - gamma * r2 - u0 * r3
    alfa_u = norma(w1, 2)
    r1 = w1 / alfa_u

    # 6) (d) Restricciones opcionales sobre K
    if sinSesgo:
        gamma = 0.0
    if alfasIguales:
        promedio = (alfa_u + alfa_v) / 2
        alfa_u = promedio
        alfa_v = promedio

    K = np.array([[alfa_u, gamma,  u0],
                  [0.0,    alfa_v, v0],
                  [0.0,    0.0,    1.0]])

    if sinSesgo or alfasIguales:
        # Con la K modificada, recalculamos R resolviendo K R = rho A
        # columna por columna (sistema triangular, sin invertir K)
        A = M[:, 0:3]
        Rnueva = np.zeros((3, 3))
        for j in range(3):
            Rnueva[:, j] = res_tri(K, rho * A[:, j], inferior=False)

        # Re-ortonormalizamos con Gram-Schmidt desde r3 hacia arriba
        r3 = Rnueva[2, :] / norma(Rnueva[2, :], 2)
        w2 = Rnueva[1, :] - np.sum(Rnueva[1, :] * r3) * r3
        r2 = w2 / norma(w2, 2)
        w1 = Rnueva[0, :] - np.sum(Rnueva[0, :] * r3) * r3 - np.sum(Rnueva[0, :] * r2) * r2
        r1 = w1 / norma(w1, 2)

    # Armamos R con las filas r1, r2, r3
    R = np.zeros((3, 3))
    R[0, :] = r1
    R[1, :] = r2
    R[2, :] = r3

    # 7) Traslación: rho b = K t  ->  sistema triangular superior
    t = res_tri(K, rho * b, inferior=False)

    return R, t, K

#%%

# EJERCICIO 3 ==============
def obtenerQ2d(imgPts, objPts):
    imgPts = np.array(imgPts, dtype=np.float64)
    objPts = np.array(objPts, dtype=np.float64)

    n = imgPts.shape[0]          # cantidad de puntos
    Q = np.zeros((2 * n, 9))

    for i in range(n):
        U = objPts[i, 0]
        V = objPts[i, 1]
        u = imgPts[i, 0]
        v = imgPts[i, 1]

        # fila asociada a la coordenada u
        Q[2 * i, :] = np.array([U, V, 1, 0, 0, 0, -u * U, -u * V, -u])

        # fila asociada a la coordenada v
        Q[2 * i + 1, :] = np.array([0, 0, 0, U, V, 1, -v * U, -v * V, -v])

    return Q

#Auxiliar
def calcularV(H, i, j):
    H = np.array(H, dtype=np.float64)
    ci = H[:, i]
    cj = H[:, j]

    v = np.array([
        ci[0] * cj[0],                    # coeficiente de B11
        ci[0] * cj[1] + ci[1] * cj[0],    # coeficiente de B12 (B12 = B21)
        ci[1] * cj[1],                    # coeficiente de B22
        ci[2] * cj[0] + ci[0] * cj[2],    # coeficiente de B13 (B13 = B31)
        ci[2] * cj[1] + ci[1] * cj[2],    # coeficiente de B23 (B23 = B32)
        ci[2] * cj[2]                     # coeficiente de B33
    ])
    return v


def armarA(Hlst):
    N = len(Hlst)
    A = np.zeros((2 * N, 6))

    for j in range(N):
        H = Hlst[j]

        # vectores v para las columnas 1 y 2 de H (índices 0 y 1 en Python)
        v12 = calcularV(H, 0, 1)
        v11 = calcularV(H, 0, 0)
        v22 = calcularV(H, 1, 1)

        # restricción de perpendicularidad
        A[2 * j, :] = v12
        # restricción de igual norma
        A[2 * j + 1, :] = v11 - v22

    return A

def calcularMatrizK2D(A):
    A = np.array(A, dtype=np.float64)

    # Escalamos A para que su mayor elemento valga 1.
    # No cambia la solución de A b = 0, pero evita números enormes en A^T A
    # que hacen fallar a diagRH por errores de redondeo.
    A = A / np.max(np.abs(A))

    # 1) Resolver A b = 0. Reutilizamos resolverDLT3D (misma lógica: autovector
    #    del menor autovalor). Devuelve 3x2, así que lo volvemos a vector de 6.
    b = resolverDLT3D(A).reshape(-1)

    # 2) Armar B simétrica. b = (B11, B12, B22, B13, B23, B33)
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # b sale a menos de un signo: si B no es definida positiva, la damos vuelta
    if not esSDP(B, atol=0):
        B = -B

    # 3) Cholesky a partir de B = L D L^T  =>  B = (L sqrt(D)) (L sqrt(D))^T
    L, D, V = calculaLDV(B)
    raizD = np.zeros((3, 3))
    for i in range(3):
        raizD[i, i] = np.sqrt(D[i, i])
    L_chol = multiplicar(L, raizD)

    # Como L_chol = K^-T, entonces K = (L_chol^T)^-1
    K = inversa(L_chol.T)

    # 4) Normalizar para que K[2,2] = 1
    K = K / K[2, 2]

    return K

#Auxiliar
def productoVectorial(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def obtenerPose2D(Hj, K):
    Hj = np.array(Hj, dtype=np.float64)
    Kinv = inversa(K)

    # K^-1 por cada columna de H: da s*r1, s*r2 y s*t
    a1 = multiplicar(Kinv, Hj[:, 0]).reshape(-1)
    a2 = multiplicar(Kinv, Hj[:, 1]).reshape(-1)
    a3 = multiplicar(Kinv, Hj[:, 2]).reshape(-1)

    # Factor de escala: r1 tiene que tener norma 1
    lam = 1 / norma(a1, 2)

    # H viene a menos de un signo. Elegimos el signo que deja el tablero
    # delante de la cámara (componente z de t positiva)
    if a3[2] < 0:
        lam = -lam

    r1 = lam * a1
    r2 = lam * a2
    r3 = productoVectorial(r1, r2)   # tercer eje, perpendicular a r1 y r2
    t = lam * a3

    R = np.zeros((3, 3))
    R[:, 0] = r1
    R[:, 1] = r2
    R[:, 2] = r3

    # Con datos reales (ruido) R no queda exactamente ortonormal.
    # Imponemos la restricción ortonormalizando sus columnas con Gram-Schmidt.
    R, _ = QR_con_GS(R)

    return R, t

