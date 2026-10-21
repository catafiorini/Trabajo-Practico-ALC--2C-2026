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
