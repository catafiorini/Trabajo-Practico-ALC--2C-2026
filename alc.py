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
        Ax = multiplicar(A, x)
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

    B = multiplicar(multiplicar(Hv1, A), Hv1.T)

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

    S = multiplicar(Hv1, bloque)

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

    AtA = multiplicar(A.T, A)

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

    XtX = multiplicar(X.T, X)

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

    B = multiplicar(X, V_chico)

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


def resolverHomogeneo(Q):
    Q = np.array(Q, dtype=np.float64)
    
    QtQ = multiplicar(Q.T, Q)
    
    S, D = diagRH(QtQ)
    
    # Como diagRH saca primero los autovalores más grandes, la última columna de S siempre corresponde al autovalor más chico.
    # S es la matriz V de la descomposicion SVD de Q
    m = S[:, -1]
    
    return m


def resolverDLT3D(Q, tol=1e-14, maxIt=1000):
    m = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(m, (3, 4))


def resolverDLT2D(Q, tol=1e-14, maxIt=1000):
    h = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(h, (3, 3))

#Auxiliar
def productoVectorial(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def devolverMatricesCalibracion3d(M, alfasIguales=False, sinSesgo=False):
    M = np.array(M, dtype=np.float64)
    
    # Partición M = [A | b]
    A = M[:, 0:3]
    b = M[:, 3]
    
    # Matriz de permutación P para invertir el orden de filas y columnas
    P = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0]])
    
    # Factorización RQ:
    # X = P @ A^T @ P = Q_qr @ R_qr
    X = multiplicar(multiplicar(P, A.T), P)
    Q_qr, R_qr = calculaQR(X, metodo='RH')
    
    K_raw = multiplicar(multiplicar(P, R_qr.T), P)
    R_raw = multiplicar(multiplicar(P, Q_qr.T), P)
    
    # Ajustamos los signos de la diagonal de K para asegurar alpha_u > 0 y alpha_v > 0
    for i in range(3):
        if K_raw[i, i] < 0:
            K_raw[:, i] = -K_raw[:, i]
            R_raw[i, :] = -R_raw[i, :]
            
    # Normalizamos K para que K[2,2] = 1 (obteniendo el factor de escala rho)
    rho = K_raw[2, 2]
    K = K_raw / rho
    
    # Verificamos que R sea una rotación y no una reflexión (det(R) = +1 y no det(R) = -1)
    r1, r2, r3 = R_raw[0, :], R_raw[1, :], R_raw[2, :]
    det_R = np.sum(r1 * productoVectorial(r2, r3))
    
    if det_R < 0:
        R_raw = -R_raw
        b = -b
        
    R = R_raw
    
    # Calculamos el vector de traslación t resolviendo K@t = b / rho
    t = res_tri(K, b / rho, inferior=False)
    
    # Restricciones opcionales sobre K (sinSesgo y/o alfasIguales)
    if sinSesgo or alfasIguales:
        if sinSesgo:
            K[0, 1] = 0.0
        if alfasIguales:
            promedio = (K[0, 0] + K[1, 1]) / 2.0
            K[0, 0] = promedio
            K[1, 1] = promedio
            
        # Recalcular R resolviendo K@R = A / rho columna por columna
        A_norm = A / rho
        R_nueva = np.zeros((3, 3))
        for j in range(3):
            R_nueva[:, j] = res_tri(K, A_norm[:, j], inferior=False)
            
        # Re-ortonormalizar R usando QR
        Q_r, _ = calculaQR(R_nueva, metodo='RH')
        det_R_nueva = np.sum(Q_r[0, :] * productoVectorial(Q_r[1, :], Q_r[2, :]))
        if det_R_nueva < 0:
            Q_r = -Q_r
        R = Q_r
        
        # Recalcular t con la matriz K modificada
        t = res_tri(K, b / rho, inferior=False)
        
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
    b = resolverHomogeneo(A)

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

#%%
# TP - EJERCICIO 6: Aterrizaje suave (POSIT)

def calculoLambdaMu(I0, J0):
    I0 = np.array(I0, dtype=np.float64)
    J0 = np.array(J0, dtype=np.float64)

    # Parte real e imaginaria de C^2
    a = np.sum(J0 * J0) - np.sum(I0 * I0)    # ||J0||^2 - ||I0||^2
    b = -2 * np.sum(I0 * J0)                 # -2 (I0 . J0)

    # Módulo de C^2
    modulo = np.sqrt(a ** 2 + b ** 2)

    # Raíz cuadrada del complejo a + ib (fórmula cerrada)
    lam = np.sqrt((modulo + a) / 2)
    mu = np.sqrt((modulo - a) / 2)

    # mu lleva el signo de b, para que 2*lam*mu = b
    if b < 0:
        mu = -mu

    return lam, mu



def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    Is = np.array(Is, dtype=np.float64)
    Js = np.array(Js, dtype=np.float64)
    m0 = np.array(m0, dtype=np.float64)
    M0 = np.array(M0, dtype=np.float64)

    # 1) Profundidad: ||I|| = ||J|| = f / Z0 (promediamos por redondeo)
    normaI = norma(Is, 2)
    normaJ = norma(Js, 2)
    Z0 = fx / ((normaI + normaJ) / 2)

    # 2) Ejes de la cámara: normalizamos I y J, y k = i x j
    i = Is / normaI
    j = Js / normaJ
    k = productoVectorial(i, j)

    # 3) Rotación: filas i, j, k
    R = np.zeros((3, 3))
    R[0, :] = i
    R[1, :] = j
    R[2, :] = k

    # 4) Posición de M0 en coordenadas de cámara, a partir de su imagen m0
    T = np.array([Z0 / fx * m0[0],
                  Z0 / fx * m0[1],
                  Z0])

    # 5) Traslación del modelo X = R M + t:  R M0 + t = T  =>  t = T - R M0
    t = T - multiplicar(R, M0).reshape(-1)

    return i, j, k, Z0, R, t

def evaluarPose(R, t, obj_vecs, img_vecs, K):
    R = np.array(R, dtype=np.float64)
    t = np.array(t, dtype=np.float64)
    obj_vecs = np.array(obj_vecs, dtype=np.float64)
    img_vecs = np.array(img_vecs, dtype=np.float64)
    K = np.array(K, dtype=np.float64)

    n = np.shape(obj_vecs)[0]
    validos = 0
    suma_err2 = 0.0

    for p in range(n):
        # 1) Punto en coordenadas de cámara: X = R M + t
        Xc = multiplicar(R, obj_vecs[p, :]).reshape(-1) + t

        # 2) Proyección válida si el punto queda delante de la cámara
        if Xc[2] > 0:
            validos += 1

        # 3) Proyección a la imagen: K X = (lambda u, lambda v, lambda)
        proy = multiplicar(K, Xc).reshape(-1)
        u_proy = proy[0] / proy[2]
        v_proy = proy[1] / proy[2]

        # 4) Error al cuadrado contra el punto observado
        du = img_vecs[p, 0] - u_proy
        dv = img_vecs[p, 1] - v_proy
        suma_err2 += du ** 2 + dv ** 2

    # RMS = raíz del promedio de los errores al cuadrado
    rms = np.sqrt(suma_err2 / n)

    return validos, rms

# Auxiliar, para que quede mas legible
def resolverPOS(A, bu, bv):
    A = np.array(A, dtype=np.float64)
    bu = np.array(bu, dtype=np.float64)
    bv = np.array(bv, dtype=np.float64)

    # Nos quedamos con las columnas U y V (la de W es toda cero)
    A2 = A[:, 0:2]

    # Ecuaciones normales: (A2^T A2) x = A2^T b
    AtA = multiplicar(A2.T, A2)                   # 2 x 2
    Atbu = multiplicar(A2.T, bu).reshape(-1)      # vector de 2
    Atbv = multiplicar(A2.T, bv).reshape(-1)      # vector de 2

    # Factorizamos una sola vez A2^T A2 = L U (sirve para I y para J)
    L, U, _ = calculaLU(AtA)

    # Resolvemos L z = A2^T b y luego U x = z, para I y para J
    I2 = res_tri(U, res_tri(L, Atbu, inferior=True), inferior=False)
    J2 = res_tri(U, res_tri(L, Atbv, inferior=True), inferior=False)

    # Completamos con 0 la componente en la dirección normal n = (0, 0, 1)
    I0 = np.zeros(3)
    I0[0:2] = I2
    J0 = np.zeros(3)
    J0[0:2] = J2

    return I0, J0

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt=100, atol=1e-6):
    obj_vecs = np.array(obj_vecs, dtype=np.float64)
    img_vecs = np.array(img_vecs, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    n = np.shape(obj_vecs)[0]

    # --- Preparación ---
    # Parámetros de la cámara
    fx = K[0, 0]
    fy = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    # Coordenadas de imagen centradas. La y se lleva a la escala de fx
    # para poder usar una sola focal en todo el algoritmo.
    x = img_vecs[:, 0] - cu
    y = (img_vecs[:, 1] - cv) * (fx / fy)

    # Punto de referencia M0 (el primero) y su imagen m0 = (x0, y0)
    M0 = obj_vecs[0, :]
    m0 = np.array([x[0], y[0]])

    # Matriz A: filas a_i = Mi - M0 (sin M0, que daría una fila nula)
    A = obj_vecs[1:n, :] - M0
    xs = x[1:n]
    ys = y[1:n]

    # Normal al plano del objeto (W = 0)
    normal = np.array([0.0, 0.0, 1.0])

    # Mejor pose entre las dos ramas
    R_final = None
    t_final = None
    rms_final = None

    # --- Una corrida por cada rama inicial: 0 = (+lam, +mu), 1 = (-lam, -mu) ---
    for rama in range(2):
        # Al principio suponemos proyección ortográfica escalada: eps_i = 0
        eps = np.zeros(n - 1)

        R_mejor = None
        t_mejor = None
        rms_mejor = None
        rms_anterior = None

        for it in range(MaxIt):
            # 1) Lados derechos con los eps actuales
            bu = xs * (1 + eps) - m0[0]
            bv = ys * (1 + eps) - m0[1]

            # 2) Paso POS
            I0, J0 = resolverPOS(A, bu, bv)

            # 3) Las dos soluciones del caso coplanar
            lam, mu = calculoLambdaMu(I0, J0)
            candidatos = [(I0 + lam * normal, J0 + mu * normal),
                          (I0 - lam * normal, J0 - mu * normal)]

            # En la primera iteración solo se usa la solución de esta rama
            if it == 0:
                candidatos = [candidatos[rama]]

            # 4) Evaluamos los candidatos y nos quedamos con el mejor válido
            rms_it = None
            for (Is, Js) in candidatos:
                i, j, k, Z0, R, t = calcularSalidaPOSIT(Is, Js, fx, m0, M0)
                validos, rms = evaluarPose(R, t, obj_vecs, img_vecs, K)

                # Si algún punto queda detrás de la cámara, se descarta
                if validos < n:
                    continue

                if rms_it is None or rms < rms_it:
                    rms_it = rms
                    R_it = R
                    t_it = t
                    k_it = k
                    Z0_it = Z0

            # Si ninguna solución es válida, cortamos esta rama
            if rms_it is None:
                break

            R_mejor = R_it
            t_mejor = t_it
            rms_mejor = rms_it

            # 5) Nuevos factores de corrección de perspectiva: eps_i = (a_i . k) / Z0
            for p in range(n - 1):
                eps[p] = np.sum(A[p, :] * k_it) / Z0_it

            # 6) Convergencia: el RMS ya no mejora
            if rms_anterior is not None and np.abs(rms_anterior - rms_it) < atol:
                break
            rms_anterior = rms_it

        # Nos quedamos con la rama de menor RMS final
        if rms_mejor is not None and (rms_final is None or rms_mejor < rms_final):
            R_final = R_mejor
            t_final = t_mejor
            rms_final = rms_mejor

    return R_final, t_final