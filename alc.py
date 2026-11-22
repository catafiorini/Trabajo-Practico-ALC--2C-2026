import numpy as np

# =========================================================
# LABORATORIO 1 - Numeros de maquina
# =========================================================

def error(x, y):
    """
    Calcula el error absoluto de aproximar x usando y.

    Parámetros:
        x: valor exacto (número o array).
        y: valor aproximado (número o array).

    Devuelve:
        |x - y| (elemento a elemento si son arrays).
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return abs(x - y)


def error_relativo(x, y):
    """
    Calcula el error relativo de aproximar x usando y.

    Parámetros:
        x: valor exacto, distinto de 0 (número o array).
        y: valor aproximado (número o array).

    Devuelve:
        |x - y| / |x| (elemento a elemento si son arrays).
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return error(x, y) / abs(x)


def matricesIguales(A, B):
    """
    Decide si dos matrices son iguales, comparando elemento a elemento
    con una tolerancia absoluta de 1e-8 (para no fallar por errores de redondeo).

    Parámetros:
        A, B: matrices (listas o arrays 2D).

    Devuelve:
        True si tienen la misma forma y todos sus elementos difieren en
        menos de 1e-8; False en caso contrario.
    """
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
    """
    Multiplica dos matrices con el algoritmo de tres ciclos:
    C[i, j] = suma sobre k de A[i, k] * B[k, j].

    Si A es un vector (1D) se lo toma como vector fila, y si B es un
    vector (1D) se lo toma como vector columna.

    Parámetros:
        A: matriz de m x n (o vector de largo n).
        B: matriz de n x p (o vector de largo n).

    Devuelve:
        C: matriz de m x p (siempre 2D; un vector columna sale como n x 1).
    """
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
    """
    Devuelve la matriz de rotación de R2 en un ángulo theta (sentido antihorario):
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]

    Parámetros:
        theta: ángulo en radianes.

    Devuelve:
        Matriz de 2 x 2.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def escala(s):
    """
    Devuelve la matriz diagonal que escala cada coordenada i por s[i].

    Parámetros:
        s: vector de largo n con los factores de escala.

    Devuelve:
        Matriz diagonal de n x n con s en la diagonal.
    """
    s = np.array(s, dtype=np.float64)
    n = np.shape(s)[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = s[i]
    return T


def rota_y_escala(theta, s):
    """
    Devuelve la matriz que primero rota en un ángulo theta y después escala por s.
    Como se aplica primero la rotación, el producto es escala(s) · rota(theta).

    Parámetros:
        theta: ángulo en radianes.
        s: vector de largo 2 con los factores de escala.

    Devuelve:
        Matriz de 2 x 2.
    """
    return multiplicar(escala(s), rota(theta))


def afin(theta, s, b):
    """
    Devuelve la matriz de la transformación afín (en coordenadas homogéneas)
    que rota en theta, escala por s y traslada por b:
        [[ A  b ],
         [ 0  1 ]]   con A = rota_y_escala(theta, s).

    Usar coordenadas homogéneas permite escribir la traslación, que no es
    lineal en R2, como un producto de matrices en R3.

    Parámetros:
        theta: ángulo en radianes.
        s: vector de largo 2 con los factores de escala.
        b: vector de largo 2 con la traslación.

    Devuelve:
        Matriz de 3 x 3.
    """
    A = rota_y_escala(theta, s)
    b = np.array(b, dtype=np.float64)

    M = np.zeros((3, 3))
    M[0:2, 0:2] = A
    M[0:2, 2] = b
    M[2, 2] = 1

    return M


def trans_afin(v, theta, s, b):
    """
    Aplica la transformación afín (rotar en theta, escalar por s, trasladar por b)
    al vector v.

    Pasa v a coordenadas homogéneas (v1, v2, 1), lo multiplica por afin(theta, s, b)
    y devuelve las dos primeras coordenadas.

    Parámetros:
        v: vector de largo 2.
        theta: ángulo en radianes.
        s: vector de largo 2 con los factores de escala.
        b: vector de largo 2 con la traslación.

    Devuelve:
        w: vector de largo 2 transformado.
    """
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
    """
    Calcula la norma p de un vector:
        ||x||_p = (suma de |x_i|^p)^(1/p)
        ||x||_inf = max |x_i|

    Parámetros:
        x: vector.
        p: número positivo, o 'inf' para la norma infinito.

    Devuelve:
        La norma (un número).
    """
    x = np.array(x, dtype=np.float64)
    n = np.shape(x)[0]

    if p == 'inf':
        return max(abs(x[i]) for i in range(n))

    res = 0.0
    for i in range(n):
        res += abs(x[i]) ** p
    return res ** (1 / p)


def normaliza(X, p):
    """
    Normaliza cada vector de una lista para que tenga norma p igual a 1.

    Parámetros:
        X: lista de vectores (no nulos).
        p: número positivo, o 'inf'.

    Devuelve:
        Lista con los vectores x / ||x||_p.
    """
    resultado = []
    for x in X:
        x = np.array(x, dtype=np.float64)
        resultado.append(x / norma(x, p))
    return resultado


def normaMatMC(A, q, p, Np):
    """
    Estima por Monte Carlo la norma inducida de A:
        ||A||_{q,p} = max ||A x||_q  sobre los x con ||x||_p = 1.

    Genera Np vectores al azar, los normaliza con la norma p y se queda con
    el que hace más grande a ||A x||_q. Es una cota inferior de la norma real,
    que mejora a medida que aumenta Np.

    Parámetros:
        A: matriz de m x n.
        q: norma usada en la llegada (para A x).
        p: norma usada en la salida (para x).
        Np: cantidad de vectores al azar.

    Devuelve:
        (max_norma, x_max): la norma estimada y el vector donde se alcanzó.
    """
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
    """
    Calcula de forma exacta la norma 1 o la norma infinito de una matriz:
        ||A||_1   = máxima suma de valores absolutos por columna.
        ||A||_inf = máxima suma de valores absolutos por fila.

    Parámetros:
        A: matriz (array 2D).
        valor: 1 o 'inf'.

    Devuelve:
        La norma, o None si valor no es 1 ni 'inf'.
    """
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
    """
    Calcula de forma exacta la norma 1 y/o la norma infinito de una matriz
    (usa normaExactaValor).

    Parámetros:
        A: matriz.
        p: 1, 'inf' o una lista con esos valores (por defecto [1, 'inf']).

    Devuelve:
        Si p es una lista, una lista con cada norma; si no, un solo número.
    """
    A = np.array(A, dtype=np.float64)

    if isinstance(p, list):
        return [normaExactaValor(A, valor) for valor in p]
    else:
        return normaExactaValor(A, p)


def condMC(A, p, Np=1000):
    """
    Estima por Monte Carlo el número de condición de A en norma p:
        cond_p(A) = ||A||_p · ||A^-1||_p

    Mide cuánto se puede amplificar el error relativo al resolver A x = b.

    Parámetros:
        A: matriz cuadrada inversible.
        p: norma a usar.
        Np: cantidad de vectores al azar para cada norma (por defecto 1000).

    Devuelve:
        El número de condición estimado (cota inferior del real).
    """
    A = np.array(A, dtype=np.float64)
    Ainv = inversa(A)
    normaA, _ = normaMatMC(A, p, p, Np)
    normaAinv, _ = normaMatMC(Ainv, p, p, Np)
    return normaA * normaAinv


def condExacta(A, p):
    """
    Calcula de forma exacta el número de condición de A en norma 1 o infinito:
        cond_p(A) = ||A||_p · ||A^-1||_p

    Parámetros:
        A: matriz cuadrada inversible.
        p: 1 o 'inf'.

    Devuelve:
        El número de condición.
    """
    A = np.array(A, dtype=np.float64)
    Ainv = inversa(A)
    return normaExacta(A, p) * normaExacta(Ainv, p)


# =========================================================
# LABORATORIO 4 - Descomposicion LU
# =========================================================

def calculaLU(A):
    """
    Calcula la factorización A = L U por eliminación gaussiana, sin pivoteo.

    En cada paso k se calculan los multiplicadores m_ik = A[i, k] / A[k, k],
    que se guardan en L, y se eliminan los elementos debajo del pivote,
    lo que queda en U.

    Parámetros:
        A: matriz cuadrada.

    Devuelve:
        (L, U, op):
            L: triangular inferior con unos en la diagonal.
            U: triangular superior.
            op: cantidad de operaciones realizadas.
        Si A no es cuadrada o aparece un pivote nulo, devuelve (None, None, 0).
    """
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
    """
    Resuelve un sistema triangular L x = b por sustitución.

        inferior=True:  sustitución hacia adelante (de la primera fila a la última).
        inferior=False: sustitución hacia atrás (de la última fila a la primera).

    Parámetros:
        L: matriz triangular de n x n con diagonal no nula.
        b: vector de largo n.
        inferior: True si L es triangular inferior, False si es superior.

    Devuelve:
        x: vector solución de largo n.
    """
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
    """
    Calcula la inversa de A usando su factorización LU.

    La columna i de A^-1 es la solución de A x = e_i. Como A = L U, se resuelve
    en dos pasos triangulares: L y = e_i y después U x = y.

    Parámetros:
        A: matriz cuadrada.

    Devuelve:
        A^-1, o None si A es singular o no admite LU sin pivoteo.
    """
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
    """
    Calcula la factorización A = L D V, con L triangular inferior y V triangular
    superior (ambas con unos en la diagonal) y D diagonal.

    Se obtiene A = L U y luego se factoriza U^T = L2 U2. La diagonal de U2 es D
    y V = L2^T. Si A es simétrica, V = L^T.

    Parámetros:
        A: matriz cuadrada que admite LU sin pivoteo.

    Devuelve:
        (L, D, V).
    """
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
    """
    Decide si A es simétrica definida positiva.

    Primero chequea que A sea simétrica. Luego usa A = L D L^T: A es definida
    positiva si y solo si todos los elementos de D son positivos.

    Parámetros:
        A: matriz cuadrada.
        atol: tolerancia para la simetría y para considerar positivos
              los elementos de D (por defecto 1e-8).

    Devuelve:
        True si A es SDP, False en caso contrario.
    """
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
    """
    Calcula la factorización QR de A con el método de Gram-Schmidt.

    Cada columna de A se ortogonaliza restándole su proyección sobre las
    columnas de Q ya calculadas, y después se normaliza. Los coeficientes
    de esas proyecciones y las normas forman R. Si una columna queda con
    norma menor a tol (es combinación lineal de las anteriores), se descarta.

    Parámetros:
        A: matriz cuadrada de n x n.
        tol: tolerancia para descartar columnas dependientes.
        retorna_nops: si es True, también devuelve la cantidad de operaciones.

    Devuelve:
        (Q, R) o (Q, R, op):
            Q: n x r con columnas ortonormales (r = columnas no descartadas).
            R: r x n triangular superior.
        Devuelve None si A no es cuadrada.
    """
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
    """
    Calcula la factorización QR de A con reflexiones de Householder.

    En el paso k se construye una reflexión H_k = I - 2 u u^T que lleva la
    parte de la columna k que está desde la diagonal hacia abajo a un múltiplo
    de e_1, lo que anula los elementos debajo de la diagonal. Al final:
        R = H_n ... H_1 A    y    Q = H_1^T ... H_n^T.

    Parámetros:
        A: matriz de m x n con m >= n.
        tol: tolerancia para saltear reflexiones con u casi nulo.
        extras: si es True, también devuelve las Q y R intermedias.

    Devuelve:
        (Q, R) o (Q, R, extra_info):
            Q: m x m ortogonal.
            R: m x n triangular superior.
            extra_info: diccionario con las listas 'R_matrices' y 'Q_matrices'.
        Devuelve None si m < n.
    """
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
    """
    Calcula la factorización QR de A con el método elegido.

    Parámetros:
        A: matriz.
        metodo: 'RH' (reflexiones de Householder) o 'GS' (Gram-Schmidt).
        tol: tolerancia que se le pasa al método.

    Devuelve:
        (Q, R), o None si el método no es válido.
    """
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
    """
    Método de la potencia para el autovalor de mayor módulo de A y su autovector.

    En cada iteración se aplica A dos veces y se normaliza (o sea, se aplica A^2).
    Esto evita que el vector oscile de signo cuando el autovalor dominante
    es negativo. Se corta cuando el vector nuevo y el anterior son casi
    iguales (producto interno cercano a 1) o al llegar a K iteraciones.
    El autovalor se obtiene con el cociente de Rayleigh: lambda = v^T A v.

    Parámetros:
        A: matriz cuadrada.
        tol: tolerancia del criterio de parada.
        K: cantidad máxima de iteraciones.

    Devuelve:
        (v, lam, k): autovector de norma 1, autovalor y cantidad de iteraciones.
    """
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
    """
    Diagonaliza una matriz simétrica: A = S D S^T, con S ortogonal y D diagonal.

    Usa deflación con Householder:
        1) Con metpot2k se obtiene el autovalor dominante lambda_1 y su autovector v1.
        2) Se arma la reflexión H que lleva v1 a e_1, así H A H^T queda con
           lambda_1 en la posición (0, 0) y ceros en el resto de la primera
           fila y columna.
        3) Se repite recursivamente con la submatriz que queda.

    Los autovalores quedan en D ordenados de mayor a menor módulo, y los
    autovectores son las columnas de S. Antes de empezar se fuerza la simetría,
    (A + A^T) / 2, para eliminar el ruido de redondeo.

    Parámetros:
        A: matriz simétrica de n x n.
        tol, K: parámetros que se le pasan a metpot2k.

    Devuelve:
        (S, D), o None si A no es cuadrada o no es simétrica.
    """
    A = np.array(A, dtype=np.float64)
    if np.ndim(A) != 2:
        return None
    n, m = np.shape(A)
    if n != m:
        return None

    # CORRECCIÓN 1: Forzar simetría inicial para eliminar el ruido flotante acumulado
    A = (A + A.T) / 2.0

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
    # CORRECCIÓN 2: Forzar simetría de la submatriz antes de llamar a la recursión
    Atilde = (Atilde + Atilde.T) / 2.0

    # CORRECCIÓN 3: Desempaquetado seguro por si acaso
    resultado_rec = diagRH(Atilde, tol, K)
    if resultado_rec is None:
        return None
    Stilde, Dtilde = resultado_rec

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
    """
    Genera una matriz de transición de Markov al azar.

    Los elementos se toman uniformes en [0, 1) y cada columna se divide por su
    suma, así queda con elementos no negativos y columnas que suman 1.

    Parámetros:
        n: tamaño de la matriz.

    Devuelve:
        T: matriz de n x n estocástica por columnas.
    """
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
    """
    Genera una matriz de transición de Markov al azar con probabilidades uniformes.

    En cada columna, los elementos cuyo número al azar es <= thres son los no
    nulos, y todos valen 1 / (cantidad de no nulos). Si en una columna no queda
    ninguno, se fuerza uno para que la columna sume 1.

    Parámetros:
        n: tamaño de la matriz.
        thres: umbral en [0, 1]; cuanto más chico, más rala queda la matriz.

    Devuelve:
        T: matriz de n x n estocástica por columnas.
    """
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
    """
    Calcula una base del núcleo de A.

    Usa que Nu(A) = Nu(A^T A): diagonaliza A^T A con diagRH y se queda con los
    autovectores cuyo autovalor es (casi) nulo.

    Parámetros:
        A: matriz de m x n.
        tol: tolerancia para considerar nulo un autovalor.

    Devuelve:
        Matriz cuyas columnas son una base ortonormal del núcleo
        (vacía si el núcleo es {0}), o None si falla la diagonalización.
    """
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
    """
    Crea una matriz rala, guardando solo sus elementos no nulos en un diccionario.

    Parámetros:
        listado: lista [filas, columnas, valores], tres listas del mismo largo
                 donde el elemento k indica que A[filas[k], columnas[k]] = valores[k].
                 Puede ser una lista vacía.
        m_filas, n_columnas: dimensiones de la matriz.
        tol: los valores con módulo menor a tol no se guardan.

    Devuelve:
        [A_dict, (m_filas, n_columnas)], con A_dict[(i, j)] = valor.
    """
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
    """
    Multiplica una matriz rala (creada con crea_rala) por un vector.

    Solo recorre los elementos no nulos: por cada A[i, j] suma A[i, j] * v[j] en w[i].

    Parámetros:
        A: matriz rala [A_dict, (m_filas, n_columnas)].
        v: vector de largo n_columnas (lista o array).

    Devuelve:
        w = A v, de largo m_filas, del mismo tipo que v (lista o array).
    """
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
    """
    Calcula la descomposición en valores singulares reducida: A = U Sigma V^T.

    Si m >= n, diagonaliza A^T A con diagRH: sus autovectores son las columnas de V
    y sus autovalores son los sigma_i^2. Las columnas de U se obtienen como
    u_i = A v_i / sigma_i. Si m < n, se trabaja con A^T y después se intercambian
    U y V (así se diagonaliza siempre la matriz más chica).

    Parámetros:
        A: matriz de m x n.
        k: cantidad de valores singulares a calcular ("max" para todos).
        tol: los autovalores menores a tol se consideran nulos.

    Devuelve:
        (hatU, sigma, hatV):
            hatU: m x r con columnas ortonormales.
            sigma: vector con los r valores singulares no nulos, de mayor a menor.
            hatV: n x r con columnas ortonormales.
        Devuelve None si falla la diagonalización.
    """
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

# TP - EJERCICIO 2: Calibración con puntos 3D (DLT / Tsai)

def obtenerQ3d(imgPts, objPts):
    """
    Arma la matriz Q del sistema DLT para calibración con puntos 3D.

    Cada correspondencia (U, V, W) <-> (u, v) aporta 2 filas a Q, que salen de
    despejar u y v de la ecuación de proyección q = M P, con P = (U, V, W, 1):
        [ P^T,  0^T, -u P^T ]
        [ 0^T,  P^T, -v P^T ]

    Parámetros:
        imgPts: array de n x 2 con las coordenadas (u, v) en píxeles.
        objPts: array de n x 3 con las coordenadas (U, V, W) en el mundo.

    Devuelve:
        Q: array de 2n x 12. El vector m (M aplanada por filas) cumple Q m = 0.
    """
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
    """
    Resuelve el sistema homogéneo Q x = 0, buscando la solución no trivial
    de norma 1 que minimiza ||Q x||.

    Esa solución es el autovector de Q^T Q asociado a su menor autovalor
    (equivalente al último vector singular derecho de Q). Como diagRH devuelve
    los autovalores de mayor a menor, es la última columna de S.
    Con datos exactos da el núcleo de Q; con ruido, la solución de
    cuadrados mínimos.

    Parámetros:
        Q: matriz del sistema.
        tol, maxIt: no se usan (diagRH usa sus valores por defecto).

    Devuelve:
        x: vector de norma 1, definido a menos de un signo.
    """
    Q = np.array(Q, dtype=np.float64)
    
    QtQ = multiplicar(Q.T, Q)
    
    S, D = diagRH(QtQ)
    
    # Como diagRH saca primero los autovalores más grandes, la última columna de S siempre corresponde al autovalor más chico.
    # S es la matriz V de la descomposicion SVD de Q
    m = S[:, -1]
    
    return m


def resolverDLT3D(Q, tol=1e-14, maxIt=1000):
    """
    Resuelve el DLT 3D: obtiene la matriz de proyección M a partir de Q.

    Resuelve Q m = 0 con resolverHomogeneo y reacomoda m como una matriz de 3 x 4.

    Parámetros:
        Q: matriz de 2n x 12 (de obtenerQ3d).
        tol, maxIt: se le pasan a resolverHomogeneo.

    Devuelve:
        M: matriz de 3 x 4, definida a menos de un factor de escala.
    """
    m = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(m, (3, 4))


def resolverDLT2D(Q, tol=1e-14, maxIt=1000):
    """
    Resuelve el DLT 2D: obtiene la homografía H a partir de Q.

    Resuelve Q h = 0 con resolverHomogeneo y reacomoda h como una matriz de 3 x 3.

    Parámetros:
        Q: matriz de 2n x 9 (de obtenerQ2d).
        tol, maxIt: se le pasan a resolverHomogeneo.

    Devuelve:
        H: matriz de 3 x 3, definida a menos de un factor de escala.
    """
    h = resolverHomogeneo(Q, tol, maxIt)
    return np.reshape(h, (3, 3))

#Auxiliar
def productoVectorial(a, b):
    """
    Calcula el producto vectorial a x b en R3. El resultado es perpendicular
    a a y a b.

    Parámetros:
        a, b: vectores de largo 3.

    Devuelve:
        Vector de largo 3.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def devolverMatricesCalibracion3d(M, alfasIguales=False, sinSesgo=False):
    """
    Descompone la matriz de proyección M en parámetros intrínsecos y extrínsecos:
        M = rho · K [R | t]

    Se parte M = [A | b], con A = rho K R. Como K es triangular superior y R es
    ortogonal, A se factoriza con una descomposición RQ, que se obtiene a partir
    de QR usando la matriz de permutación P que invierte el orden de filas y
    columnas. Después:
        - se corrigen los signos para que la diagonal de K sea positiva,
        - se normaliza K para que K[2, 2] = 1 (eso da rho),
        - si det(R) = -1, se cambia el signo para que R sea una rotación,
        - t se obtiene resolviendo K t = b / rho.

    Restricciones opcionales sobre K:
        alfasIguales: reemplaza alfa_u y alfa_v por su promedio.
        sinSesgo: impone K[0, 1] = 0.
    Si se usa alguna, se recalcula R resolviendo K R = A / rho, se vuelve a
    ortonormalizar con QR, y se recalcula t.

    Parámetros:
        M: matriz de proyección de 3 x 4.
        alfasIguales: si es True, impone alfa_u = alfa_v.
        sinSesgo: si es True, impone K[0, 1] = 0.

    Devuelve:
        (R, t, K): rotación de 3 x 3, traslación de largo 3 e intrínsecos de 3 x 3.
    """
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
    """
    Arma la matriz Q del sistema DLT para estimar una homografía (caso coplanar).

    Si los puntos están en el plano W = 0, la proyección se reduce a q = H p,
    con p = (U, V, 1). Cada correspondencia (U, V) <-> (u, v) aporta 2 filas:
        [ U, V, 1, 0, 0, 0, -u U, -u V, -u ]
        [ 0, 0, 0, U, V, 1, -v U, -v V, -v ]

    Parámetros:
        imgPts: array de n x 2 con las coordenadas (u, v) en píxeles.
        objPts: array de n x 2 con las coordenadas (U, V) sobre el plano.

    Devuelve:
        Q: array de 2n x 9. El vector h (H aplanada por filas) cumple Q h = 0.
    """
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
    """
    Calcula el vector v_ij que permite escribir h_i^T B h_j como un producto
    escalar lineal en los elementos de B:
        h_i^T B h_j = v_ij · b,   con b = (B11, B12, B22, B13, B23, B33)

    donde h_i y h_j son columnas de H y B = K^-T K^-1 es simétrica
    (por eso tiene 6 incógnitas y no 9).

    Parámetros:
        H: homografía de 3 x 3.
        i, j: índices de las columnas de H (0, 1 o 2).

    Devuelve:
        v: vector de largo 6.
    """
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
    """
    Arma la matriz A del método de Zhang para estimar K a partir de homografías.

    Como H = s K [r1 r2 t] y r1, r2 son ortonormales, cada homografía da 2
    ecuaciones lineales en b (los elementos de B = K^-T K^-1):
        perpendicularidad:  h1^T B h2 = 0            ->  v12 · b = 0
        igual norma:        h1^T B h1 = h2^T B h2    ->  (v11 - v22) · b = 0

    Parámetros:
        Hlst: lista de N homografías de 3 x 3.

    Devuelve:
        A: matriz de 2N x 6, tal que A b = 0.
    """
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
    """
    Calcula la matriz de intrínsecos K a partir de la matriz A del método de Zhang.

    Pasos:
        1) Resuelve A b = 0 con resolverHomogeneo (antes escala A para que su
           mayor elemento sea 1, lo que mejora la estabilidad numérica).
        2) Arma la matriz simétrica B con b. Como b sale a menos de un signo,
           si B no es definida positiva se le cambia el signo.
        3) Como B = K^-T K^-1, se obtiene K^-T con la factorización de Cholesky
           de B, calculada a partir de B = L D L^T como L sqrt(D).
        4) Invierte y normaliza para que K[2, 2] = 1.

    Hacen falta al menos 3 homografías: K tiene 5 grados de libertad
    y cada homografía aporta 2 ecuaciones.

    Parámetros:
        A: matriz de 2N x 6 (de armarA).

    Devuelve:
        K: matriz de 3 x 3 triangular superior. Si por el ruido B no queda
           definida positiva, puede tener valores NaN.
    """
    A = np.array(A, dtype=np.float64)

    # Escalamos A para que su mayor elemento valga 1.
    # No cambia la solución de A b = 0, pero evita números enormes en A^T A
    # que hacen fallar a diagRH por errores de redondeo.
    A = A / np.max(np.abs(A))

    # 1) Resolver A b = 0 con resolverHomogeneo (autovector del menor autovalor).
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
    """
    Obtiene la pose (R, t) de la cámara a partir de una homografía y de K.

    Como H = s K [r1 r2 t], multiplicando por K^-1 cada columna de H se obtienen
    s r1, s r2 y s t. La escala s se fija pidiendo ||r1|| = 1, y su signo se
    elige para que el plano quede delante de la cámara (t_z > 0).
    La tercera columna es r3 = r1 x r2. Como con datos reales R no queda
    exactamente ortonormal, al final se ortonormaliza con Gram-Schmidt.

    Parámetros:
        Hj: homografía de 3 x 3 de la vista.
        K: matriz de intrínsecos de 3 x 3.

    Devuelve:
        (R, t): rotación de 3 x 3 y traslación de largo 3, tales que
                X_cámara = R X_mundo + t.
    """
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
    """
    Calcula lambda y mu para el caso coplanar de POSIT.

    Si los puntos son coplanares, el sistema POS deja indeterminada la componente
    de I y J en la dirección normal n al plano:
        I = I0 + lambda n,   J = J0 + mu n.
    Imponiendo I · J = 0 y ||I|| = ||J|| se llega a:
        lambda^2 - mu^2 = ||J0||^2 - ||I0||^2
        2 lambda mu    = -2 (I0 · J0)
    o sea, (lambda + i mu)^2 = C^2 con C^2 = a + i b. Se resuelve con la fórmula
    de la raíz cuadrada de un número complejo.

    Parámetros:
        I0, J0: vectores de largo 3 (de resolverPOS).

    Devuelve:
        (lam, mu): una de las dos raíces. La otra es (-lam, -mu).
    """
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
    """
    Obtiene la pose a partir de los vectores I y J de POSIT.

    Como I = (f / Z0) i y J = (f / Z0) j:
        - Z0 = f / ||I|| (se promedian ||I|| y ||J|| por el redondeo).
        - i = I / ||I||, j = J / ||J||, k = i x j: son las filas de R.
        - La posición de M0 en la cámara es T = (Z0/f · x0, Z0/f · y0, Z0),
          y de R M0 + t = T sale t = T - R M0.

    Parámetros:
        Is, Js: vectores I y J de largo 3.
        fx: distancia focal en píxeles.
        m0: imagen centrada (x0, y0) del punto de referencia.
        M0: punto de referencia en coordenadas del objeto.

    Devuelve:
        (i, j, k, Z0, R, t).
    """
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
    """
    Evalúa una pose candidata: cuenta cuántos puntos quedan delante de la cámara
    y calcula el error de reproyección.

    Cada punto se pasa a coordenadas de cámara (X = R M + t), se proyecta con K
    y se compara con el punto observado en la imagen.

    Parámetros:
        R, t: pose a evaluar.
        obj_vecs: array de n x 3 con los puntos del objeto.
        img_vecs: array de n x 2 con los puntos observados (u, v) en píxeles.
        K: matriz de intrínsecos de 3 x 3.

    Devuelve:
        (validos, rms): cantidad de puntos con Z > 0 y error RMS en píxeles.
    """
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
def resolverPOS(A, bu, bv, tol=1e-10):
    """
    Paso POS de POSIT: resuelve A I0 = bu y A J0 = bv por cuadrados mínimos.

    Usa la pseudoinversa calculada con la SVD reducida: A+ = V Sigma^-1 U^T.
    Se descartan los valores singulares casi nulos. En el caso coplanar A tiene
    rango 2, y así se obtiene la solución de norma mínima, sin componente en
    la dirección normal al plano (esa componente se agrega con calculoLambdaMu).

    Parámetros:
        A: matriz de (n-1) x 3 con filas Mi - M0.
        bu, bv: lados derechos de largo n-1.
        tol: tolerancia relativa para descartar valores singulares.

    Devuelve:
        (I0, J0): vectores de largo 3.
    """
    A = np.array(A, dtype=np.float64)
    bu = np.array(bu, dtype=np.float64)
    bv = np.array(bv, dtype=np.float64)

    # 1) SVD reducida: A = U Sigma V^T
    U, sigma, V = svd_reducida(A)

    # 2) Nos quedamos solo con los valores singulares no nulos (rango de A)
    r = 0
    for s in sigma:
        if s > tol * sigma[0]:
            r += 1
    U = U[:, 0:r]
    V = V[:, 0:r]
    sigma = sigma[0:r]

    # 3) A+ b = V Sigma^-1 U^T b
    #    U^T b: proyectamos b sobre las columnas de U
    Utbu = multiplicar(U.T, bu).reshape(-1)
    Utbv = multiplicar(U.T, bv).reshape(-1)

    #    Sigma^-1 es diagonal: dividimos cada componente por su valor singular,
    #    y después multiplicamos por V
    I0 = multiplicar(V, Utbu / sigma).reshape(-1)
    J0 = multiplicar(V, Utbv / sigma).reshape(-1)

    return I0, J0

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt=100, atol=1e-6):
    """
    Estima la pose de la cámara con POSIT para puntos coplanares (sobre W = 0).

    POSIT empieza suponiendo una proyección ortográfica escalada (eps_i = 0)
    y la va corrigiendo hacia la perspectiva real:
        1) Arma los lados derechos con los eps actuales.
        2) Resuelve el sistema POS (resolverPOS).
        3) Obtiene las dos soluciones del caso coplanar (calculoLambdaMu).
        4) Calcula la pose de cada una (calcularSalidaPOSIT), descarta las que
           dejan puntos detrás de la cámara y se queda con la de menor error
           de reproyección.
        5) Actualiza eps_i = (Mi - M0) · k / Z0.
    Se repite hasta que el error RMS deja de cambiar o se llega a MaxIt.

    Las coordenadas v se reescalan por fx / fy para trabajar con una sola focal.

    Parámetros:
        obj_vecs: array de n x 3 con los puntos del objeto. El primero es el
                  punto de referencia M0.
        img_vecs: array de n x 2 con los puntos (u, v) en píxeles, en el mismo orden.
        K: matriz de intrínsecos de 3 x 3.
        MaxIt: cantidad máxima de iteraciones.
        atol: tolerancia sobre el cambio del error RMS.

    Devuelve:
        (R, t): pose tal que X_cámara = R X_objeto + t.
                Si ninguna solución es válida, (None, None).
    """
    obj_vecs = np.array(obj_vecs, dtype=np.float64)
    img_vecs = np.array(img_vecs, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    n = np.shape(obj_vecs)[0]

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

    # Al principio suponemos proyección ortográfica escalada: eps_i = 0
    eps = np.zeros(n - 1)

    R_mejor = None
    t_mejor = None
    rms_anterior = None

    # Iteraciones
    for it in range(MaxIt):
        # 1) Lados derechos con los eps actuales
        bu = xs * (1 + eps) - m0[0]
        bv = ys * (1 + eps) - m0[1]

        # 2) Paso POS
        I0, J0 = resolverPOS(A, bu, bv)

        # 3) Las dos soluciones del caso coplanar: +(lam, mu) y -(lam, mu)
        lam, mu = calculoLambdaMu(I0, J0)
        candidatos = [(I0 + lam * normal, J0 + mu * normal),
                      (I0 - lam * normal, J0 - mu * normal)]

        # 4) Evaluamos cada candidato y nos quedamos con el mejor válido
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

        # Si ninguna solución es válida, cortamos con la última pose buena
        if rms_it is None:
            break

        R_mejor = R_it
        t_mejor = t_it

        # 5) Nuevos factores de corrección de perspectiva: eps_i = (a_i . k) / Z0
        for p in range(n - 1):
            eps[p] = np.sum(A[p, :] * k_it) / Z0_it

        # 6) Convergencia: el RMS ya no mejora
        if rms_anterior is not None and np.abs(rms_anterior - rms_it) < atol:
            break
        rms_anterior = rms_it

    return R_mejor, t_mejor