import numpy as np
import alc

# =========================================================
# TESTS LABO 1
# =========================================================

def sonIguales(x, y, atol=1e-08):
    return np.allclose(alc.error(x, y), 0, atol=atol)

# Tests de la clase
assert(not sonIguales(1, 1.1))
assert(sonIguales(1, 1 + np.finfo('float64').eps))
assert(not sonIguales(1, 1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3))

assert(np.allclose(alc.error_relativo(1, 1.1), 0.1))
assert(np.allclose(alc.error_relativo(2, 1), 0.5))
assert(np.allclose(alc.error_relativo(-1, -1), 0))
assert(np.allclose(alc.error_relativo(1, -1), 2))

assert(alc.matricesIguales(np.diag([1, 1]), np.eye(2)))
assert(alc.matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2)))
assert(not alc.matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]])))
print('Los tests de la clase pasaron correctamente')

# Tests particulares
assert(not sonIguales(1, 1.1))
assert(sonIguales(1, 1 + np.finfo('float64').eps))
assert(not sonIguales(1, 1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3))

assert(np.allclose(alc.error_relativo(1, 1.1), 0.1))
assert(np.allclose(alc.error_relativo(2, 1), 0.5))
assert(np.allclose(alc.error_relativo(-1, -1), 0))
assert(np.allclose(alc.error_relativo(1, -1), 2))

assert(alc.matricesIguales(np.diag([1, 1]), np.eye(2)))
assert(alc.matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2)))
assert(not alc.matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]])))

# Test: distintas dimensiones
assert(not alc.matricesIguales(np.eye(2), np.eye(3)))

print("Todos los tests particulares pasaron correctamente.")


# =========================================================
# TESTS LABO 2
# =========================================================

# Tests de la clase
assert(np.allclose(alc.rota(0), np.eye(2)))
assert(np.allclose(alc.rota(np.pi/2), np.array([[0,-1],[1,0]])))
assert(np.allclose(alc.rota(np.pi), np.array([[-1,0],[0,-1]])))

assert(np.allclose(alc.escala([2,3]), np.array([[2,0],[0,3]])))
assert(np.allclose(alc.escala([1,1,1]), np.eye(3)))
assert(np.allclose(alc.escala([0.5,0.25]), np.array([[0.5,0],[0,0.25]])))

assert(np.allclose(alc.rota_y_escala(0,[2,3]), np.array([[2,0],[0,3]])))
assert(np.allclose(alc.rota_y_escala(np.pi/2,[1,1]), np.array([[0,-1],[1,0]])))
assert(np.allclose(alc.rota_y_escala(np.pi,[2,2]), np.array([[-2,0],[0,-2]])))

assert(np.allclose(alc.afin(0,[1,1],[1,2]),np.array([[1,0,1],[0,1,2],[0,0,1]])))
assert(np.allclose(alc.afin(np.pi/2,[1,1],[0,0]),np.array([[0,-1,0],[1, 0,0],[0, 0,1]])))
assert(np.allclose(alc.afin(0,[2,3],[1,1]),np.array([[2,0,1],[0,3,1],[0,0,1]])))

assert(np.allclose(alc.trans_afin(np.array([1,0]), np.pi/2,[1,1],[0,0]),np.array([0,1])))
assert(np.allclose(alc.trans_afin(np.array([1,1]), 0,[2,3],[0,0]),np.array([2,3])))
assert(np.allclose(alc.trans_afin(np.array([1,0]), np.pi/2,[3,2],[4,5]),np.array([4,7])))

print(' Los tests de la clase pasan correctamente')

# Tests particulares

# ROTA: casos borde
t = np.pi/3
assert(np.allclose(alc.rota(-t), alc.rota(t).T))
assert(np.allclose(alc.rota(2*np.pi), np.eye(2)))
v = alc.multiplicar(alc.rota(3*np.pi/2), np.array([1, 0]))
assert(np.allclose(np.reshape(v, (2,)), np.array([0, -1])))
R = alc.rota(np.pi/5)
assert(np.allclose(alc.multiplicar(R, R.T), np.eye(2)))

# ESCALA: casos borde
assert(np.allclose(alc.escala([5]), np.array([[5]])))
assert(np.allclose(alc.escala([0, 1]), np.array([[0, 0], [0, 1]])))
assert(np.allclose(alc.escala([-1, 1]), np.array([[-1, 0], [0, 1]])))
assert(np.allclose(alc.escala([1, 2, 3, 4]),np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])))
s = np.array([2., 4., 5.])
assert(np.allclose(alc.multiplicar(alc.escala(s), alc.escala(1/s)), np.eye(3)))

# ROTAYESCALA: casos borde
assert(np.allclose(alc.rota_y_escala(np.pi/4, [1, 1]), alc.rota(np.pi/4)))
assert(np.allclose(alc.rota_y_escala(0, [3, 7]), alc.escala([3, 7])))
theta, s = np.pi/3, [2, 5]
RS = alc.multiplicar(alc.rota(theta), alc.escala(s))
SR = alc.rota_y_escala(theta, s)
assert(not np.allclose(RS, SR))

# AFIN: casos borde
assert(np.allclose(alc.afin(0, [1, 1], [0, 0]), np.eye(3)))
assert(np.allclose(alc.afin(0, [1, 1], [8, -3]),np.array([[1, 0, 8],[0, 1, -3],[0, 0, 1]])))
M = alc.afin(np.pi/7, [2.5, -1.2], [10, 20])
assert(np.allclose(M[2, :], np.array([0, 0, 1])))

# TRANSAFIN: casos borde
assert(np.allclose(alc.trans_afin(np.array([0, 0]), np.pi/3, [4, 9], [7, -2]),
np.array([7, -2])))
v = np.array([3.5, -2.1])
assert(np.allclose(alc.trans_afin(v, 0, [1, 1], [0, 0]), v))
theta, s, b = np.pi/6, [2, 3], [1, -1]
v = np.array([2.0, 5.0])
esperado = alc.multiplicar(alc.afin(theta, s, b), np.array([v[0], v[1], 1]))
esperado = np.reshape(esperado, (3,))[0:2]
assert(np.allclose(alc.trans_afin(v, theta, s, b), esperado))

print('Tests particulares pasan correctamente')


# =========================================================
# TESTS LABO 3
# =========================================================

# Tests alc.norma
assert(np.allclose(alc.norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(alc.norma(np.array([1]*10),2),np.sqrt(10)))
assert(alc.norma(np.random.rand(10),2)<=np.sqrt(10))
assert(alc.norma(np.random.rand(10),2)>=0)

# Tests alc.normaliza
for x in alc.normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(alc.norma(x,2),1))
for x in alc.normaliza([np.array([1]*k) for k in range(2,11)],1):
    print(not np.allclose(alc.norma(x,2),1))
for x in alc.normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert(np.allclose(alc.norma(x,'inf'),1))

# Tests alc.normaExacta
assert(np.allclose(alc.normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(alc.normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(alc.normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(alc.normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(alc.normaExacta(np.random.random((10,10)),1)<=10)
assert(alc.normaExacta(np.random.random((4,4)),'inf')<=4)

# Test normaMC
nMC = alc.normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = alc.normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = alc.normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],alc.normaExacta(A,'inf'),rtol=2e-1))

# Test alc.condMC
A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = alc.normaMatMC(A,2,2,10000)
normaA_ = alc.normaMatMC(A_,2,2,10000)
condA = alc.condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = alc.normaMatMC(A,2,2,10000)
normaA_ = alc.normaMatMC(A_,2,2,10000)
condA = alc.condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

# Test alc.condExacta
A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = alc.normaExacta(A,1)
normaA_ = alc.normaExacta(A_,1)
condA = alc.condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = alc.normaExacta(A,'inf')
normaA_ = alc.normaExacta(A_,'inf')
condA = alc.condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))


# =========================================================
# TESTS LABO 4
# =========================================================

print("TESTS calculaLU")
L0 = np.array([[1,0,0],[0,1,0],[1,1,1]])
U0 = np.array([[10,1,0],[0,2,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = alc.calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))
L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = alc.calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 13)
L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,0,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = alc.calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0)
assert(alc.calculaLU(None) == (None, None, 0))
assert(alc.calculaLU(np.array([[1,2,3],[4,5,6]])) == (None, None, 0))
print("-----ÉXITO!!!!\n")

print("TESTS res_tri")
A = np.array([[1,0,0],[1,1,0],[1,1,1]])
b = np.array([1,1,1])
assert(np.allclose(alc.res_tri(A,b),np.array([1,0,0])))
b = np.array([0,1,0])
assert(np.allclose(alc.res_tri(A,b),np.array([0,1,-1])))
b = np.array([-1,1,-1])
assert(np.allclose(alc.res_tri(A,b),np.array([-1,2,-2])))
b = np.array([-1,1,-1])
assert(np.allclose(alc.res_tri(A,b,inferior=False),np.array([-1,1,-1])))
A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(alc.res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))
A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(alc.res_tri(A,b,inferior=False),np.array([1,1,1])))
print("-----ÉXITO!!!!\n")

print("TESTS inversa")
def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except:
        return True

ntest = 10
for i in range(ntest):
    A = np.random.random((4,4))
    A_ = alc.inversa(A)
    if not esSingular(A):
        inversaConNumpy = np.linalg.inv(A)
        assert(A_ is not None)
        assert(np.allclose(inversaConNumpy,A_))
    else:
        assert(A_ is None)

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(alc.inversa(A) is None)
print("-----ÉXITO!!!!\n")

print("TESTS calculaLDV")
L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0 @ V0
L,D,V = alc.calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))
L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V = alc.calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))
print("-----ÉXITO!!!!\n")

print("TESTS esSDP")
L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(alc.esSDP(A))
D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not alc.esSDP(A))
D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not alc.esSDP(A))
L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],[1,1,0],[1,1+1e-3,1]]).T
A = L0 @ D0 @ V0
assert(alc.esSDP(A,1e-3))
print("-----ÉXITO!!!!\n")
print("---FINALIZADO LABO 4!---")


# =========================================================
# TESTS LABO 5
# =========================================================

A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

def check_QR(Q,R,A,tol=1e-10):
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

Q2,R2 = alc.QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = alc.QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = alc.QR_con_GS(A4)
check_QR(Q4,R4,A4)

Q2h,R2h = alc.QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = alc.QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = alc.QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

Q2c,R2c = alc.calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = alc.calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = alc.calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)

print("TESTS LABO 5 OK")


# =========================================================
# TESTS LABO 6
# =========================================================

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = alc.metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95

exitos = 0
for i in range(100):
    v = np.random.rand(9)
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)

    A = H@D@H.T
    v,l,_ = alc.metpot2k(A, 1e-15, 1e5)
    if abs(l - D[0,0]) < 1e-8:
        exitos +=1
assert exitos > 95

D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = alc.diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)

exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = alc.diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = alc.normaExacta(ARH-A,p='inf')
    if e < 1e-5:
        exitos += 1
assert exitos >= 95

print("TESTS LABO 6 OK")


# =========================================================
# TESTS LABO 7
# =========================================================

def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

for i in range(1,100):
    T = alc.transiciones_al_azar_continuas(i)
    assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"

    T = alc.transiciones_al_azar_uniformes(i,0.3)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = alc.transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = alc.transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"

# nucleo
A = np.eye(3)
S = alc.nucleo(A)
assert S.shape[0]==0, "nucleo fallo para matriz identidad"
A[1,1] = 0
S = alc.nucleo(A)
msg = "nucleo fallo para matriz con un cero en diagonal"
assert esNucleo(A,S), msg
assert S.shape==(3,1), msg
assert abs(S[2,0])<1e-2, msg
assert abs(S[0,0])<1e-2, msg

v = np.random.random(5)
v = v / np.linalg.norm(v)
H = np.eye(5) - np.outer(v, v)
S = alc.nucleo(H)
msg = "nucleo fallo para matriz de proyeccion ortogonal"
assert S.shape==(5,1), msg
v_gen = S[:,0]
v_gen = v_gen / np.linalg.norm(v_gen)
assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

# crea rala
listado = [[0,17],[3,4],[0.5,0.25]]
A_rala_dict, dims = alc.crea_rala(listado,32,89)
assert dims == (32,89), "crea_rala fallo en dimensiones"
assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
A_rala_dict, dims = alc.crea_rala(listado,50,50)
assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
assert A_rala_dict.get((32,3)) == 7
assert A_rala_dict[(16,4)] == 0.5
assert A_rala_dict[(5,7)] == 0.25

listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
A_rala_dict, dims = alc.crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
assert (1,4) not in A_rala_dict
assert A_rala_dict[(2,5)] == 0.5
assert A_rala_dict[(3,6)] == 0.25
assert len(A_rala_dict) == 2

listado = []
A_rala_dict, dims = alc.crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

listado = [[0,1,2],[0,1,2],[1,2,3]]
A_rala = alc.crea_rala(listado,3,3)
v = np.random.random(3)
v = v / np.linalg.norm(v)
res = alc.multiplica_rala_vector(A_rala,v)
A = np.array([[1,0,0],[0,2,0],[0,0,3]])
res_esperado = A @ v
assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

A = np.random.random((5,5))
A = A * (A > 0.5)
listado = [[],[],[]]
for i in range(5):
    for j in range(5):
        listado[0].append(i)
        listado[1].append(j)
        listado[2].append(A[i,j])

A_rala = alc.crea_rala(listado,5,5)
v = np.random.random(5)
assert np.allclose(alc.multiplica_rala_vector(A_rala,v), A @ v)

print("TESTS LABO 7 OK")
print()
print("TODOS LOS TESTS PASARON")
