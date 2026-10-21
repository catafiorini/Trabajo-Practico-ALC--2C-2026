# Calibración y Pose de la Cámara

Trabajo Práctico de **Álgebra Lineal Computacional**
Licenciatura en Ciencia de Datos — FCEN, UBA (2do cuatrimestre 2026)

## ¿De qué se trata?

A partir de fotos 2D, recuperar información del mundo real en 3D: distancias, posiciones y velocidades. Para eso se usa el modelo de cámara *pinhole*, la calibración de la cámara y la estimación de su posición (pose).

Todas las operaciones de álgebra lineal (productos de matrices, factorizaciones QR, SVD y Cholesky, inversas, normas) están implementadas desde cero, sin usar las funciones ya hechas de NumPy ni OpenCV.

## Aplicaciones

- **Fútbol:** medir a qué distancia del arco estaba un jugador y analizar un posible fuera de juego.
- **Tránsito:** calcular la velocidad real de autos y peatones en una esquina de Buenos Aires.
- **Drones:** estimar la posición de un drone respecto de una marca "H" para aterrizar.

## Contenido

- `alc.py`: biblioteca con todas las funciones.
- `grupo_XX_TP.ipynb`: notebook con los resultados y conclusiones.
- Carpetas `calib`, `futbol`, `salguero` y `heli`: datos de cada ejercicio.

## Integrantes

- Chicolino de Luca Juan Pablo
- D'Angelo Genaro
- Fiorini Catalina
- Flores Garcia Emanuel
