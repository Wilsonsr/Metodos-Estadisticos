###############################################################################
#  ESTADÍSTICA Y PROBABILIDAD · SESIÓN 2
#  Variables aleatorias continuas y vectores aleatorios
#
#  Script de la clase. Cada sección lleva el número de la diapositiva
#  («Diap. N») de Clase_VA_Distribuciones.pptx.
#
#  Cómo usarlo:
#   - Ejecute línea por línea (Ctrl + Enter en RStudio) o por secciones.
#   - Solo usa R base (no requiere instalar paquetes).
#   - Los resultados esperados están en los comentarios «# ->».
#   - Con set.seed(2026) todos obtienen los mismos números simulados
#     (R 3.6 o posterior).
###############################################################################

rm(list = ls())
op <- par(no.readonly = TRUE)        # para restaurar la configuración gráfica

# Colores de la presentación
azul    <- "#1B2A41"
naranja <- "#E07A1F"
verde   <- "#2A9D8F"
celeste <- "#CADCFC"
morado  <- "#8A4FBF"

# -----------------------------------------------------------------------------
# FUNCIÓN AUXILIAR: sombrear el área bajo una densidad entre a y b
#   dens : función de densidad, por ejemplo function(x) dnorm(x, 300, 10)
#   a, b : límites del área que se quiere sombrear
#   desde, hasta : rango del eje x que se dibuja
# Devuelve (invisible) el área calculada con integrate().
# -----------------------------------------------------------------------------
sombrear <- function(dens, a, b, desde, hasta, col = naranja,
                     main = "", xlab = "x", ylab = "f(x)") {
  curve(dens(x), from = desde, to = hasta, lwd = 3, col = azul,
        main = main, xlab = xlab, ylab = ylab, n = 500)
  xs <- seq(max(a, desde), min(b, hasta), length.out = 300)
  polygon(c(xs[1], xs, xs[length(xs)]), c(0, dens(xs), 0),
          col = adjustcolor(col, 0.7), border = NA)
  area <- integrate(dens, max(a, desde), min(b, hasta))$value
  invisible(area)
}

###############################################################################
#  BLOQUE 1 · VARIABLES ALEATORIAS CONTINUAS
###############################################################################

# ---- Diap. 4: De contar a medir ---------------------------------------------
# Discreta: la probabilidad es la ALTURA de una barra.
# Continua: la probabilidad es un ÁREA bajo la curva.
par(mfrow = c(1, 2))
barplot(dpois(0:8, 3), names.arg = 0:8, col = azul,
        main = "CONTAR: P(X = x)", xlab = "x")
sombrear(function(x) dgamma(x, 4, 1), 3, 5, 0, 10,
         main = "MEDIR: P(a < X < b) = área", xlab = "x (min)")
par(mfrow = c(1, 1))

# ---- Diap. 5: Definición de variable aleatoria continua ---------------------
# P(X = a) = 0: un intervalo de longitud cero no tiene área.
integrate(function(x) dunif(x, 0, 2), 1, 1)$value      # -> 0
punif(1.5, 0, 2) - punif(0.5, 0, 2)                    # -> 0.5
# En continuas, P(X < 1.5) y P(X <= 1.5) son iguales:
punif(1.5, 0, 2)                                       # -> 0.75

# ---- Diap. 6: Función de densidad -------------------------------------------
# Condiciones: f(x) >= 0  y  área total = 1.
# Una densidad PUEDE valer más de 1: U(0, 0.5) tiene altura 2.
dunif(0.2, 0, 0.5)                                     # -> 2  (altura, no probabilidad)
integrate(function(x) dunif(x, 0, 0.5), 0, 0.5)$value  # -> 1  (área total)
curve(dunif(x, 0, 0.5), -0.2, 0.8, lwd = 3, col = azul, n = 1000,
      main = "U(0, 0.5): altura 2, área 1", ylab = "f(x)")

# ---- Diap. 7: Ejemplo resuelto f(x) = 3x^2 en [0, 1] ------------------------
f <- function(x) ifelse(x >= 0 & x <= 1, 3 * x^2, 0)

# 1) No negativa: 3x^2 >= 0.   2) Área total:
integrate(f, 0, 1)$value                               # -> 1  (sí es densidad)
# 3) Una probabilidad: P(0.5 < X < 1)
integrate(f, 0.5, 1)$value                             # -> 0.875
integrate(f, 0, 0.5)$value                             # -> 0.125
# 4) Valor esperado: E(X) = ∫ x f(x) dx
EX  <- integrate(function(x) x * f(x), 0, 1)$value
EX                                                     # -> 0.75
# 5) Varianza: Var(X) = E(X^2) - [E(X)]^2
EX2 <- integrate(function(x) x^2 * f(x), 0, 1)$value
EX2                                                    # -> 0.6
EX2 - EX^2                                             # -> 0.0375
sqrt(EX2 - EX^2)                                       # -> 0.1936 (SD)

sombrear(f, 0.5, 1, 0, 1, main = "f(x) = 3x^2: P(0.5 < X < 1) = 0.875")
abline(v = EX, lty = 2, lwd = 2, col = verde)

# ---- Diap. 8: Función de distribución acumulada F(x) ------------------------
# Ejemplo de la guía: f(x) = 1/2 en [0, 2]  ->  F(x) = x/2
x <- seq(-0.5, 2.5, by = 0.01)
par(mfrow = c(2, 1))
plot(x, dunif(x, 0, 2), type = "l", lwd = 3, col = azul,
     main = "Densidad f(x) = 1/2", ylab = "f(x)")
plot(x, punif(x, 0, 2), type = "l", lwd = 3, col = azul,
     main = "Acumulada F(x) = x/2", ylab = "F(x)")
abline(h = c(0.25, 0.75), v = c(0.5, 1.5), lty = 3, col = naranja)
par(mfrow = c(1, 1))

punif(1.5, 0, 2)                                       # -> 0.75 = F(1.5)
punif(0.5, 0, 2)                                       # -> 0.25 = F(0.5)
punif(1.5, 0, 2) - punif(0.5, 0, 2)                    # -> 0.5  = F(b) - F(a)

# ---- Diap. 9: Propiedades de F(x) y cuantiles -------------------------------
# Para f(x) = 3x^2:  F(x) = x^3   ->   cuantil x_p = p^(1/3)
Fcub <- function(x) x^3
Fcub(0.5)                                              # -> 0.125
1 - Fcub(0.5)                                          # -> 0.875 = P(X > 0.5)
c(Q1 = 0.25^(1/3), mediana = 0.5^(1/3), Q3 = 0.75^(1/3))
# -> 0.630 0.794 0.909
# Verificación numérica del cuantil: resolver F(x) = 0.5
uniroot(function(x) Fcub(x) - 0.5, c(0, 1))$root       # -> 0.794
# La letra q de R resuelve F(x) = p:
qunif(0.5, 0, 2)                                       # -> 1
qexp(0.5, rate = 4)                                    # -> 0.1733 h
qexp(0.5, rate = 4) * 60                               # -> 10.4 min
# f = F': derivada numérica de F en x = 0.6 (debe ser 3 * 0.6^2 = 1.08)
h <- 1e-6; (Fcub(0.6 + h) - Fcub(0.6 - h)) / (2 * h)   # -> 1.08

# ---- Diap. 10: Valor esperado y varianza de una continua --------------------
# Ejemplo de la guía: f(x) = 1/2 en [0, 2]
fu  <- function(x) dunif(x, 0, 2)
EX  <- integrate(function(x) x * fu(x), 0, 2)$value    # -> 1
EX2 <- integrate(function(x) x^2 * fu(x), 0, 2)$value  # -> 4/3
c(EX = EX, EX2 = EX2, Var = EX2 - EX^2, SD = sqrt(EX2 - EX^2))
# -> 1  1.333  0.333  0.577

# ---- Diap. 11: Propiedades de E y Var ---------------------------------------
# C = 0.02 X + 5, con E(X) = 300 y SD(X) = 10
0.02 * 300 + 5                                         # -> 11   = E(C)
0.02^2 * 10^2                                          # -> 0.04 = Var(C)
0.02 * 10                                              # -> 0.2  = SD(C)
set.seed(2026)
xs <- rnorm(100000, 300, 10)
C  <- 0.02 * xs + 5
c(media = mean(C), var = var(C), sd = sd(C))           # -> ≈ 11  0.04  0.2
# Cambio de unidades: minutos -> segundos (Y = 60 X)
c(media = mean(60 * xs) / mean(xs), var = var(60 * xs) / var(xs))  # -> 60 3600

# ---- Diap. 12: Regla d · p · q · r con la Normal(300, 10) -------------------
dnorm(310, mean = 300, sd = 10)    # -> 0.0242  ALTURA (densidad), NO es P(X = 310)
pnorm(310, mean = 300, sd = 10)    # -> 0.8413  P(X <= 310)
qnorm(0.95, mean = 300, sd = 10)   # -> 316.45  valor con área 0.95 a la izquierda
set.seed(2026)
r <- rnorm(1000, mean = 300, sd = 10)   # 1000 valores simulados
head(round(r, 1))

par(mfrow = c(2, 2))
curve(dnorm(x, 300, 10), 265, 335, lwd = 3, col = azul, main = "d: altura en 310")
segments(310, 0, 310, dnorm(310, 300, 10), lwd = 4, col = azul)
sombrear(function(x) dnorm(x, 300, 10), -Inf, 310, 265, 335, col = verde,
         main = "p: área hasta 310 = 0.841")
sombrear(function(x) dnorm(x, 300, 10), -Inf, qnorm(0.95, 300, 10), 265, 335,
         main = "q: valor con área 0.95 = 316.4")
hist(r, probability = TRUE, col = morado, border = "white", main = "r: simulación")
par(mfrow = c(1, 1))

# ---- Diap. 13: Uniforme U(0, 10) --------------------------------------------
a <- 0; b <- 10
curve(dunif(x, a, b), -1, 11, lwd = 3, col = azul, n = 1000,
      main = "X ~ U(0, 10)", ylab = "f(x)")
(a + b) / 2                                            # -> 5     = E(X)
(b - a)^2 / 12                                         # -> 8.33  = Var(X)

# ---- Diap. 14: Uniforme: deducción de E(X) y Var(X) -------------------------
g1 <- function(x) x   * dunif(x, 0, 10)
g2 <- function(x) x^2 * dunif(x, 0, 10)
integrate(g1, 0, 10)$value                             # -> 5      = (a + b)/2
integrate(g2, 0, 10)$value                             # -> 33.33  = (a^2 + ab + b^2)/3
integrate(g2, 0, 10)$value - 5^2                       # -> 8.33   = (b - a)^2/12
# Propiedad de longitud: P(c < X < d) = (d - c)/(b - a)
(15 - 0) / 2                                           # bus cada 15 min: espera media 7.5
1 - punif(10, 0, 15)                                   # -> 0.333 = P(esperar > 10 min)

# ---- Diap. 15: Uniforme: P(2 < X < 5) ---------------------------------------
punif(5, min = 0, max = 10) - punif(2, min = 0, max = 10)   # -> 0.3
(5 - 2) * (1 / 10)                                          # -> 0.3 (base x altura)
sombrear(function(x) dunif(x, 0, 10), 2, 5, -1, 11,
         main = "P(2 < X < 5) = 0.3", xlab = "tiempo de espera (min)")

# ---- Diap. 16: Uniforme: simulación -----------------------------------------
set.seed(2026)
x <- runif(10000, 0, 10)
mean(x)                                                # -> 4.989  (teórico 5)
var(x)                                                 # -> 8.312  (teórico 8.333)
hist(x, probability = TRUE, col = celeste, border = "white",
     main = "10 000 esperas simuladas", ylim = c(0, 0.13))
abline(h = 1/10, lwd = 3, col = naranja)
abline(v = mean(x), lwd = 2, lty = 2, col = azul)

# ---- Diap. 17: Normal -------------------------------------------------------
curve(dnorm(x, mean = 300, sd = 10), 260, 340, lwd = 3, col = azul,
      main = "N(300, 10^2)", ylab = "f(x)")
abline(v = 300, col = naranja, lwd = 2, lty = 2)
# Cuidado: en R se escribe sd = sigma (no sigma^2).

# ---- Diap. 18: Normal: propiedades ------------------------------------------
# Regla empírica (valores exactos):
pnorm(1) - pnorm(-1)                                   # -> 0.6827
pnorm(2) - pnorm(-2)                                   # -> 0.9545
pnorm(3) - pnorm(-3)                                   # -> 0.9973
# Propiedad aX + b: si X ~ N(300, 10^2), Y = X/1000 ~ N(0.3, 0.01^2)
set.seed(2026)
y <- rnorm(100000, 300, 10) / 1000
c(media = mean(y), sd = sd(y))                         # -> ≈ 0.3  0.01

# ---- Diap. 19: Efecto de mu y sigma -----------------------------------------
curve(dnorm(x, 0, 1), -7, 7, lwd = 3, col = azul, ylim = c(0, 0.45),
      main = "mu desplaza, sigma abre", ylab = "f(x)")
curve(dnorm(x, 2, 1), add = TRUE, lwd = 3, col = naranja)   # cambia mu
curve(dnorm(x, 0, 2), add = TRUE, lwd = 3, col = verde)     # cambia sigma
legend("topleft", c("N(0, 1)", "N(2, 1)", "N(0, 2^2)"),
       col = c(azul, naranja, verde), lwd = 3, bty = "n")

# ---- Diap. 20–22: Caso volumen de llenado X ~ N(300, 10^2) -----------------
mu <- 300; sigma <- 10
pa <- 1 - pnorm(310, mean = mu, sd = sigma)                     # a) P(X > 310)
pb <- pnorm(305, mean = mu, sd = sigma) - pnorm(290, mean = mu, sd = sigma)  # b)
pc <- 1 - pnorm(320, mean = mu, sd = sigma)                     # c) P(X > 320)
round(c(a = pa, b = pb, c = pc), 4)                    # -> 0.1587 0.5328 0.0228
# Alternativa para colas derechas:
pnorm(310, mu, sigma, lower.tail = FALSE)              # -> 0.1587
# Interpretación (Diap. 22):
cat(sprintf("a) %.1f %% de los envases tendría más de 310 ml\n", 100 * pa))
cat(sprintf("b) %.1f %% tendría entre 290 y 305 ml\n", 100 * pb))
cat(sprintf("c) %.1f %% superaría 320 ml (unos %.0f de cada 1000)\n", 100 * pc, 1000 * pc))

par(mfrow = c(1, 3))
dn <- function(x) dnorm(x, mu, sigma)
sombrear(dn, 310, Inf, 265, 335, main = "P(X > 310) = 0.1587")
sombrear(dn, 290, 305, 265, 335, col = verde, main = "P(290 < X < 305) = 0.5328")
sombrear(dn, 320, Inf, 265, 335, main = "P(X > 320) = 0.0228")
par(mfrow = c(1, 1))

# ---- Diap. 23: Estandarización ---------------------------------------------
z <- (320 - mu) / sigma
z                                                      # -> 2
pnorm(320, mu, sigma)                                  # -> 0.9772
pnorm(z)                                               # -> 0.9772 (misma probabilidad)
(285 - mu) / sigma                                     # -> -1.5

# ---- Diap. 24: Problema inverso (cuantiles) ---------------------------------
qnorm(0.95)                                            # -> 1.645 (z)
mu + qnorm(0.95) * sigma                               # -> 316.4 (x = mu + z sigma)
qnorm(0.95, mu, sigma)                                 # -> 316.4 (directo)
qnorm(c(0.025, 0.975), mu, sigma)                      # -> 280.4 319.6 (95 % central)
qnorm(0.10, mu, sigma)                                 # -> 287.2

# ---- Diap. 25: Laboratorio: simular la máquina de llenado -------------------
set.seed(2026)
x <- rnorm(10000, mean = 300, sd = 10)
mean(x)                                                # -> 300.04
sd(x)                                                  # -> 10.01
hist(x, probability = TRUE, col = celeste, border = "white", breaks = 40,
     main = "10 000 envases simulados", xlab = "volumen (ml)")
curve(dnorm(x, 300, 10), add = TRUE, lwd = 3, col = naranja)
# Proporción simulada de envases > 310 ml vs teórica:
c(simulada = mean(x > 310), teorica = 1 - pnorm(310, 300, 10))

# ---- Diap. 26: Exponencial: tiempo entre eventos ----------------------------
# Poisson cuenta eventos; Exponencial mide el tiempo entre eventos.
lambda <- 4                                            # clientes por hora
1 / lambda                                             # -> 0.25 h = 15 min (E(X))
1 / lambda^2                                           # -> 0.0625 (Var(X))
curve(dexp(x, rate = lambda), 0, 1.5, lwd = 3, col = azul,
      main = "Exponencial(4 por hora)", xlab = "horas", ylab = "f(x)")
# Ilustración: llegadas Poisson y tiempos entre llegadas
set.seed(2026)
llegadas <- cumsum(rexp(30, rate = lambda))
llegadas <- llegadas[llegadas < 3]
table(cut(llegadas, breaks = 0:3))                     # conteos por hora (Poisson)
round(diff(llegadas) * 60, 1)                          # tiempos entre llegadas (min)

# ---- Diap. 27: Exponencial: deducción y falta de memoria --------------------
# F(x) = 1 - exp(-lambda x);  P(X > x) = exp(-lambda x)
integrate(function(t) lambda * exp(-lambda * t), 0, 0.5)$value  # -> 0.8647
1 - exp(-lambda * 0.5)                                          # -> 0.8647
pexp(0.5, rate = lambda)                                        # -> 0.8647
# Falta de memoria: P(X > 0.75 | X > 0.5) = P(X > 0.25)
(1 - pexp(0.75, lambda)) / (1 - pexp(0.5, lambda))     # -> 0.3679
1 - pexp(0.25, lambda)                                 # -> 0.3679
# Conexión con Poisson: P(X > t) = P(0 llegadas en [0, t]), N ~ Poisson(lambda t)
dpois(0, lambda = lambda * 0.5)                        # -> 0.1353
# Mediana vs media
qexp(0.5, lambda) * 60                                 # -> 10.4 min (mediana)
(1 / lambda) * 60                                      # -> 15 min   (media)

# ---- Diap. 28: Exponencial: ¿esperar más de 30 minutos? ---------------------
1 - pexp(0.5, rate = 4)                                # -> 0.1353 (30 min = 0.5 h)
exp(-2)                                                # -> 0.1353
set.seed(2026)
x <- rexp(10000, rate = 4)
mean(x)                                                # -> 0.2575 h
mean(x) * 60                                           # -> 15.4 min
mean(x > 0.5)                                          # proporción simulada > 30 min
sombrear(function(x) dexp(x, 4), 0.5, Inf, 0, 1.5,
         main = "P(X > 0.5 h) = 0.135", xlab = "horas")

# ---- Diap. 29: Gamma --------------------------------------------------------
curve(dgamma(x, shape = 1, rate = 1), 0, 10, lwd = 3, col = azul,
      ylab = "f(x)", main = "Gamma(shape, rate = 1)")
curve(dgamma(x, shape = 2, rate = 1), 0, 10, add = TRUE, lwd = 3, col = naranja)
curve(dgamma(x, shape = 5, rate = 1), 0, 10, add = TRUE, lwd = 3, col = verde)
legend("topright", paste("shape =", c(1, 2, 5)),
       col = c(azul, naranja, verde), lwd = 3, bty = "n")
# Ejemplo de la guía: alpha = 2, lambda = 1
alpha <- 2; lam <- 1
c(E = alpha / lam, Var = alpha / lam^2)                # -> 2 2
# alpha = 1 es la Exponencial:
all.equal(dgamma(1.3, shape = 1, rate = 4), dexp(1.3, rate = 4))   # -> TRUE

# ---- Diap. 30: Gamma: definición y ejemplo resuelto -------------------------
gamma(4); factorial(3)                                 # -> 6 6  (Gamma(n) = (n-1)!)
# Fallas a razón de 2 por día; T = tiempo hasta la 3.ª falla ~ Gamma(3, 2)
c(E = 3 / 2, SD = sqrt(3 / 2^2))                       # -> 1.5 0.866
pgamma(1, shape = 3, rate = 2)                         # -> 0.3233
1 - ppois(2, lambda = 2)                               # -> 0.3233 (mismo evento)
# Gamma = suma de exponenciales independientes
set.seed(2026)
t3 <- rexp(1e4, 2) + rexp(1e4, 2) + rexp(1e4, 2)
c(media = mean(t3), var = var(t3), P_menor_1 = mean(t3 < 1))   # -> ≈ 1.5 0.75 0.32
hist(t3, probability = TRUE, breaks = 50, col = celeste, border = "white",
     main = "Suma de 3 exponenciales vs Gamma(3, 2)", xlab = "días")
curve(dgamma(x, 3, 2), add = TRUE, lwd = 3, col = naranja)

# ---- Diap. 31: Beta ---------------------------------------------------------
curve(dbeta(x, 2, 5), 0, 1, lwd = 3, col = azul, ylim = c(0, 3),
      ylab = "f(x)", main = "Beta(alpha, beta)")
curve(dbeta(x, 5, 2), 0, 1, add = TRUE, lwd = 3, col = naranja)
curve(dbeta(x, 5, 5), 0, 1, add = TRUE, lwd = 3, col = verde)
legend("top", c("Beta(2,5)", "Beta(5,2)", "Beta(5,5)"),
       col = c(azul, naranja, verde), lwd = 3, bty = "n")
media_beta <- function(a, b) a / (a + b)
var_beta   <- function(a, b) a * b / ((a + b)^2 * (a + b + 1))
rbind(E   = c(media_beta(2, 5), media_beta(5, 2), media_beta(5, 5)),
      Var = c(var_beta(2, 5),   var_beta(5, 2),   var_beta(5, 5)))
# -> E: 0.286 0.714 0.500 · Var: 0.0255 0.0255 0.0227

# ---- Diap. 32: Beta: definición y ejemplo resuelto --------------------------
1 / beta(8, 2)                                         # -> 72  -> f(x) = 72 x^7 (1 - x)
fb <- function(x) 72 * x^7 * (1 - x)
integrate(fb, 0, 1)$value                              # -> 1
c(E = media_beta(8, 2), Var = var_beta(8, 2), SD = sqrt(var_beta(8, 2)))
# -> 0.8 0.0145 0.121
1 - pbeta(0.9, 8, 2)                                   # -> 0.2252
integrate(fb, 0.9, 1)$value                            # -> 0.2252
1 - (9 * 0.9^8 - 8 * 0.9^9)                            # -> 0.2252 (a mano)
qbeta(0.5, 8, 2)                                       # -> 0.820 (mediana)
# Beta(1, 1) = U(0, 1)
all.equal(dbeta(0.37, 1, 1), dunif(0.37))              # -> TRUE

# ---- Diap. 33: t, Chi-cuadrado y F ------------------------------------------
par(mfrow = c(1, 3))
curve(dnorm(x), -4, 4, lwd = 2, lty = 2, col = "gray50", main = "t (df = 10)", ylab = "")
curve(dt(x, df = 10), add = TRUE, lwd = 3, col = azul)
curve(dchisq(x, df = 4), 0, 20, lwd = 3, col = naranja, main = "Chi-cuadrado (df = 4)", ylab = "")
curve(df(x, df1 = 5, df2 = 10), 0, 5, lwd = 3, col = verde, main = "F (5, 10)", ylab = "")
par(mfrow = c(1, 1))

# ---- Diap. 34: Construcción de t, Chi-cuadrado y F --------------------------
c(normal = qnorm(0.975), t10 = qt(0.975, 10), chi4 = qchisq(0.95, 4), F5_10 = qf(0.95, 5, 10))
# -> 1.960 2.228 9.488 3.326
set.seed(2026)
z <- matrix(rnorm(4 * 10000), ncol = 4)
v <- rowSums(z^2)                                      # suma de 4 normales al cuadrado
c(media = mean(v), var = var(v))                       # -> ≈ 4 ≈ 8  (chi-cuadrado(4))
# t con 10 g.l. construida: Z / sqrt(V/10)
set.seed(2026)
tt <- rnorm(10000) / sqrt(rchisq(10000, 10) / 10)
c(var_simulada = var(tt), var_teorica = 10 / 8)        # -> ≈ 1.25

# ---- Diap. 35: Resumen: ¿qué distribución continua uso? ---------------------
# Ejemplo de la pregunta: 2.ª reclamación, 3 por día -> Gamma(2, 3)
2 / 3                                                  # -> 0.667 días en promedio

###############################################################################
#  BLOQUE 2 · VECTORES ALEATORIOS
###############################################################################

# ---- Diap. 37: Dos variables a la vez (ilustración) -------------------------
set.seed(3)
estatura <- rnorm(120, 165, 9)
peso     <- 0.75 * estatura - 60 + rnorm(120, 0, 6)
plot(estatura, peso, pch = 19, col = adjustcolor(azul, 0.6),
     main = "Cada punto es un par (x, y) · datos simulados")

# ---- Diap. 38: Definición de vector aleatorio -------------------------------
# Lanzar una moneda 3 veces (1 = cara). X = caras en L1, L2; Y = caras en L2, L3
omega <- expand.grid(L1 = 0:1, L2 = 0:1, L3 = 0:1)
omega$X <- omega$L1 + omega$L2
omega$Y <- omega$L2 + omega$L3
omega                                                  # cada resultado -> un par (X, Y)

# ---- Diap. 39: Distribución conjunta ----------------------------------------
Pxy <- table(X = omega$X, Y = omega$Y) / 8
Pxy
# Misma tabla escrita a mano (guía):
Pxy <- matrix(c(1/8, 1/8, 0,
                1/8, 2/8, 1/8,
                0,   1/8, 1/8), nrow = 3, byrow = TRUE,
              dimnames = list(X = 0:2, Y = 0:2))
Pxy["1", "2"]                                          # -> 0.125 = P(X = 1, Y = 2)
sum(Pxy)                                               # -> 1
image(0:2, 0:2, Pxy, col = c("white", celeste, azul), xlab = "X", ylab = "Y",
      main = "Distribución conjunta P(X = x, Y = y)")
text(rep(0:2, 3), rep(0:2, each = 3), labels = c("1/8","1/8","0","1/8","2/8","1/8","0","1/8","1/8"))

# ---- Diap. 40: Conjunta continua (uniforme en [0, 2] x [0, 2]) --------------
fxy <- function(x, y) ifelse(x >= 0 & x <= 2 & y >= 0 & y <= 2, 1/4, 0)
# Integral doble con integrate anidado
doble <- function(f, ax, bx, ay, by) {
  integrate(function(y) sapply(y, function(yy)
    integrate(function(x) f(x, yy), ax, bx)$value), ay, by)$value
}
doble(fxy, 0, 2, 0, 2)                                 # -> 1    (es densidad)
doble(fxy, 0, 1, 0, 1)                                 # -> 0.25 P(0<=X<=1, 0<=Y<=1)
# Monte Carlo
set.seed(2026)
xu <- runif(1e5, 0, 2); yu <- runif(1e5, 0, 2)
mean(xu <= 1 & yu <= 1)                                # -> ≈ 0.25
mean(xu + yu <= 2)                                     # -> ≈ 0.5

# ---- Diap. 41: Distribuciones marginales ------------------------------------
rowSums(Pxy)                                           # -> 0.25 0.50 0.25 marginal de X
colSums(Pxy)                                           # -> 0.25 0.50 0.25 marginal de Y
addmargins(Pxy)                                        # tabla con los márgenes
all.equal(as.numeric(rowSums(Pxy)), dbinom(0:2, 2, 0.5))   # -> TRUE (Binomial(2, 0.5))

# ---- Diap. 42: Distribución condicional -------------------------------------
Pxy[, "2"] / sum(Pxy[, "2"])                           # -> 0 0.5 0.5 = P(X = x | Y = 2)
prop.table(Pxy, margin = 2)                            # todas las condicionales X | Y
barplot(rbind(rowSums(Pxy), Pxy[, "2"] / sum(Pxy[, "2"])), beside = TRUE,
        col = c(celeste, naranja), names.arg = 0:2,
        legend.text = c("P(X = x)", "P(X = x | Y = 2)"),
        main = "Saber Y = 2 cambia la distribución de X")

# ---- Diap. 43: Marginales y condicionales (caso continuo del RED) -----------
# f(x, y) = x + y en [0, 1]^2
fX  <- function(x) x + 0.5                             # marginal de X
fXy <- function(x) (x + 1) / 1.5                       # condicional de X dado Y = 1
integrate(fX, 0.5, 1)$value                            # -> 0.625 = P(X > 0.5)
integrate(fXy, 0.5, 1)$value                           # -> 0.583 = P(X > 0.5 | Y = 1)
integrate(fXy, 0, 1)$value                             # -> 1 (la condicional integra 1)

# ---- Diap. 44: Independencia (actividad) ------------------------------------
indep <- outer(rowSums(Pxy), colSums(Pxy))             # producto de marginales
indep
round(Pxy - indep, 4)                                  # distinto de 0 -> dependientes
isTRUE(all.equal(as.numeric(Pxy), as.numeric(indep)))  # -> FALSE

# ---- Diap. 45: Independencia: consecuencias ---------------------------------
P <- matrix(c(0.2, 0.3,
              0.1, 0.4), nrow = 2, byrow = TRUE, dimnames = list(X = 0:1, Y = 0:1))
outer(rowSums(P), colSums(P))                          # 0.15 en (0,0) ≠ 0.2 -> dependientes
# Contraste: dos variables independientes simuladas -> cov ≈ 0
set.seed(2026)
a1 <- rnorm(1e5); a2 <- rnorm(1e5)
c(cov = cov(a1, a2), var_suma = var(a1 + a2))          # -> ≈ 0  ≈ 2 (= 1 + 1)

# ---- Diap. 46: Covarianza: intuición con cuadrantes -------------------------
set.seed(11)
xx <- rnorm(120)
par(mfrow = c(1, 3))
for (yy in list(xx + rnorm(120, 0, 0.6), -xx + rnorm(120, 0, 0.6), rnorm(120))) {
  plot(xx, yy, pch = 19, col = ifelse((xx - mean(xx)) * (yy - mean(yy)) > 0, verde, naranja),
       main = sprintf("cov = %.2f", cov(xx, yy)), xlab = "X", ylab = "Y")
  abline(v = mean(xx), h = mean(yy), lty = 3)
}
par(mfrow = c(1, 1))

# ---- Diap. 47–50: Covarianza paso a paso (tabla de la guía) -----------------
xv <- 0:1; yv <- 0:1
# Paso 1: marginales y esperanzas
pX <- rowSums(P); pY <- colSums(P)
EX <- sum(xv * pX)                                     # -> 0.5
EY <- sum(yv * pY)                                     # -> 0.7
# Paso 2: E(XY) = suma de x*y*p(x, y)
outer(xv, yv)                                          # tabla de productos x*y
EXY <- sum(outer(xv, yv) * P)                          # -> 0.4
# Paso 3: covarianza
CovXY <- EXY - EX * EY                                 # -> 0.05
# Paso 4: varianzas e interpretación
VarX <- sum(xv^2 * pX) - EX^2                          # -> 0.25
VarY <- sum(yv^2 * pY) - EY^2                          # -> 0.21
c(EX = EX, EY = EY, EXY = EXY, Cov = CovXY, VarX = VarX, VarY = VarY)

# ---- Diap. 51: Propiedades de la covarianza ---------------------------------
# Var(X + Y) = Var(X) + Var(Y) + 2 Cov(X, Y)
VarX + VarY + 2 * CovXY                                # -> 0.56
# Verificación directa: S = X + Y toma 0, 1, 2
S  <- c(P[1, 1], P[1, 2] + P[2, 1], P[2, 2])           # -> 0.2 0.4 0.4
ES <- sum(0:2 * S)                                     # -> 1.2
sum((0:2)^2 * S) - ES^2                                # -> 0.56
VarX + VarY                                            # -> 0.46 (ignorando la covarianza)
# Cov(aX + b, cY + d) = ac Cov(X, Y), con datos simulados:
set.seed(1)
u <- rnorm(50); w <- u + rnorm(50)
c(cov(u, w), cov(3 * u + 1, 2 * w - 5) / 6)            # iguales

# ---- Diap. 52: Correlación --------------------------------------------------
rho <- CovXY / sqrt(VarX * VarY)
rho                                                    # -> 0.218 (guía)
# Ejemplo de las 3 monedas:
EXm  <- sum(0:2 * rowSums(Pxy)); EYm <- sum(0:2 * colSums(Pxy))
EXYm <- sum(outer(0:2, 0:2) * Pxy)                     # -> 1.25
VXm  <- sum((0:2)^2 * rowSums(Pxy)) - EXm^2            # -> 0.5
VYm  <- sum((0:2)^2 * colSums(Pxy)) - EYm^2            # -> 0.5
(EXYm - EXm * EYm) / sqrt(VXm * VYm)                   # -> 0.5

# ---- Diap. 53: Propiedades de la correlación --------------------------------
set.seed(4)
x <- runif(40, 0, 10)
cor(x, 3 + 2 * x)                                      # -> 1
cor(x, 20 - 1.5 * x)                                   # -> -1
cor(100 * x, 3 + 2 * x)                                # -> 1 (cambiar unidades no altera)
cor(-x, x + rnorm(40, 0, 2.5))                         # signo negativo

# ---- Diap. 54: 3 monedas con propiedades (simulación) -----------------------
set.seed(2026)
L <- matrix(rbinom(3 * 1e5, 1, 0.5), ncol = 3)
X <- L[, 1] + L[, 2]
Y <- L[, 2] + L[, 3]
c(cov = cov(X, Y), cor = cor(X, Y))                    # -> ≈ 0.25 ≈ 0.5
# Si no comparten lanzamiento: X = L1 + L2, Y = L3 -> cov ≈ 0
cov(L[, 1] + L[, 2], L[, 3])

# ---- Diap. 55: Covarianza vs correlación (metros vs centímetros) ------------
set.seed(1)
xm <- rnorm(50); yv2 <- xm + rnorm(50)
c(cov_m = cov(xm, yv2), cov_cm = cov(xm * 100, yv2))   # -> 0.66  65.97
c(cor_m = cor(xm, yv2), cor_cm = cor(xm * 100, yv2))   # -> 0.634 0.634

# ---- Diap. 56: Laboratorio: covarianza y correlación con datos --------------
set.seed(2026)
x  <- rnorm(100, mean = 50, sd = 10)
y  <- 2 * x + rnorm(100, mean = 0, sd = 15)
y2 <- -2 * x + rnorm(100, 0, 15)
par(mfrow = c(1, 2))
plot(x, y,  pch = 19, col = azul,    main = sprintf("cor = %.2f", cor(x, y)))
plot(x, y2, pch = 19, col = naranja, main = sprintf("cor = %.2f", cor(x, y2)))
par(mfrow = c(1, 1))
cov(x, y);  cor(x, y)                                  # -> 192.7   0.80
cov(x, y2); cor(x, y2)                                 # -> -201.3 -0.83

# ---- Diap. 57: Laboratorio: relación no lineal ------------------------------
x <- seq(-3, 3, length.out = 200)
y <- x^2
plot(x, y, pch = 19, col = verde, main = "y = x^2: relación fuerte, cor ≈ 0")
cor(x, y)                                              # -> ≈ 0 (-1.8e-16)

###############################################################################
#  MINI RETO (respuestas para el docente)
###############################################################################

# ---- Diap. 58: Parte A · distribuciones continuas ---------------------------
pnorm(490, 500, 5)                                     # 1) -> 0.0228 Normal(500, 5^2)
1 - pexp(0.25, rate = 6)                               # 2) -> 0.2231 Exp(6/h), 15 min = 0.25 h
pgamma(1, shape = 3, rate = 2)                         # 3) -> 0.3233 Gamma(3, 2)
1 - pbeta(0.9, 8, 2)                                   # 4) -> 0.2252 Beta(8, 2)
punif(15, 10, 20) - punif(12, 10, 20)                  # 5) -> 0.3    U(10, 20)

# ---- Diap. 59: Parte B · vectores aleatorios --------------------------------
R <- matrix(c(.30, .10,
              .15, .20,
              .05, .20), nrow = 3, byrow = TRUE, dimnames = list(X = 0:2, Y = 0:1))
sum(R)                                                 # 1)  -> 1
rowSums(R)                                             # 2)  -> 0.40 0.35 0.25
colSums(R)                                             # 3)  -> 0.50 0.50
R["2", "1"]                                            # 4)  -> 0.20
outer(rowSums(R), colSums(R))                          # 5)  0.30 ≠ 0.20 -> dependientes
EXr  <- sum(0:2 * rowSums(R))                          # 6)  -> 0.85
EYr  <- sum(0:1 * colSums(R))                          # 7)  -> 0.50
EXYr <- sum(outer(0:2, 0:1) * R)                       # 8)  -> 0.60
EXYr - EXr * EYr                                       # 9)  -> 0.175 (positiva)
# Bonus: correlación
VXr <- sum((0:2)^2 * rowSums(R)) - EXr^2               # -> 0.6275
VYr <- sum((0:1)^2 * colSums(R)) - EYr^2               # -> 0.25
(EXYr - EXr * EYr) / sqrt(VXr * VYr)                   # -> 0.442 (moderada)

###############################################################################
#  ANEXOS
###############################################################################

# ---- Diap. 62: Vector aleatorio continuo f(x, y) = x + y --------------------
fxy2 <- function(x, y) x + y
doble(fxy2, 0, 1, 0, 1)                                # -> 1
EXc  <- integrate(function(x) x * (x + 0.5), 0, 1)$value          # -> 7/12
EXYc <- doble(function(x, y) x * y * (x + y), 0, 1, 0, 1)         # -> 1/3
EXYc - EXc^2                                           # -> -0.00694 = -1/144
xg <- seq(0, 1, length.out = 50)
persp(xg, xg, outer(xg, xg, "+"), theta = 30, phi = 25, col = celeste,
      xlab = "X", ylab = "Y", zlab = "f(x,y)", main = "f(x, y) = x + y")

# ---- Diap. 63: Encontrar la constante (anexo, ej. 6) ------------------------
f6 <- function(x) 2 * x / 9
# ∫_0^c 2x/9 dx = c^2/9 = 1  ->  c = 3
uniroot(function(c) integrate(f6, 0, c)$value - 1, c(1, 5))$root   # -> 3
integrate(f6, 0, 3)$value                              # -> 1
integrate(f6, 1, 2)$value                              # -> 0.3333
E6  <- integrate(function(x) x * f6(x), 0, 3)$value    # -> 2
E62 <- integrate(function(x) x^2 * f6(x), 0, 3)$value  # -> 4.5
E62 - E6^2                                             # -> 0.5

par(op)
cat("\nFin del script. ¡Todo ejecutado sin errores!\n")
