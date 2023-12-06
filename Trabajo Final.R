library(readr)
library(ggplot2)
library(ChainLadder)
library(dplyr)
library(glmnet)
library(tidyr)
library(purrr)
library(class)
library(caret)
library(keras)
library(caret)

datos <- read.csv("wkcomp_pos.csv")

# Número de filas y columnas del conjunto de datos
dim(datos)

# Nombres de las variables del conjunto de datos
names(datos)

# Tipo de dato de cada variable del conjunto de datos
class(datos)

# Resumen estadístico de cada variable del conjunto de datos
summary(datos)

# Número de aseguradoras únicas
n_aseguradoras <- length(unique(datos$GRNAME))
cat("El número de aseguradoras es", n_aseguradoras, "\n")

# Datos por la variable Single
datos_single <- datos[datos$Single == 1, ]

# Quitamos valores duplicados de la variable GRNAME
aseguradoras_single <- unique(datos_single$GRNAME)

# Número de aseguradoras únicas que pertenecen a un grupo
n_single <- length(aseguradoras_single)
cat("El número de aseguradoras únicas que pertenecen a un grupo es", n_single, "\n")

# Tendencia de la pérdida incurrida en dólares por año de accidente
media_perdida <- aggregate(IncurLoss_D ~ AccidentYear, data = datos, FUN = mean)
plot(media_perdida, type = "l", main = "Tendencia de la pérdida incurrida", xlab = "Año de accidente", ylab = "Media de la pérdida incurrida")


# Relación entre la pérdida incurrida y la prima devengada neta
correlacion <- cor(datos$IncurLoss_D, datos$EarnedPremNet_D)
cat("El coeficiente de correlación entre la pérdida incurrida y la prima devengada neta es", correlacion, "\n")
plot(IncurLoss_D ~ EarnedPremNet_D, data = datos, main = "Relación entre la pérdida incurrida y la prima devengada neta", xlab = "Prima devengada neta", ylab = "Pérdida incurrida")


# Outliers de la pérdida incurrida en dólares
boxplot(datos$IncurLoss_D, main = "Outliers de la pérdida incurrida", xlab = "Pérdida incurrida", outline = FALSE)
outliers <- boxplot.stats(datos$IncurLoss_D)$out
cat("Los valores de los outliers son:\n")
print(outliers)


# Homogeneidad de los datos de cada aseguradora
desviacion <- as.data.frame(tapply(datos$IncurLoss_D, datos$GRNAME, FUN = sd))
plot(desviacion, main = "Homogeneidad de los datos de cada aseguradora", xlab = "Nombre del grupo de aseguradoras", ylab = "Desviación estándar de la pérdida incurrida")


# Conjunto de datos de desviación estándar de forma descendente
desviacion <- desviacion[order(-desviacion$sd), ]

# Aseguradora con la mayor desviación estándar
aseguradora <- rownames(desviacion)[1]

# Gráfica desviación
abline(h = desviacion[1, ], col = "red")

# Gráfica con el nombre de la aseguradora con la mayor desviación estándar
text(x = 1, y = desviacion[1, ] + 100000, labels = aseguradora, col = "red")

# Número de observaciones y variables del conjunto de datos
dim(datos)

# Tipos de datos de las variables del conjunto de datos
str(datos)

# Estadísticas descriptivas de las variables numéricas del conjunto de datos
summary(datos)

# Valores faltantes (NA) de las variables del conjunto de datos
sapply(datos, function(x) sum(is.na(x)))

# Seis observaciones del conjunto de datos
head(datos)

# Seis observaciones del conjunto de datos
tail(datos)

# Elininación de variable
datos <- select(datos, -GRCODE)

# Conversión de variable
datos$GRNAME <- as.factor(datos$GRNAME)
str(datos$GRNAME)

datos1 <- filter(datos, DevelopmentYear <= 1997)

# Conjunto de datos filtrado en un objeto de clase triangle
triangulo <- as.triangle(datos1, origin = "AccidentYear", dev = "DevelopmentYear", value = "IncurLoss_D")

# Aplicación del modelo chain ladder determinístico
modeloCL <- chainladder(triangulo)

# Predicciones del modelo para el conjunto de datos
y_pred <- fitted(modeloCL)

# Valores reales del conjunto de datos
y_real <- datos1$IncurLoss_D

# MSE entre las predicciones y los valores reales
mseCL <- mean((y_pred - y_real)^2)
print(mseCL)


# RIDGE, LASSO y ELASTIC
# Convertimos las variables numéricas que representen categorías en factores
datos$GRNAME <- as.factor(datos$GRNAME)
datos$AccidentYear <- as.factor(datos$AccidentYear)
datos$DevelopmentYear <- as.factor(datos$DevelopmentYear)

# Creamos una nueva variable que sea la pérdida total
datos <- mutate(datos, TotalLoss_D = IncurLoss_D + CumPaidLoss_D + BulkLoss_D)

# Eliminamos las filas que tienen valores negativos o cero en la variable pérdida total
datos <- filter(datos, TotalLoss_D > 0)

# Transformamos la variable pérdida total
datos$TotalLoss_D <- log(datos$TotalLoss_D)

# Dividimos el conjunto de datos en dos subconjuntos: uno de entrenamiento y otro de prueba
set.seed(123) 
n <- nrow(datos) 
train_index <- sample(1:n, size = 0.9 * n) 
train <- datos[train_index, ] 
test <- datos[-train_index, ] 

# Ajustamos un modelo de regresión Lasso
x_train <- model.matrix(TotalLoss_D ~ ., data = train) 
y_train <- train$TotalLoss_D 
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1) 
lasso_fit <- glmnet(x_train, y_train, alpha = 1, lambda = cv_fit$lambda.min)

# Evaluamos el rendimiento del modelo
x_test <- model.matrix(TotalLoss_D ~ ., data = test)
y_test <- test$TotalLoss_D
y_pred <- predict(lasso_fit, newx = x_test)
y_pred <- exp(y_pred)
y_test <- exp(y_test)
mseL <- mean((y_pred - y_test)^2)
print(mseL)


# Ajustamos un modelo de regresión Ridge
x_trainRid <- model.matrix(TotalLoss_D ~ ., data = train)
y_trainRid <- train$TotalLoss_D
cv_fitRid <- cv.glmnet(x_trainRid, y_trainRid, alpha = 0)
ridge_fit <- glmnet(x_trainRid, y_trainRid, alpha = 0, lambda = cv_fitRid$lambda.min)

# Evaluamos el rendimiento del modelo
x_testRid <- model.matrix(TotalLoss_D ~ ., data = test)
y_testRid <- test$TotalLoss_D
y_predRid <- predict(ridge_fit, newx = x_test)
y_predRid <- exp(y_predRid)
y_testRid <- exp(y_testRid)
mseRid <- mean((y_predRid - y_testRid)^2)
print(mseRid)

# Ajustamos un modelo de regresión Elastic Net
x_trainEN <- model.matrix(TotalLoss_D ~ ., data = train)
y_trainEN <- train$TotalLoss_D
cv_fitEN <- cv.glmnet(x_trainEN, y_trainEN, alpha = 0.5)
elastic_fit <- glmnet(x_trainEN, y_trainEN, alpha = 0.5, lambda = cv_fitEN$lambda.min)

# Evaluamos el rendimiento del modelo
x_testEN <- model.matrix(TotalLoss_D ~ ., data = test)
y_testEN <- test$TotalLoss_D
y_predEN <- predict(elastic_fit, newx = x_test)
y_predEN <- exp(y_predEN)
y_testEN <- exp(y_testEN)
mseEN <- mean((y_predEN - y_testEN)^2)
print(mseEN)

# RED NEURONAL

datos <- read.csv("wkcomp_pos.csv")
datos <- na.omit(datos)

# Creamos la variable pérdida total
datos <- mutate(datos, TotalLoss_D = IncurLoss_D + CumPaidLoss_D + BulkLoss_D)

# Eliminamos las filas que tengan valores negativos o cero en la variable pérdida total
datos <- filter(datos, TotalLoss_D > 0)

# Creamos la variable objetivo (TotalLoss_D) y los atributos predictivos (AccidentYear y DevelopmentYear)
y <- datos$TotalLoss_D
X <- datos[, c("AccidentYear", "DevelopmentYear")]

# Normalizamos los datos
X <- scale(X)
y <- scale(y)

# Creamos el modelo de red neuronal
model <- keras_model_sequential() %>%
  
# Primera capa oculta con 64 unidades y función de activación relu
layer_dense(units = 64, activation = "relu", input_shape = c(2)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

# Compilamos el modelo con el optimizador adam, la función de pérdida mean_squared_error y la métrica mean_squared_error
model %>% compile(
  optimizer = optimizer_adam(),
  loss = "mean_squared_error",
  metrics = c("mean_squared_error")
)

# Creamos una función para calcular el Mean Squared Error (MSE)
mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

# Dividimos los datos en conjuntos de entrenamiento y prueba
set.seed(123)  
index <- sample(1:nrow(datos), nrow(datos)*0.9)
train_data <- datos[index, ]
test_data <- datos[-index, ]

X_train <- scale(train_data[, c("AccidentYear", "DevelopmentYear")])
y_train <- scale(train_data$TotalLoss_D)
X_test <- scale(test_data[, c("AccidentYear", "DevelopmentYear")])
y_test <- scale(test_data$TotalLoss_D)

# Especificamos el método de validación cruzada dejando uno fuera (LOOCV)
ctrl <- trainControl(method = "LOOCV")

# Configuramos los callbacks con una menor frecuencia de actualización
callbacks <- list(
  callback_early_stopping(patience = 10),
  callback_model_checkpoint("model_checkpoint.h5", save_best_only = TRUE),
  callback_tensorboard("logs", update_freq = "epoch")  
)

# Ajustamos el modelo con los datos de entrenamiento
fit <- model %>% fit(
  x = as.matrix(X_train),
  y = y_train,
  epochs = 50,
  batch_size = 20,
  verbose = 1,
  callbacks = callbacks,
  validation_data = list(as.matrix(X_test), y_test),
  trControl = ctrl
)

# Calculamos el MSE para el modelo
nn_avg_mse <- mean((predict(model, as.matrix(X_test)) - y_test)^2, na.rm = TRUE)

# Imprimimos el resultado
cat("Neural Network LOOCV-MSE:", nn_avg_mse, "\n")

