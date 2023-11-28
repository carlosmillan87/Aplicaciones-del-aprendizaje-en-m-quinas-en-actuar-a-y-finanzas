library(naivebayes)
library(keras)

mnist <- dataset_mnist()

# Extraemos los conjuntos de datos de entrenamiento y prueba desde el objeto mnist

train <- mnist$train
test <- mnist$test

# Convertimos las etiquetas de entrenamiento train$y a un factor, toda vez que se espera que sean de tipo factor para el modelo de Naive Bayes

train$y <- as.factor(train$y)

# Entrenamos un modelo de Naive Bayes con las imágenes (x) como características y las etiquetas (y) como variable de respuesta

nb_model <- naive_bayes(x = train$x, y = train$y)

# Seleccionamos aleatoriamente un índice de imagen del conjunto de prueba

test_index <- sample(1:nrow(test$x), 1)

# Convertimos el tensor de la imagen seleccionada a una matriz

test_image_matrix <- array_reshape(test$x[test_index, , , drop = FALSE], c(1, 784))

# Usamos el modelo de Naive Bayes para predecir la probabilidad de cada dígito para la imagen de prueba

prob <- predict(nb_model, newdata = as.data.frame(test_image_matrix), type = "prob")

# Generamos un nuevo número aleatorio basado en las probabilidades predichas por el modelo de Naive Bayes

new_number <- sample(levels(train$y), 1, prob = prob)

# Mostramos el resultado

cat("La imagen de prueba es el dígito", test$y[test_index], "\n")
cat("La probabilidad de cada dígito es:\n")
print(prob)
cat("El nuevo número generado es", new_number, "\n")
