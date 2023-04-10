########arboles de regresion
library(MASS)
library(dplyr)
library(tidyr)
library(skimr)

library(ggplot2)
library(ggpubr)

library(tidymodels)
library(ranger)
library(doParallel)
library(tree)


library(rpart)
library(randomForest)
library(rfutilities)
library(rpart.plot)

library(BART)


install.packages("partykit", dependencies = TRUE)
install.packages("constparty", dependencies = TRUE)
install.packages("grid", dependencies = TRUE)
install.packages("libcoin", dependencies = TRUE)
install.packages("mvtnorm", dependencies = TRUE)
library(partykit)
library(ggparty)

############################################################
RF_1 <- RF[,0:18]
RF_1
RF_2 <- RF_1[,5:18]
RF_2
#cambio los nombres de la columna "Estadio"
RF_2$Estadio[RF_2$Estadio == "Z1.2"] <- "v"
RF_2$Estadio[RF_2$Estadio == "Z1.3"] <- "v"
RF_2$Estadio[RF_2$Estadio == "Z1.4"] <- "v"
RF_2$Estadio[RF_2$Estadio == "Z1.5"] <- "v"
RF_2$Estadio[RF_2$Estadio == "espiga"] <- "r"
RF_2$Estadio[RF_2$Estadio == "postsecado"] <- "barbecho"
RF_2$Estadio[RF_2$Estadio == "V1"] <- "v"
RF_2$Estadio[RF_2$Estadio == "v1"] <- "v"
RF_2$Estadio[RF_2$Estadio == "V2"] <- "v"
RF_2$Estadio[RF_2$Estadio == "V2/V3"] <- "v"
RF_2$Estadio[RF_2$Estadio == "V4"] <- "v"
RF_2$Estadio[RF_2$Estadio == "V6"] <- "v"
RF_2$Estadio[RF_2$Estadio == "R1"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R1-R2"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R1/R2"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R2"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R3"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R4"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R5"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R6"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R7"] <- "r"
RF_2$Estadio[RF_2$Estadio == "R8"] <- "r"
RF_2$Estadio[RF_2$Estadio == "postcosecha"] <- "barbecho"
RF_2$Estadio[RF_2$Estadio == "postsiembra"] <- "emergencia"
print(RF_2, n= 204)
sum(is.na(RF_2$Estadio))

RF_2
RF_7 <- RF_2[,-6] #sin hum
RF_7
RF_8 <- RF_7[,-11] #sin co2
RF_8
RF_8 <- RF_8[,-10] #sin ch4
RF_8
head(RF_8, 5)

skim(RF_8)


#arbol de regresión, 1 sólo arbol
# División de los RF_8 en train y test
# ==============================================================================
set.seed(123)
train <- sample(1:nrow(RF_8), size = nrow(RF_8)/2)
RF_8_train <- RF_8[train,]
RF_8_test  <- RF_8[-train,]

# Creación y entrenamiento del modelo
# ==============================================================================
set.seed(123)
arbol_regresion <- tree::tree(
  formula = n2o ~ .,
  data    = RF_8_train,
  split   = "deviance",
  mincut  = 20,
  minsize = 50
)

summary(arbol_regresion)

# Estructura del árbol creado
# ==============================================================================
par(mar = c(1,1,1,1))
plot(x = arbol_regresion, type = "proportional")
text(x = arbol_regresion, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")

# Pruning (const complexity pruning) por validación cruzada
# ==============================================================================

# El árbol se crece al máximo posible para luego aplicar el pruning
arbol_regresion <- tree(
  formula = n2o ~ .,
  data    = RF_8_train,
  split   = "deviance",
  mincut  = 1,
  minsize = 2,
  mindev  = 0
)

# Búsqueda por validación cruzada
set.seed(123)
cv_arbol <- cv.tree(arbol_regresion, K = 5)

# Tamaño óptimo encontrado
# ==============================================================================
size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
paste("Tamaño óptimo encontrado:", size_optimo)

resultados_cv <- data.frame(
  n_nodos  = cv_arbol$size,
  deviance = cv_arbol$dev,
  alpha    = cv_arbol$k
)

p1 <- ggplot(data = resultados_cv, aes(x = n_nodos, y = deviance)) +
  geom_line() + 
  geom_point() +
  geom_vline(xintercept = size_optimo, color = "red") +
  labs(title = "Error vs tamaño del árbol") +
  theme_bw() 

p2 <- ggplot(data = resultados_cv, aes(x = alpha, y = deviance)) +
  geom_line() + 
  geom_point() +
  labs(title = "Error vs penalización alpha") +
  theme_bw() 

ggarrange(p1, p2)

# Estructura del árbol creado final
# ==============================================================================
arbol_final <- prune.tree(
  tree = arbol_regresion,
  best = size_optimo
)

par(mar = c(1,1,1,1))
plot(x = arbol_final, type = "proportional")
text(x = arbol_final, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")


# Error de test del modelo inicial
# ==============================================================================
predicciones <- predict(arbol_regresion, newdata = RF_8_test)
test_rmse    <- sqrt(mean((predicciones - RF_8_test$n2o)^2))
paste("Error de test (rmse) del árbol inicial:", round(test_rmse,2))

# Error de test del modelo final
# ==============================================================================
predicciones <- predict(arbol_final, newdata = RF_8_test)
test_rmse    <- sqrt(mean((predicciones - RF_8_test$n2o)^2))
paste("Error de test (rmse) del árbol final:", round(test_rmse,2))

####### RandomForest
set.seed(123)
train       <- sample(1:nrow(RF_8), size = nrow(RF_8)/2)
datos_train <- RF_8[train,]
datos_test  <- RF_8[-train,]

set.seed(123)
modelo  <- ranger(
  formula   = n2o ~ .,
  data      = datos_train,
  num.trees = 500,
  seed      = 123
)

print(modelo)


profundidad <- tree.depth(modelo$forest[[1]])
ggparty(modelo) +
  geom_edge() +
  geom_edge_label() +
  geom_node_label(aes(label = splitvar), ids = "inner") +
  # identical to  geom_node_splitvar() +
  geom_node_label(aes(label = info), ids = "terminal")

# Error de test del modelo
# ==============================================================================
predicciones <- predict(
  modelo,
  data = datos_test
)

predicciones <- predicciones$predictions
test_rmse    <- sqrt(mean((predicciones - datos_test$n2o)^2))
paste("Error de test (rmse) del modelo:", round(test_rmse,2))


# Validación empleando el Out-of-Bag error (root mean squared error)
# ==============================================================================

# Valores evaluados
num_trees_range <- seq(1, 400, 20)

# Bucle para entrenar un modelo con cada valor de num_trees y extraer su error
# de entrenamiento y de Out-of-Bag.

train_errors <- rep(NA, times = length(num_trees_range))
oob_errors   <- rep(NA, times = length(num_trees_range))

for (i in seq_along(num_trees_range)){
  modelo  <- ranger(
    formula   = n2o ~ .,
    data      = datos_train,
    num.trees = num_trees_range[i],
    oob.error = TRUE,
    seed      = 123
  )
  
  predicciones_train <- predict(
    modelo,
    data = datos_train
  )
  predicciones_train <- predicciones_train$predictions
  
  train_error <- mean((predicciones_train - datos_train$medv)^2)
  oob_error   <- modelo$prediction.error
  
  train_errors[i] <- sqrt(train_error)
  oob_errors[i]   <- sqrt(oob_error)
  
}

# Gráfico con la evolución de los errores
df_resulados <- data.frame(n_arboles = num_trees_range, train_errors, oob_errors)
ggplot(data = df_resulados) +
  geom_line(aes(x = num_trees_range, y = train_errors, color = "train rmse")) + 
  geom_line(aes(x = num_trees_range, y = oob_errors, color = "oob rmse")) +
  geom_vline(xintercept = num_trees_range[which.min(oob_errors)],
             color = "firebrick",
             linetype = "dashed") +
  labs(
    title = "Evolución del out-of-bag-error vs número árboles",
    x     = "número de árboles",
    y     = "out-of-bag-error (rmse)",
    color = ""
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

paste("Valor óptimo de num.trees:", num_trees_range[which.min(oob_errors)])


# Validación empleando k-cross-validation (root mean squared error)
# ==============================================================================

# Valores evaluados
num_trees_range <- seq(1, 400, 10)

# Bucle para entrenar un modelo con cada valor de num_trees y extraer su error
# de entrenamiento y de Out-of-Bag.

train_errors <- rep(NA, times = length(num_trees_range))
cv_errors    <- rep(NA, times = length(num_trees_range))

for (i in seq_along(num_trees_range)){
  
  # Definición del modelo
  modelo <- rand_forest(
    mode  = "regression",
    trees = num_trees_range[i]
  ) %>%
    set_engine(
      engine = "ranger",
      seed   = 123
    )
  
  # Particiones validación cruzada
  set.seed(1234)
  cv_folds <- vfold_cv(
    data    = datos_train,
    v       = 5,
    repeats = 1
  )
  
  # Ejecución validación cruzada
  validacion_fit <- fit_resamples(
    preprocessor = n2o ~ .,
    object       = modelo,
    resamples    = cv_folds,
    metrics      = metric_set(rmse)
  )
  
  # Extraer la métrica de validación 
  cv_error <- collect_metrics(validacion_fit)$mean
  
  # Predicción datos train
  modelo_fit <- modelo %>% fit(n2o ~ ., data = datos_train)
  predicciones_train <- predict(
    modelo_fit,
    new_data = datos_train
  )
  predicciones_train <- predicciones_train$.pred
  train_error <- sqrt(mean((predicciones_train - datos_train$n2o)^2))
  
  # Resultados
  train_errors[i] <- train_error
  cv_errors[i]    <- cv_error
}

# Gráfico con la evolución de los errores
df_resulados <- data.frame(n_arboles = num_trees_range, train_errors, cv_errors)
ggplot(data = df_resulados) +
  geom_line(aes(x = num_trees_range, y = train_errors, color = "train rmse")) + 
  geom_line(aes(x = num_trees_range, y = cv_errors, color = "cv rmse")) +
  geom_vline(xintercept = num_trees_range[which.min(cv_errors)],
             color = "firebrick",
             linetype = "dashed") +
  labs(
    title = "Evolución del cv-error vs número árboles",
    x     = "número de árboles",
    y     = "cv-error (rmse)",
    color = ""
  ) +
  theme_bw() +
  theme(legend.position = "bottom")


paste("Valor óptimo de num.trees:", num_trees_range[which.min(cv_errors)])

# Validación empleando el Out-of-Bag error (root mean squared error)
# ==============================================================================

# Valores evaluados
mtry_range <- seq(1, ncol(datos_train)-1)

# Bucle para entrenar un modelo con cada valor de mtry y extraer su error
# de entrenamiento y de Out-of-Bag.

train_errors <- rep(NA, times = length(mtry_range))
oob_errors   <- rep(NA, times = length(mtry_range))

for (i in seq_along(mtry_range)){
  modelo  <- ranger(
    formula   = n2o ~ .,
    data      = datos_train,
    num.trees = 50,
    mtry      = mtry_range[i],
    oob.error = TRUE,
    seed      = 123
  )
  
  predicciones_train <- predict(
    modelo,
    data = datos_train
  )
  predicciones_train <- predicciones_train$predictions
  
  train_error <- mean((predicciones_train - datos_train$n2o)^2)
  oob_error   <- modelo$prediction.error
  
  train_errors[i] <- sqrt(train_error)
  oob_errors[i]   <- sqrt(oob_error)
  
}

# Gráfico con la evolución de los errores
df_resulados <- data.frame(mtry = mtry_range, train_errors, oob_errors)
ggplot(data = df_resulados) +
  geom_line(aes(x = mtry_range, y = train_errors, color = "train rmse")) + 
  geom_line(aes(x = mtry_range, y = oob_errors, color = "oob rmse")) +
  geom_vline(xintercept =  mtry_range[which.min(oob_errors)],
             color = "firebrick",
             linetype = "dashed") +
  labs(
    title = "Evolución del out-of-bag-error vs mtry",
    x     = "mtry",
    y     = "out-of-bag-error (rmse)",
    color = ""
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

paste("Valor óptimo de mtry:", mtry_range[which.min(oob_errors)])

# Validación empleando k-cross-validation (root mean squared error)
# ==============================================================================

# Valores evaluados
mtry_range <- seq(1, ncol(datos_train)-1)

# Bucle para entrenar un modelo con cada valor de mtry y extraer su error
# de entrenamiento y de Out-of-Bag.

train_errors <- rep(NA, times = length(mtry_range))
cv_errors    <- rep(NA, times = length(mtry_range))

for (i in seq_along(mtry_range)){
  
  # Definición del modelo
  modelo <- rand_forest(
    mode  = "regression",
    trees = 151,
    mtry  = mtry_range[i]
  ) %>%
    set_engine(
      engine = "ranger",
      seed   = 123
    )
  
  # Particiones validación cruzada
  set.seed(1234)
  cv_folds <- vfold_cv(
    data    = datos_train,
    v       = 5,
    repeats = 1
  )
  
  # Ejecución validación cruzada
  validacion_fit <- fit_resamples(
    preprocessor = n2o ~ .,
    object       = modelo,
    resamples    = cv_folds,
    metrics      = metric_set(rmse)
  )
  
  # Extraer datos de validación
  cv_error <- collect_metrics(validacion_fit)$mean
  
  # Predicción datos train
  modelo_fit <- modelo %>% fit(n2o ~ ., data = datos_train)
  predicciones_train <- predict(
    modelo_fit,
    new_data = datos_train
  )
  predicciones_train <- predicciones_train$.pred
  
  train_error <- sqrt(mean((predicciones_train - datos_train$n2o)^2))
  
  # Resultados
  train_errors[i] <- train_error
  cv_errors[i]    <- cv_error
  
}

# Gráfico con la evolución de los errores
df_resulados <- data.frame(mtry = mtry_range, train_errors, cv_errors)
ggplot(data = df_resulados) +
  geom_line(aes(x = mtry_range, y = train_errors, color = "train error")) + 
  geom_line(aes(x = mtry_range, y = cv_errors, color = "cv error")) +
  geom_vline(xintercept =  mtry_range[which.min(cv_errors)],
             color = "firebrick",
             linetype = "dashed") +
  labs(
    title = "Evolución del out-of-bag-error vs mtry",
    x     = "mtry",
    y     = "cv-error (mse)",
    color = ""
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

paste("Valor óptimo de mtry:", mtry_range[which.min(cv_errors)])

########Analisis de todos los hiperparametros juntos

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = expand_grid(
  'num_trees' = c(50, 100, 500, 1000, 5000),
  'mtry'      = c(3, 10, ncol(datos_train)-1),
  'max_depth' = c(1, 3, 10, 20)
  
)

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================

oob_error = rep(NA, nrow(param_grid))

for(i in 1:nrow(param_grid)){
  
  modelo <- ranger(
    formula   = n2o ~ .,
    data      = datos_train, 
    num.trees = param_grid$num_trees[i],
    mtry      = param_grid$mtry[i],
    max.depth = param_grid$max_depth[i],
    seed      = 123
  )
  
  oob_error[i] <- sqrt(modelo$prediction.error)
}


# Resultados
# ==============================================================================
resultados <- param_grid
resultados$oob_error <- oob_error
resultados <- resultados %>% arrange(oob_error)

head(resultados, 1)


# DEFINICIÓN DEL MODELO Y DE LOS HIPERPARÁMETROS A OPTIMIZAR
# ==============================================================================
modelo <- rand_forest(
  mode  = "regression",
  mtry  = tune(),
  trees = tune()
) %>%
  set_engine(
    engine     = "ranger",
    max.depth  = tune(),
    importance = "none",
    seed       = 123
  )

# DEFINICIÓN DEL PREPROCESADO
# ==============================================================================
# En este caso no hay preprocesado, por lo que el transformer solo contiene
# la definición de la fórmula y los datos de entrenamiento.
transformer <- recipe(
  formula = n2o ~ .,
  data    =  datos_train
)

# DEFINICIÓN DE LA ESTRATEGIA DE VALIDACIÓN Y CREACIÓN DE PARTICIONES
# ==============================================================================
set.seed(1234)
cv_folds <- vfold_cv(
  data    = datos_train,
  v       = 5,
  strata  = n2o
)

# WORKFLOW
# ==============================================================================
workflow_modelado <- workflow() %>%
  add_recipe(transformer) %>%
  add_model(modelo)


# GRID DE HIPERPARÁMETROS
# ==============================================================================
hiperpar_grid <- expand_grid(
  'trees'     = c(50, 100, 500, 1000, 5000),
  'mtry'      = c(3, 10, ncol(datos_train)-1),
  'max.depth' = c(1, 3, 10, 20)
)

# EJECUCIÓN DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS
# ==============================================================================
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

grid_fit <- tune_grid(
  object    = workflow_modelado,
  resamples = cv_folds,
  metrics   = metric_set(rmse),
  grid      = hiperpar_grid
)

stopCluster(cl)

# Mejores hiperparámetros por validación cruzada
# ==============================================================================
show_best(grid_fit, metric = "rmse", n = 1)


# ENTRENAMIENTO FINAL
# =============================================================================
mejores_hiperpar <- select_best(grid_fit, metric = "rmse")

modelo_final_fit <- finalize_workflow(
  x = workflow_modelado,
  parameters = mejores_hiperpar
) %>%
  fit(
    data = datos_train
  ) %>%
  pull_workflow_fit()


# Error de test del modelo final
# ==============================================================================
predicciones <- modelo_final_fit %>%
  predict(
    new_data = datos_test,
    type     = "numeric"
  )

predicciones <- predicciones %>% 
  bind_cols(datos_test %>% dplyr::select(n2o))

rmse_test  <- rmse(
  data     = predicciones,
  truth    = n2o,
  estimate = .pred,
  na_rm    = TRUE
)
rmse_test

####################
# Entrenamiento modelo
modelo <- rand_forest(
  mode  = "regression"
) %>%
  set_engine(
    engine     = "ranger",
    importance = "impurity",
    seed       = 123
  )

modelo <- modelo %>% finalize_model(mejores_hiperpar)
modelo <- modelo %>% fit(n2o ~., data = datos_train)

# Importancia
importancia_pred <- modelo$fit$variable.importance %>%
  enframe(name = "predictor", value = "importancia")

# Gráfico
ggplot(
  data = importancia_pred,
  aes(x    = reorder(predictor, importancia),
      y    = importancia,
      fill = importancia)
) +
  labs(x = "predictor", title = "Importancia predictores (pureza de nodos)") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none")

#Importancia por permutación
# Entrenamiento modelo
modelo <- rand_forest(
  mode  = "regression"
) %>%
  set_engine(
    engine     = "ranger",
    importance = "permutation",
    seed       = 123
  )

modelo <- modelo %>% finalize_model(mejores_hiperpar)
modelo <- modelo %>% fit(n2o ~., data = datos_train)

# Importancia
importancia_pred <- modelo$fit$variable.importance %>%
  enframe(name = "predictor", value = "importancia")

# Gráfico
ggplot(
  data = importancia_pred,
  aes(x    = reorder(predictor, importancia),
      y    = importancia,
      fill = importancia)
) +
  labs(x = "predictor", title = "Importancia predictores (permutación)") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none")

#########Dibujo del árbol
tree<-rpart(n2o~., data = datos_train, cp=0)
summary(tree)
prp(tree)

par(mfrow=c(1,1), xpd=NA)
plot(tree)
text(tree, use.n=T)

rpart.plot(tree, type=0, extra=0, digits=1, varlen=0, box.palette="Grays", shadow.col="darkgray")

tree_1 <- rpart(n2o ~ ., data = datos_train, cp = 0)
printcp(tree_1)
plotcp(tree_1)

head(tree_1$cptable, 10)
xerror <- tree_1$cptable[,"xerror"]
imin.xerror <- which.min(xerror)
# Valor óptimo
tree_1$cptable[imin.xerror, ]

# Límite superior "oneSE rule" y complejidad mínima por debajo de ese valor
upper.xerror <- xerror[imin.xerror] + tree_1$cptable[imin.xerror, "xstd"]
icp <- min(which(xerror <= upper.xerror))
cp <- tree_1$cptable[icp, "CP"]

tree <- prune(tree_1, cp = cp)
rpart.plot(tree) 

##########Bayesian Additive Regression Trees
library(BART)
x <- RF_8[, -9]
y <- RF_8[, 9]
xtrain <- x[train, ]
ytrain <- y[train]

xtest <- x[-train, ]
ytest <- y[-train]
set.seed (1)
bartfit <- gbart(xtrain , ytrain , x.test = xtest)
##########no funciona el y[train]

##PROBAR ESTO
#http://rafalab.dfci.harvard.edu/dsbook/examples-of-algorithms.html#classification-and-regression-trees-cart
datos_train
fit <- randomForest(n2o~., data = datos_train)
datos_train |>
  mutate(y_hat = predict(fit, newdata = datos_train)) |> 
  ggplot() +
  geom_point(aes(########date, n2o)) +
  geom_line(aes(#########date, y_hat), col="red")

#Matriz de confusion
train_rf <- randomForest(y ~ ., data=mnist_27$train)

confusionMatrix(predict(train_rf, mnist_27$test),
                mnist_27$test$y)$overall["Accuracy"]

####sacar Estadio y probar soja y trigo por separado