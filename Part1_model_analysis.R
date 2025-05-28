setwd('C:/Users/alexw/Assignments/AIML331/AIML331_Assignment3')
data <- read.csv('Part1_model_results.csv')

data2 <- read.csv('part2_model_results.csv')

data

data$activation <- as.factor(data$activation)
data$residuals <- as.factor(data$residuals)
data$batch_normalisation <- as.factor(data$batch_normalisation)
model <- lm(accuracy ~ batch_normalisation + residuals + num_layers * activation, data = data)
summary(model)


median_3_accuracy <- mean(data[data$num_layers == 3, "accuracy"])
median_4_accuracy <- mean(data[data$num_layers == 4, "accuracy"])
median_5_accuracy <- mean(data[data$num_layers == 5, "accuracy"])
median_6_accuracy <- mean(data[data$num_layers == 6, "accuracy"])

mean_acc <- median(data$accuracy)
mean_acc

median_ReLU_accuracy <- mean(data[data$activation == "ReLU", "accuracy"])
median_Leaky_accuracy <- mean(data[data$activation == "LeakyReLU", "accuracy"])
median_GELU_accuracy <- mean(data[data$activation == "GELU", "accuracy"])

data2

mean_acc2 <- median(data2$accuracy)
mean_acc2



data$pos_embedding <- as.factor(data$pos_embedding)
model <- lm(accuracy ~ pos_embedding + patch_size + num_layers, data = data2)
summary(model)

learnable_data <- data2[data2$pos_embedding == "Learnable", ]
learnable_data
learnable_data$num_layers <- as.factor(learnable_data$num_layers)
learnable_data$patch_size <- as.factor(learnable_data$patch_size)
model <- lm(accuracy ~ patch_size + num_layers, data = learnable_data)
summary(model)
