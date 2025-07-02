#Histogram of age
age<-data$Age
hist(age,xlab="age",ylab="frequency",col="light blue")

#Histogram of BMI
bmi<-data$BMI
hist(bmi,xlab="BMI",ylab="frequency",col="lightpink",main="Histogram of BMI")

#correlation Heat map
library(corrplot)
corr_matrix <- cor(data[, c("Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1")], use="complete.obs")
library(pheatmap)
pheatmap(corr_matrix, color=colorRampPalette(c("blue", "white", "red"))(50), display_numbers=TRUE, fontsize_number=10)
