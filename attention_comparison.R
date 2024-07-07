if(!require("ggplot2")){
  install.packages("ggplot2")
}
library(ggplot2)

if(!require("dplyr")) {
  install.packages("dplyr")
}
library(dplyr)

if(!require("arrow")) {
  install.packages("arrow")
}
library(arrow)

library(glue)


data_path <- file.path(getwd(), "data")

plot_seq_mean_attention <- function(model, data_path) {

  data_ft <- read_parquet(file.path(data_path, glue("attention_weights_{model}_FT_selected.parquet")))
  data_pt <- read_parquet(file.path(data_path, glue("attention_weights_{model}_PT_selected.parquet")))

  #head(data_ft)
  #all.equal(data_ft, data_pt)

  chains <- c("H", "L")

  # Skip cross-chain pairs
  data_ft_mean <- data_ft %>%
    filter(Chain_1 == Chain_2, Chain_1 %in% chains) %>%
    group_by(Seq_2) %>%
    summarise(mean_attention = mean(Weight))

  data_pt_mean <- data_pt %>%
    filter(Chain_1 == Chain_2, Chain_1 %in% chains) %>%
    group_by(Seq_2) %>%
    summarise(mean_attention = mean(Weight))

  data_mean <- data_ft_mean
  data_mean$mean_attention <- data_ft_mean$mean_attention - data_pt_mean$mean_attention

  light_chain_start <- 128

  # CDR1-3 region locations for IMGT taken from:
  # https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
  cdr_regions <- data.frame(
    xmin = append(c(27, 56, 105), c(27, 56, 105) + light_chain_start),
    xmax = append(c(38, 65, 117), c(38, 65, 117) + light_chain_start),
    ymin = -Inf,
    ymax = Inf)

  ggplot(data_mean, aes(x = Seq_2, y = mean_attention, group = 1)) +
    geom_rect(data = cdr_regions,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = "grey", alpha = 0.5, inherit.aes = FALSE) +
    geom_line(linetype = "solid") +
    labs(title = glue("Model: {model}")) +
    theme_classic()
}

plot_seq_mean_attention("AntiBERTa2", data_path)
