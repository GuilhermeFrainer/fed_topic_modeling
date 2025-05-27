library(stm)
library(ggplot2)
library(optparse)

main <- function() {
  timestamp <- Sys.time()
  
  option_list <- list(
    make_option(
      "--max_topics",
      type = "integer",
      default = 30,
      help = "Highest number of topics with which to train model"),
    make_option(
      "--min_topics",
      type = "integer",
      default = 5,
      help = "Lowest number of topics with which to train model"))
  
  opt <- parse_args(OptionParser(option_list = option_list))
  
  
  data <- read.csv("data/communications_preprocessed.csv", stringsAsFactors = FALSE)
  
  processed_data <- prepare_data(data)
  docs <- processed_data$documents
  vocab <- processed_data$vocab
  meta <- processed_data$meta
  
  modelSearchResults <- searchK(
    documents = docs,
    vocab = vocab,
    data = meta,
    K = seq.int(opt$min_topics, opt$max_topics, 1)
  )
  
  p <- generate_plot(modelSearchResults$results)
  
  df <- correct_df(modelSearchResults$results)
  
  dirs <- create_dirs(timestamp)
  
  output_filename = paste(dirs$output, "searchK_results.csv", sep="/")
  write.csv(df, output_filename, row.names = FALSE)
  message("Model search output saved to", output_filename)
  
  figure_filename = paste(dirs$figures, "stm_model_comparison.png", sep="/")
  ggsave(
    figure_filename,
    plot = p,
    width = 8,
    height = 6,
    dpi = 300)
  message("Model comparison plot saved to", figure_filename)
}

prepare_data <- function(data) {
  # No preprocessing, as text was already preprocessed in Python
  processed <- textProcessor(
    data$stemmed_text,
    lowercase = FALSE,
    removestopwords = FALSE,
    removenumbers = FALSE,
    removepunctuation = FALSE,
    stem = FALSE,
    wordLengths = c(1, Inf))
  out <- prepDocuments(
    processed$documents,
    processed$vocab,
    processed$meta,
    lower.thresh = 0) # Doesn't remove uncommon words
  return(out)
}

generate_plot <- function(results) {
  df <- results
  df$semcoh <- as.numeric(df$semcoh)
  df$exclus <- as.numeric(df$exclus)
  
  # Get axis midpoints
  x_min <- min(df$semcoh, na.rm = TRUE)
  x_max <- max(df$semcoh, na.rm = TRUE)
  x_mid <- (x_min + x_max) / 2
  
  y_min <- min(df$exclus, na.rm = TRUE)
  y_max <- max(df$exclus, na.rm = TRUE)
  y_mid <- (y_min + y_max) / 2
  
  p <- ggplot(df, aes(x = semcoh, y = exclus, label = K)) +
    geom_point() +
    geom_text(vjust = -0.5) +  # or use ggrepel later
    
    # Quadrant lines at axis midpoints
    geom_vline(xintercept = x_mid, linetype = "dashed", color = "gray30") +
    geom_hline(yintercept = y_mid, linetype = "dashed", color = "gray30") +
    
    labs(
      x = "Semantic Coherence",
      y = "Exclusivity"
    ) +
    theme_minimal() + 
    theme(
      panel.grid = element_blank(),         # remove default grid
      panel.border = element_blank(),       # remove border
      axis.line = element_line(color = "black"),  # add clean axes
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
  )
  return(p)
}

create_dirs <- function(timestamp) {
  dir_name <- format(timestamp, "stm_fed_%Y-%m-%d_%H-%M-%S")
  output_dir <- here::here("output", dir_name)
  fig_dir <- here::here("figures", dir_name)
  
  dir.create(output_dir, recursive = TRUE)
  dir.create(fig_dir, recursive = TRUE)
  
  return(list(output = output_dir, figures = fig_dir))
}

# For whatever reason, some of the columns in the dataframe
# seem to be of type "list", which doesn't allow me to save as a csv
# All this does is correct that
correct_df <- function(df) {
  list_cols <- sapply(df, is.list)
  df[list_cols] <- lapply(df[list_cols], function(col) {
    sapply(col, function(x) if(length(x) > 0) as.numeric(x[[1]]) else NA)
  })
  return(df)
}

if (sys.nframe() == 0) {
  main()
}