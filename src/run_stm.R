library(stm)
library(ggplot2)
library(optparse)
library(arrow)
library(quanteda)

main <- function() {
    now_timestamp <- Sys.time()
    
    option_list <- list(
        make_option(
            c("-k", "--num_topics"),
            type = "integer",
            help = "Number of topics to train the model for"),
        make_option(
            c("-d", "--dataset"),
            type = "character",
            help = "Dataset to use. Either 'fed', 'news' or 'dummy'"))
    
    opt <- parse_args(OptionParser(option_list = option_list))
    
    output_dir <- create_output_dir(now_timestamp, opt$dataset, opt$num_topics)
    model_path <- sprintf(
        "models/stm_%s_%s_topics_%02d.rds",
        opt$dataset,
        format(now_timestamp, "%Y-%m-%d_%H-%M-%S"),
        opt$num_topics)
    
    valid_datasets <- c("fed", "news", "dummy")
    if (!(opt$dataset %in% valid_datasets)) {
        stop("Invalid dataset. Choose one of: fed, news, dummy.")
    }
    
    dataset_path <- switch(
        opt$dataset,
        fed = "data/processed/communications_final.parquet",
        news = "data/processed/news_final.parquet",
        dummy = "data/processed/dummy_news.parquet"
    )
    
    data <- read_parquet(dataset_path)
    stm_input <- prepare_data(data)
    
    model <- stm(documents = stm_input$documents,
                 vocab = stm_input$vocab,
                 data = stm_input$meta,
                 K = opt$num_topics,
                 init.type = "Spectral")
    
    saveRDS(model, file = model_path)
    
    # Saves topics as texts
    label_topics <- labelTopics(model, n = 10)
    label_topics_path = paste(output_dir, "topic_labels.txt", sep="/")
    sink(label_topics_path)
    print(label_topics)
    sink()
    
    # Saves topic distributions as parquet
    topic_dist <- get_topic_dist(model)
    topic_dist_path <- paste(output_dir, "topic_dist.parquet", sep="/")
    write_parquet(topic_dist, topic_dist_path)
    
    # Saves theta, the distribution of topics per document
    theta_path <- topic_dist_path <- paste(output_dir, "theta.parquet", sep="/")
    theta <- as.data.frame(model$theta)
    colnames(theta) <- paste0("Topic_", seq_len(ncol(theta)))
    write_parquet(theta, theta_path)
}

create_output_dir <- function(now_timestamp, dataset, n_topics) {
    dir_name <- sprintf(
        "stm_%s_%s_topics_%02d",
        dataset,
        format(now_timestamp, "%Y-%m-%d_%H-%M-%S"),
        n_topics)
    
    output_dir <- here::here("output", dir_name)
    
    dir.create(output_dir, recursive = TRUE)
    
    return(output_dir)
}

# Converts dataset into format usable by stm
prepare_data <- function(data) {
    corp <- corpus(data, text_field = "text")
    corp_tokens <- tokens(corp, what = "word", remove_punct = FALSE)
    dfm <- dfm(corp_tokens)
    stm_input <- convert(dfm, to = "stm")
    stm_input$meta <- docvars(corp)
    return(stm_input)
}

# Gets the word distribution for each topic
get_topic_dist <- function(model) {
    log_probs <- as.data.frame(model$beta$logbeta[[1]])
    colnames(log_probs) <- model$vocab
    rownames(log_probs) <- paste0("Topic_", seq_len(nrow(log_probs)))
    return(log_probs)
}

if (sys.nframe() == 0) {
    main()
}
