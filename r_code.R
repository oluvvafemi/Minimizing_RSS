library(ggplot2) # data visualization package

# Simulation in R
simulate_linear_regression_data <- function(m, n){
    #-------Input--------
    # m- number of samples
    # n - number of predictors/features
    #------Return--------
    # X - matrix m*n 
    # y - vector size - m
    # beta - vector size - n
    
    # Note: a column of "ones" was added to last column of X for convenience
    set.seed(1)
    X <- matrix(nrow = m, ncol =n-1)
    beta <- matrix(nrow = n)
    
    for (i in 1:(n-1)) {
        mu <- runif(1, 0, 50)
        std <- rnorm(1,3,0.1)
        X[, i] <- rnorm(m, mu, std)
        beta[i,] <- rnorm(1, mu, std)
    }
    X <- cbind(X, rep(1, m))
    beta[length(beta)] <- rnorm(1)
    
    y <- X %*% beta
    
    return( list(X, y, beta))
}

# closed-form in R
estimate_beta_LR <- function(X, y){
    
    # function for closed-form estimation of beta
    # solve(a, b)  for matrix multiplication of inverse(a) *  b
    return(solve( t(X) %*% X,
                  t(X) %*% y)
    )
}



# Gradient Descent in R
calc_rss <- function(beta, x, y){
    # function to calculate RSS
    return( t(y- (x%*%beta)) %*% 
                (y - (x %*%beta)) 
    )
}

oracle <- function(beta, x, y){
    # Oracle computes and returns the gradient of RSS 
    return( (-2 *  t(x)) %*% (y - (x %*% beta) )
            
    )
}

gd <- function(x, y, maxit, step_size){
    # -----Input-----
    # x - matrix m * n
    # y - vector of target values
    # maxit - number of iterations
    # step_size - step-size t
    # 
    # -----Return-----
    # beta - the gradient descent estimate of beta
    # rss_history - the rss computed at each iteration of gd
    set.seed(2)
    
    beta = rnorm(dim(X)[2]) #initial guess
    rss_history <- rep(0, maxit)
    for (i in 1:maxit) {
        rss_history[i] <- calc_rss(beta, x, y)
        beta = beta - (step_size * oracle(beta, x, y))
        
    }
    iteration <- 1:maxit
    plot_df <- data.frame(iteration, rss_history)
    
    return( list(beta, plot_df))
}

# assessment
calc_tss <- function(y){
    return ( t(y - mean(y)) %*%
                 (y - mean(y))
    )
}

rse_r2 <- function(beta, x,y){
    rss <- calc_rss(beta, x, y)
    rse <- sqrt(rss/length(y)-dim(x)[2] - 1)
    tss <- calc_tss(y)
    r2 <- 1 - (rss/tss)
    
    return( list(rse, r2))
}

# diagnostics
plot_residuals <- function(x, y, beta){
    y_hat <- x%*%beta
    e <- y - y_hat
    if (dim(X)[2] >1){
        e_df <- data.frame(y_hat, e)
        ggplot(e_df, aes(x=y_hat, y=e)) +
            geom_point() +
            geom_smooth()
    }
    else{
        e_df <- data.frame(x, e)
        ggplot(e_df, aes(x=x, y=e)) +
            geom_point() +
            geom_smooth()
    }
}