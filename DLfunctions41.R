

# Conmpute the Relu of a vector
relu   <-function(Z){
    A <- apply(Z, 1:2, function(x) max(0,x))
    cache<-Z
    retvals <- list("A"=A,"Z"=Z)
    return(retvals)
}



# Conmpute the softmax of a vector
softmax   <- function(Z){
    # get unnormalized probabilities
    exp_scores = exp(t(Z))
    # normalize them for each example
    A = exp_scores / rowSums(exp_scores)   
    retvals <- list("A"=A,"Z"=Z)
    return(retvals)
}

# Compute the detivative of Relu 
reluDerivative   <-function(dA, cache){
    Z <- cache
    dZ <- dA
    # Create a logical matrix of values > 0
    a <- Z > 0
    # When z <= 0, you should set dz to 0 as well. Perform an element wise multiple
    dZ <- dZ * a
    return(dZ)
}


softmaxDerivative    <- function(dA, cache ,y,numTraining){
    # Note : dA not used. dL/dZ = dL/dA * dA/dZ = pi-yi
    Z <- cache 
    # Compute softmax
    exp_scores = exp(t(Z))
    # normalize them for each example
    probs = exp_scores / rowSums(exp_scores)
    # Get the number of 0, 1 and 2 classes and store in a,b,c
    a=sum(y==0)
    b=sum(y==1)
    c=sum(y==2)
    # Create a yi matrix based on yi for each class
    m= matrix(rep(c(1,0,0),a),nrow=a,ncol=3,byrow=T)
    n= matrix(rep(c(0,1,0),b),nrow=b,ncol=3,byrow=T)
    o= matrix(rep(c(0,0,1),c),nrow=c,ncol=3,byrow=T)
    # Stack them vertically
    yi=rbind(m,n,o)

    dZ = probs-yi
    return(dZ)
}

# Initialize the model 
# Input : number of features
#         number of hidden units
#         number of units in output
# Returns: Weight and bias matrices and vectors
initializeModel <- function(numFeats,numHidden,numOutput){
    set.seed(2)
    a<-rnorm(numHidden*numFeats)*0.01 #  Multiply by .01 
    W1 <- matrix(a,nrow=numHidden,ncol=numFeats)
    a<-rnorm(numHidden*1)
    b1 <- matrix(a,nrow=numHidden,ncol=1)
    a<-rnorm(numOutput*numHidden)*0.01
    W2 <- matrix(a,nrow=numOutput,ncol=numHidden)
    a<-rnorm(numOutput*1)
    b2 <- matrix(a,nrow=numOutput,ncol=1)
    parameters <- list("W1"=W1,"b1"=b1,"W2"=W2,"b2"=b2)
    return(parameters)
    
}




# Compute the activation at a layer 'l' for forward prop in a Deep Network
# Input : A_prec - Activation of previous layer
#         W,b - Weight and bias matrices and vectors
#         activationFunc - Activation function - sigmoid, tanh, relu etc
# Returns : The Activation of this layer
#         : 
# Z = W * X + b
# A = sigmoid(Z), A= Relu(Z), A= tanh(Z)
layerActivationForward <- function(A_prev, W, b, activationFunc){
    
    # Compute Z
    z = W %*% A_prev
    Z <-sweep(z,1,b,'+')
    
    forward_cache <- list("A_prev"=A_prev, "W"=W, "b"=b) 
    # Compute the activation for sigmoid
    if(activationFunc == "sigmoid"){
        vals = sigmoid(Z)  
    } else if (activationFunc == "relu"){ # Compute the activation for relu
        vals = relu(Z)
    } else if(activationFunc == 'tanh'){ # Compute the activation for tanh
        vals = tanhActivation(Z) 
    } else if(activationFunc == 'softmax'){
        vals = softmax(Z)
    }
    
    cache <- list("forward_cache"=forward_cache, "activation_cache"=vals[['Z']])
    retvals <- list("A"=vals[['A']],"cache"=cache)
    return(retvals)
}



# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
layerActivationBackward  <- function(dA, cache, y, activationFunc){
    # Get A_prev,W,b
    forward_cache <-cache[['forward_cache']]
    activation_cache <- cache[['activation_cache']]
    A_prev <- forward_cache[['A_prev']]
    numtraining = dim(A_prev)[2]
    # Get Z

    if(activationFunc == "relu"){
        dZ <- reluDerivative(dA, activation_cache)  
    } else if(activationFunc == "sigmoid"){
        dZ <- sigmoidDerivative(dA, activation_cache)      
    } else if(activationFunc == "tanh"){
        dZ <- tanhDerivative(dA, activation_cache)
    } else if(activationFunc == "softmax"){
        dZ <- softmaxDerivative(dA,  activation_cache,y,numtraining)
    }
    #print(dZ)
    if (activationFunc == 'softmax'){
        W <- forward_cache[['W']]
        b <- forward_cache[['b']]
        dW = 1/numtraining * A_prev%*%dZ
        db = 1/numtraining* matrix(colSums(dZ),nrow=1,ncol=3)
        dA_prev = dZ %*%W
    } else {
        W <- forward_cache[['W']]
        b <- forward_cache[['b']]
        numtraining = dim(A_prev)[2]
        dW = 1/numtraining * dZ %*% t(A_prev)
        db = 1/numtraining * rowSums(dZ)
        dA_prev = t(W) %*% dZ
    }
    retvals <- list("dA_prev"=dA_prev,"dW"=dW,"db"=db)
    return(retvals)
}




# Plot a decision boundary
# This function uses ggplot2
plotDecisionBoundary <- function(Z,W1,b1,W2,b2){
    xmin<-min(Z[,1])
    xmax<-max(Z[,1])
    ymin<-min(Z[,2])
    ymax<-max(Z[,2])
    
    # Create a grid of points
    a=seq(xmin,xmax,length=100)
    b=seq(ymin,ymax,length=100)
    grid <- expand.grid(x=a, y=b)
    colnames(grid) <- c('x1', 'x2')
    grid1 <-t(grid)
    
    
    # Predict the output based on the grid of points
    retvals <- layerActivationForward(grid1,W1,b1,'relu')
    A1 <- retvals[['A']]
    cache1 <- retvals[['cache']]
    forward_cache1 <- cache1[['forward_cache1']]
    activation_cache <- cache1[['activation_cache']]
    
    retvals = layerActivationForward(A1,W2,b2,'softmax')
    A2 <- retvals[['A']]
    cache2 <- retvals[['cache']]
    forward_cache2 <- cache2[['forward_cache1']]
    activation_cache2 <- cache2[['activation_cache']]
    
    # From the  softmax probabilities pick the one with the highest probability
    q= apply(A2,1,which.max)
    
    
    ###
    
    
    q1 <- t(data.frame(q))
    q2 <- as.numeric(q1)
    grid2 <- cbind(grid,q2)
    colnames(grid2) <- c('x1', 'x2','q2')
    
    z1 <- data.frame(Z)
    names(z1) <- c("x1","x2","y")
    atitle=paste("Decision boundary")
    ggplot(z1) + 
        geom_point(data = z1, aes(x = x1, y = x2, color = y)) +
        stat_contour(data = grid2, aes(x = x1, y = x2, z = q2,color=q2), alpha = 0.9)+
        ggtitle(atitle) + scale_colour_gradientn(colours = brewer.pal(10, "Spectral"))
}

# Predict the output
computeScores <- function(parameters, X,hiddenActivationFunc='relu'){
    
    fwdProp <- forwardPropagationDeep(X, parameters,hiddenActivationFunc)
    scores <- fwdProp$AL
    
    return (scores)
}
