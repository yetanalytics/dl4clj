(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:import [org.deeplearning4j.nn.api Layer]))


;; still to implement:
;; INDArray	activate()
;; Trigger an activation with the last specified input
;; INDArray	activate(boolean training)
;; Trigger an activation with the last specified input
;; INDArray	activate(INDArray input)
;; Initialize the layer with the given input and return the activation for this layer given this input
;; INDArray	activate(INDArray input, boolean training)
;; Initialize the layer with the given input and return the activation for this layer given this input
;; INDArray	activate(INDArray input, Layer.TrainingMode training)
;; Initialize the layer with the given input and return the activation for this layer given this input
;; INDArray	activate(Layer.TrainingMode training)
;; Trigger an activation with the last specified input
;; INDArray	activationMean()
;; Calculate the mean representation for the activation for this layer
;; Pair<Gradient,INDArray>	backpropGradient(INDArray epsilon)
;; Calculate the gradient relative to the error in the next layer
;; Gradient	calcGradient(Gradient layerError, INDArray indArray)
;; Calculate the gradient
;; double	calcL1()
;; Calculate the l1 regularization term
;; 0.0 if regularization is not used.
;; double	calcL2()
;; Calculate the l2 regularization term
;; 0.0 if regularization is not used.
;; Layer	clone()
;; Clone the layer
;; INDArray	derivativeActivation(INDArray input)
;; Take the derivative of the given input based on the activation
;; Gradient	error(INDArray input)
;; Calculate error with respect to the current layer.
;; int	getIndex()
;; Get the layer index.
;; int	getInputMiniBatchSize()
;; Get current/last input mini-batch size, as set by setInputMiniBatchSize(int)
;; java.util.Collection<IterationListener>	getListeners()
;; Get the iteration listeners for this layer.
;; void	merge(Layer layer, int batchSize)
;; Parameter averaging
;; INDArray	preOutput(INDArray x)
;; Raw activations
;; INDArray	preOutput(INDArray x, boolean training)
;; Raw activations
;; INDArray	preOutput(INDArray x, Layer.TrainingMode training)
;; Raw activations
;; void	setIndex(int index)
;; Set the layer index.
;; void	setInput(INDArray input)
;; Get the layer input.
;; void	setInputMiniBatchSize(int size)
;; Set current/last input mini-batch size.
;; Used for score and gradient calculations.
;; void	setListeners(java.util.Collection<IterationListener> listeners)
;; Set the iteration listeners for this layer.
;; void	setListeners(IterationListener... listeners)
;; Set the iteration listeners for this layer.
;; void	setMaskArray(INDArray maskArray) 
;; Layer	transpose()
;; Return a transposed copy of the weights/bias (this means reverse the number of inputs and outputs on the weights)
;; Layer.Type	type()
;; Returns the layer type
;; void	update(Gradient gradient)
;; Update layer weights and biases with gradient change
;; void	update(INDArray gradient, java.lang.String paramType)
;; Update layer weights and biases with gradient change
