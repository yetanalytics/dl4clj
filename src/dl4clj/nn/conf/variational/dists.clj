(ns dl4clj.nn.conf.variational.dists
  (:import [org.deeplearning4j.nn.conf.layers.variational ReconstructionDistribution]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shared fns from the ReconstructionDistribution interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn distribution-input-size
  "Get the number of distribution parameters for the given input data size.

  :data-size (int) Size of the data. i.e., nIn value
  :dist (distribution), the distribution for a variational layer"
  [& {:keys [dist data-size]}]
  (.distributionInputSize dist data-size))

(defn example-neg-log-probability
  "Calculate the negative log probability for each example individually

  :dist (distribution), the distribution for a variational layer
  :features (INDArray), input data to be modelled
  :pre-out-dist-params (INDArray), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist features pre-out-dist-params]}]
  (.exampleNegLogProbability dist features pre-out-dist-params))

(defn generate-at-mean
  "Generate a sample from P(x|z), where x = E[P(x|z)]
  i.e., return the mean value for the distribution

  :dist (distribution), the distribution for a variational layer
  :pre-out-dist-params (INDArray), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist pre-out-dist-params]}]
  (.generateAtMean dist pre-out-dist-params))

(defn generate-random
  "Randomly sample from P(x|z) using the specified distribution parameters

  :dist (distribution), the distribution for a variational layer
  :pre-out-dist-params (INDArray), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist pre-out-dist-params]}]
  (.generateRandom dist pre-out-dist-params))

(defn gradient
  "Calculate the gradient of the negative log probability with
  respect to the preOutDistributionParams

  :dist (distribution), the distribution for a variational layer
  :features (INDArray), input data to be modelled
  :pre-out-dist-params (INDArray), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist features pre-out-dist-params]}]
  (.gradient dist features pre-out-dist-params))

(defn has-loss-fn?
  "Does this reconstruction distribution has a standard neural network loss function
  (such as mean squared error, which is deterministic)
  or is it a standard VAE with a probabilistic reconstruction distribution?"
  [dist]
  (.hasLossFunction dist))

(defn neg-log-probability
  "Calculate the negative log probability
  (summed or averaged over each example in the minibatch)

  :dist (distribution), the distribution for a variational layer
  :features (INDArray), input data to be modelled
  :pre-out-dist-params (INDArray), Distribution parameters used by :dist
   - before applying activation fn
  :average? (boolean), Whether the log probability should be averaged over the minibatch, or simply summed."
  [& {:keys [dist features pre-out-dist-params average?]}]
  (.negLogProbability dist features pre-out-dist-params average?))
