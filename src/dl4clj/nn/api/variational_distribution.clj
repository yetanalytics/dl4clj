(ns ^{:doc "fucntions from the Reconstruction Distribution class in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/variational/ReconstructionDistribution.html"}
    dl4clj.nn.api.variational-distribution
  (:import [org.deeplearning4j.nn.conf.layers.variational ReconstructionDistribution
            CompositeReconstructionDistribution])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]
            [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shared fns from the ReconstructionDistribution interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn distribution-input-size
  "Get the number of distribution parameters for the given input data size.

  :data-size (int) Size of the data. i.e., nIn value

  :dist (distribution), the distribution for a variational layer"
  [& {:keys [dist data-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :data-size (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.distributionInputSize ~dist (int ~data-size)))
         :else
         (let [[d d-s] (eval-if-code [dist seq?] [data-size seq? number?])]
           (.distributionInputSize d d-s))))

(defn example-neg-log-probability
  "Calculate the negative log probability for each example individually

  :dist (distribution), the distribution for a variational layer

  :features (INDArray or vec), input data to be modelled

  :pre-out-dist-params (INDArray or vec), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist features pre-out-dist-params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :pre-out-dist-params (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.exampleNegLogProbability ~dist
                                     (vec-or-matrix->indarray ~features)
                                     (vec-or-matrix->indarray ~pre-out-dist-params)))
         :else
         (let [[d f p] (eval-if-code [dist seq?] [features seq?]
                                     [pre-out-dist-params seq?])]
          (.exampleNegLogProbability d (vec-or-matrix->indarray f) (vec-or-matrix->indarray p)))))

(defn generate-at-mean
  "Generate a sample from P(x|z), where x = E[P(x|z)]
  i.e., return the mean value for the distribution

  :dist (distribution), the distribution for a variational layer

  :pre-out-dist-params (INDArray or vec), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist pre-out-dist-params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :pre-out-dist-params (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.generateAtMean ~dist (vec-or-matrix->indarray ~pre-out-dist-params)))
         :else
         (let [[d p] (eval-if-code [dist seq?] [pre-out-dist-params seq?])]
           (.generateAtMean d (vec-or-matrix->indarray p)))))

(defn generate-random
  "Randomly sample from P(x|z) using the specified distribution parameters

  :dist (distribution), the distribution for a variational layer

  :pre-out-dist-params (INDArray or vec), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist pre-out-dist-params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :pre-out-dist-params (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.generateRandom ~dist (vec-or-matrix->indarray ~pre-out-dist-params)))
         :else
         (let [[d p] (eval-if-code [dist seq?]
                                   [pre-out-dist-params seq?])]
           (.generateRandom d (vec-or-matrix->indarray p)))))

(defn gradient
  "Calculate the gradient of the negative log probability with
  respect to the preOutDistributionParams

  :dist (distribution), the distribution for a variational layer

  :features (INDArray or vec), input data to be modelled

  :pre-out-dist-params (INDArray or vec), Distribution parameters used by :dist
   - before applying activation fn"
  [& {:keys [dist features pre-out-dist-params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :pre-out-dist-params (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.gradient ~dist (vec-or-matrix->indarray ~features)
                     (vec-or-matrix->indarray ~pre-out-dist-params)))
         :else
         (let [[d f p] (eval-if-code [dist seq?] [features seq?]
                                     [pre-out-dist-params seq?])]
           (.gradient d (vec-or-matrix->indarray f)
                      (vec-or-matrix->indarray p)))))

(defn has-loss-fn?
  "Does this reconstruction distribution has a standard neural network loss function
  (such as mean squared error, which is deterministic)
  or is it a standard VAE with a probabilistic reconstruction distribution?"
  [dist & {:keys [as-code?]
           :or {as-code? true}}]
  (match [dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasLossFunction ~dist))
         :else
         (.hasLossFunction dist)))

(defn neg-log-probability
  "Calculate the negative log probability
  (summed or averaged over each example in the minibatch)

  :dist (distribution), the distribution for a variational layer

  :features (INDArray or vec), input data to be modelled

  :pre-out-dist-params (INDArray or vec), Distribution parameters used by :dist
   - before applying activation fn

  :average? (boolean), Whether the log probability should be averaged over the minibatch, or simply summed."
  [& {:keys [dist features pre-out-dist-params average? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :pre-out-dist-params (:or (_ :guard vector?)
                                     (_ :guard seq?))
           :average? (:or (_ :guard boolean?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.negLogProbability ~dist (vec-or-matrix->indarray ~features)
                              (vec-or-matrix->indarray ~pre-out-dist-params) ~average?))
         :else
         (let [[d f p a?] (eval-if-code [dist seq?] [features seq?]
                                        [pre-out-dist-params seq?]
                                        [average? seq? boolean?])]
           (.negLogProbability d (vec-or-matrix->indarray f) (vec-or-matrix->indarray p) a?))))

(defn compute-loss-fn-score-array
  "computes the loss function score.
  only works with composite reconstruction distributions

  :composite-dist (distribution), a composite distribution with a loss fn

  :features (INDArray or vec), the input data

  :reconstruction (INDArray or vec), the output of a variational model"
  [& {:keys [composite-dist features reconstruction as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:composite-dist (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :reconstruction (:or (_ :guard vector?)
                                (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.computeLossFunctionScoreArray ~composite-dist (vec-or-matrix->indarray ~features)
                                          (vec-or-matrix->indarray ~reconstruction)))
         :else
         (let [[d f r] (eval-if-code [composite-dist seq?] [features seq?]
                                     [reconstruction seq?])]
          (.computeLossFunctionScoreArray d (vec-or-matrix->indarray f) (vec-or-matrix->indarray r)))))
