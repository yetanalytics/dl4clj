(ns ^{:doc "Often times transfer learning models have frozen layers where parameters are held constant during training For ease of training and quick turn around times, the dataset to be trained on can be featurized and saved to disk.

 Featurizing in this case refers to conducting a forward pass on the network and saving the activations from the output of the frozen layers.

 During training the forward pass and the backward pass through the frozen layers can be skipped entirely and the featurized dataset can be fit with the smaller unfrozen part of the computation graph which allows for quicker iterations.

 The class internally traverses the computation graph/MLN and builds an instance of the computation graph/MLN that is equivalent to the unfrozen subset.

 Currently, computation graphs are not supported by dl4clj

 see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/TransferLearningHelper.html" }
    dl4clj.nn.transfer-learning.helper
  (:import [org.deeplearning4j.nn.transferlearning TransferLearningHelper]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [dl4clj.helpers :refer [reset-iterator!]]
            [dl4clj.utils :refer [array-of obj-or-code?]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]]))

(defn new-helper
  "creates a new instance of TransferLearningHelper with the supplied opts

  :mln (multi-layer-network) a model with multiple layers
   - see: dl4clj.nn.conf.builders.multi-layer-builders

  :frozen-til (int), indicates the index of the layer and below to freeze

  if :frozen-til is supplied, Will modify the given MLN (in place!
  to freeze layers (hold params constant during training) specified and below
  otherwise expects a mln where some layers are already frozen"
  [& {:keys [mln frozen-til as-code?]
      :or {as-code? true}
      :as opts}]
  ;; update this to use match
  (let [code (if frozen-til
               `(TransferLearningHelper. ~mln ~frozen-til)
               `(TransferLearningHelper. ~mln))]
    (obj-or-code? as-code? code)))
